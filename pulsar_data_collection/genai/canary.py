"""Canary set: fixed prompts answered several times, to check later that the endpoint still serves the same model.

The canary prompts are synthetic and fixed; their answers are not user data and are kept (a JSON file per run) so that
`pulsar_metrics.genai.model_equality` can compare a period with the reference.

Sample size. A change of weights or of model family shows with a few dozen answers, a change of **quantisation** does
not: in the lab (llama3.2 3B and llama3.1 8B, q4_K_M against q8_0 or fp16), 12 prompts x 5 answers (60) detected it in
20 to 40 % of the runs, 24 x 5 in 40 %, 48 x 5 in 60 % and 48 x 10 (480 answers) in 5 runs out of 5 for q4_K_M against
q8_0. These figures come from few, correlated draws: treat 480 as an order of magnitude, not a guarantee, and check it
on your own model (`required_answers`). Hence the presets: "quick" (12 x 5), "standard" (48 x 10, the default since
0.10) and "thorough" (48 x 20).
"""

import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

DEFAULT_PROMPTS = [
    "Explain in two sentences what a rent assistance program is.",
    "Expliquez en deux phrases ce qu'est une subvention à la rénovation.",
    "List three documents usually needed to apply for child care benefits.",
    "Quelles sont trois raisons courantes de refus d'une demande d'aide ?",
    "Write a short polite reply to someone asking about their application status.",
    "Rédigez une courte réponse polie à une personne qui demande l'état de son dossier.",
    "What is the difference between a grant and a loan? Answer briefly.",
    "Donnez un exemple de calcul d'un montant d'aide selon le revenu.",
    "Summarize in one sentence why income limits exist in public programs.",
    "Expliquez en une phrase pourquoi une authentification est demandée avant d'ouvrir un dossier.",
    "Name two ways a citizen can contact a public service office.",
    "Décrivez en deux phrases la formation professionnelle subventionnée.",
    # since 0.10: 36 more prompts (the first 12 are unchanged, so that older canary files stay comparable)
    "Explain in three sentences how a property tax credit works.",
    "Expliquez en trois phrases le fonctionnement d'un crédit d'impôt foncier.",
    "Give two examples of proof of residence accepted by a public office.",
    "Donnez deux exemples de preuves de résidence acceptées par un bureau public.",
    "Write a short message telling a citizen that a document is missing from their file.",
    "Rédigez un court message informant une citoyenne qu'il manque un document à son dossier.",
    "What is a waiting list for social housing? Answer in two sentences.",
    "Qu'est-ce qu'une liste d'attente pour un logement social ? Répondez en deux phrases.",
    "List four steps to renew a driving permit.",
    "Énumérez quatre étapes pour renouveler un permis de conduire.",
    "Explain the difference between a refundable and a non-refundable tax credit.",
    "Expliquez la différence entre un crédit d'impôt remboursable et non remboursable.",
    "Describe in one paragraph how to appeal an administrative decision.",
    "Décrivez en un paragraphe comment contester une décision administrative.",
    "What should a person do if they lose their health insurance card?",
    "Que doit faire une personne qui perd sa carte d'assurance maladie ?",
    "Summarize in two sentences the purpose of a parental leave benefit.",
    "Résumez en deux phrases l'objectif d'une prestation de congé parental.",
    "Give an example of a monthly budget for a family of four with a modest income.",
    "Donnez un exemple de budget mensuel pour une famille de quatre au revenu modeste.",
    "Write a polite email asking for an appointment at a service centre.",
    "Rédigez un courriel poli pour demander un rendez-vous dans un centre de services.",
    "Explain why some programs ask for the previous year's tax return.",
    "Expliquez pourquoi certains programmes demandent la déclaration de revenus de l'année précédente.",
    "List three common mistakes when filling in an application form.",
    "Énumérez trois erreurs courantes en remplissant un formulaire de demande.",
    "What is a means test? Answer in two sentences.",
    "Qu'est-ce qu'un critère de ressources ? Répondez en deux phrases.",
    "Describe how a citizen can follow the progress of an online application.",
    "Décrivez comment une personne peut suivre l'avancement d'une demande en ligne.",
    "Compute the yearly total of a benefit of 245 dollars paid every two weeks, and show the steps.",
    "Calculez le total annuel d'une prestation de 245 dollars versée toutes les deux semaines, en montrant les étapes.",
    "Explain in simple words what personal information protection means for a public service.",
    "Expliquez simplement ce que signifie la protection des renseignements personnels pour un service public.",
    "Write three short tips for preparing a meeting with a social worker.",
    "Rédigez trois conseils courts pour préparer une rencontre avec une travailleuse sociale.",
]

PRESETS = {"quick": (12, 5), "standard": (48, 10), "thorough": (48, 20)}  # (prompts, answers per prompt)

# lab: share of canary runs that detected q4_K_M against q8_0 (llama3.2 3B), by number of answers
QUANTISATION_DETECTION = {60: 0.2, 120: 0.4, 240: 0.6, 480: 1.0}


def required_answers(change: str = "quantisation") -> int:
    """Order of magnitude of the canary answers (prompts x samples) needed to see a change of the served model

    "weights" (another model or fine-tune): 60; "quantisation" (same weights, other precision): 480, from the lab,
    with few correlated draws (see the module docstring)
    """
    sizes = {"weights": 60, "quantisation": 480, "quantization": 480}
    if change not in sizes:
        raise ValueError(f"change must be one of {sorted(sizes)}")
    return sizes[change]


class CanaryProbe:
    """Ask each canary prompt `samples` times (at a temperature above zero) and collect the answers

    Parameters
    ----------
    prompts : sequence of str, optional
        The canary set (default: the 48 public-services prompts of `DEFAULT_PROMPTS`, French and English); keep it
        fixed over time
    samples : int, optional
        Answers per prompt (default 10; 5 before 0.10). Fewer than `required_answers("quantisation")` answers in all
        warn that a quantisation change will likely go unseen (`warn=False` to silence)

    `CanaryProbe.preset("quick")` gives the former 12 x 5 set, enough for a change of weights only. The answers of a
    sample round are asked one prompt after the other (all prompts, then again), so that the repetitions of a prompt
    are spread in time rather than drawn back to back.
    """

    def __init__(self, prompts: Optional[Sequence[str]] = None, samples: int = 10, warn: bool = True):
        if samples < 1:
            raise ValueError("samples must be at least 1")
        self.prompts = list(prompts or DEFAULT_PROMPTS)
        self.samples = samples
        self.failures = 0
        if warn and self.planned_answers < required_answers("quantisation"):
            warnings.warn(
                f"canary of {len(self.prompts)} prompts x {samples} answers = {self.planned_answers}: a quantisation change "
                f"needs about {required_answers('quantisation')} answers to be seen (lab); enough for a change of weights",
                stacklevel=2,
            )

    @classmethod
    def preset(cls, name: str = "standard") -> "CanaryProbe":
        """ "quick" (12 x 5: changes of weights), "standard" (48 x 10), "thorough" (48 x 20)"""
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")
        n_prompts, samples = PRESETS[name]
        return cls(DEFAULT_PROMPTS[:n_prompts], samples, warn=False)

    @property
    def planned_answers(self) -> int:
        """Answers asked by `run` (prompts x samples)"""
        return len(self.prompts) * self.samples

    def run(self, ask: Callable[[str], str], progress: Optional[Callable[[int, int], None]] = None) -> dict:
        """`ask(prompt) -> answer`; returns {prompt: [answers]}; failed calls are skipped and counted (`failures`)"""
        answers, total, done = {p: [] for p in self.prompts}, len(self.prompts) * self.samples, 0
        self.failures = 0
        for _ in range(self.samples):
            for prompt in self.prompts:
                try:
                    answers[prompt].append(ask(prompt))
                except Exception:  # noqa: BLE001 - a failed canary call is counted and skipped
                    self.failures += 1
                done += 1
                if progress:
                    progress(done, total)
        return answers

    @staticmethod
    def save(answers: dict, path, **metadata) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "metadata": metadata, "answers": answers}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    @staticmethod
    def load(path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))["answers"]


def timed(ask: Callable[[str], str]) -> Callable[[str], str]:
    """Wrap `ask` to retry once after a short pause (local endpoints sometimes drop the first call)"""

    def wrapper(prompt: str) -> str:
        try:
            return ask(prompt)
        except Exception:  # noqa: BLE001
            time.sleep(1)
            return ask(prompt)

    return wrapper
