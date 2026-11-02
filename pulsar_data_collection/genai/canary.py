"""Canary set: fixed prompts answered several times, to check later that the endpoint still serves the same model.

The canary prompts are synthetic and fixed; their answers are not user data and are kept (a JSON file per run) so that
`pulsar_metrics.genai.model_equality` can compare a period with the reference.
"""

import json
import time
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
]


class CanaryProbe:
    """Ask each canary prompt `samples` times (at a temperature above zero) and collect the answers

    Parameters
    ----------
    prompts : sequence of str, optional
        The canary set (default: 12 public-services prompts in French and English); keep it fixed over time
    samples : int, optional
        Answers per prompt (default 5)
    """

    def __init__(self, prompts: Optional[Sequence[str]] = None, samples: int = 5):
        self.prompts = list(prompts or DEFAULT_PROMPTS)
        self.samples = samples

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
