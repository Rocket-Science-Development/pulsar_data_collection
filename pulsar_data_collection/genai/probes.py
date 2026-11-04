"""Counterfactual probes for LLM applications and agents: paired requests that differ only by one attribute.

An audit of a language model can produce its own verdict: answers change with the order of the options, with the
wording, or at random. The probes therefore build the controls in:

- every set holds one request per group of the probed attribute (e.g. a first name typical of each group), everything
  else being identical;
- **placebo** sets vary an attribute that must not matter (e.g. the day the request is written): their flip rate is
  the noise floor of the instrument, the probed attribute is compared with it;
- comparative probes (choose between two profiles) are asked in **both orders**, with or without a tie option, to
  measure and cancel the position effect; the option labels can rotate across sets (A/B, 1/2, X/Y, also reversed),
  and a log-likelihood instrument (`ComparativeProbe.run_loglik`) compares the two profiles without asking anything
  when the target exposes log-probabilities;
- refusals are recorded as such (`answer_status="refusal"`), apart from the answers that could not be parsed;
- requests are sent in a random order and look like ordinary requests (nothing tells the model it is audited).

The collected rows are analysed with `pulsar_metrics.fairness.paired`. When the target is an instrumented agent, pass
`probe.span_attributes()` to its agent span: the `pulsar.cf.*` attributes then reach the `agent_run` records.

The value lists are the user's responsibility: lists of names standing for groups are sensitive and must be reviewed
(e.g. by a GBA Plus expert). The ones shipped with the demo are provisional and marked as such.
"""

import logging
import random  # reproducible sampling and ordering of the probe requests, never used for security
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from . import semconv as sc

log = logging.getLogger(__name__)

NO_CONTROL = "none"
PLACEBO = "placebo"

# Values of the `answer_status` column of ComparativeProbe.run / run_loglik: "ok" (a choice was read), "refusal" (the
# model declined to choose), "unparsed" (no choice could be read), "error" (the request failed). Refusals and unparsed
# answers are the unusable answers (column `unusable`); errors are counted apart (column `error`).
ANSWER_OK = "ok"
ANSWER_REFUSAL = "refusal"
ANSWER_UNPARSED = "unparsed"
ANSWER_ERROR = "error"
UNUSABLE_STATUSES = frozenset({ANSWER_REFUSAL, ANSWER_UNPARSED})

# Neutral label pairs rotated across the sets of a ComparativeProbe (labels="neutral"); the reversed pairs separate a
# preference for a label (always "A") from a preference for a position (always the first profile)
NEUTRAL_LABELS = (("A", "B"), ("1", "2"), ("X", "Y"), ("B", "A"), ("2", "1"), ("Y", "X"))
DEFAULT_LABELS = ("A", "B")

# Refusal phrasings, French and English, looked for at the start of an answer that gave no choice
_Q = "['\u2019]"
_REFUSAL = re.compile(
    r"^\W*(?:"
    r"je ne (?:peux|puis|pourrais|saurais|vais) pas|je ne peux|je ne suis pas en mesure|je suis (?:désolée?|navrée?)|"
    rf"désolée?|navrée?|je refuse|je préfère ne pas|il m{_Q}est impossible|je n{_Q}ai pas le droit|"
    rf"en tant qu{_Q}(?:ia|intelligence artificielle|assistant)|"
    rf"i can(?:not|{_Q}t| not)|i am sorry|i{_Q}m sorry|sorry|i (?:won{_Q}t|will not)|"
    rf"i(?:{_Q}m| am) (?:unable|not able)|i (?:must|have to) decline|i refuse|as an ai"
    r")",
    re.IGNORECASE,
)


def is_refusal(answer: Any) -> bool:
    """The answer starts with a refusal ("Je ne peux pas...", "Je suis désolé...", "I can't...", "I'm sorry...")"""
    return isinstance(answer, str) and bool(_REFUSAL.match(answer.strip()))


_LEADING = re.compile(r"^[\s«»\"'*:().\[\]-]+")
_TIE = re.compile(r"^(?:é|e)gal|^aucune|^les deux|^ni l|^tie\b|^equal|^neither|^both|^none", re.IGNORECASE)
_PREFIX = re.compile(r"^(?:la\s+)?(?:réponse|answer|phrase|option|choix|choice|profil|profile)\s*:?\s*", re.IGNORECASE)


def label_choice(answer: Any, labels: tuple = DEFAULT_LABELS) -> Optional[str]:
    """ "first", "second" or "tie" read from an answer that starts with one of the two labels (or a tie word), else None

    Leading quotes, asterisks and a prefix such as "Réponse :" are ignored: "« B »", "**1**", "Réponse : X." are read.
    """
    if not isinstance(answer, str):
        return None
    text = _LEADING.sub("", answer.strip())
    if _TIE.match(text):
        return "tie"
    text = _LEADING.sub("", _PREFIX.sub("", text, count=1))
    # the longer label first ("II" before "I")
    for label, position in sorted(zip(labels, ("first", "second")), key=lambda item: -len(item[0])):
        if re.match(re.escape(label) + r"(?!\w)", text, re.IGNORECASE):
            return position
    return None


@dataclass(frozen=True)
class Probe:
    """One request of a counterfactual set"""

    set_id: str
    variant: str  # group of the value used, e.g. "fr"
    attribute: str  # slot of the template that varies within the set
    value: str
    control: str  # "none" or "placebo"
    text: str
    context: dict = field(default_factory=dict, compare=False)
    # ComparativeProbe: texts of the first and second profiles as shown, and their labels (e.g. ("A", "B"))
    profiles: tuple = field(default=(), compare=False)
    labels: tuple = field(default=(), compare=False)

    def span_attributes(self) -> dict:
        return {
            sc.CF_SET_ID: self.set_id,
            sc.CF_VARIANT: self.variant,
            sc.CF_ATTRIBUTE: self.attribute,
            sc.CF_CONTROL: self.control,
        }


def _check_values(name: str, values: dict) -> None:
    if len(values) < 2 or any(not v for v in values.values()):
        raise ValueError(f"{name} needs at least two groups, each with at least one value")


class CounterfactualProbe:
    """Generate paired requests from templates, run them against a target and collect the outcomes

    Parameters
    ----------
    templates : sequence of str
        Request templates with a `{<attribute>}` slot, a `{<placebo attribute>}` slot when a placebo is used, and any
        other slot filled from `contexts`
    attribute : str
        Slot that varies within a set
    values : dict
        Group -> candidate values; one value per group is drawn for each set
    placebo : tuple, optional
        (slot, {group: values}) of an attribute that must not change the outcome
    contexts : sequence of dict, optional
        Fillings of the other slots; one is drawn per set and shared by all the requests of the set
    seed : int, optional
    """

    def __init__(
        self,
        templates: Sequence[str],
        attribute: str,
        values: dict,
        placebo: Optional[tuple] = None,
        contexts: Optional[Sequence[dict]] = None,
        seed: int = 0,
    ):
        if not templates:
            raise ValueError("at least one template is required")
        _check_values("values", values)
        slots = [attribute] + ([placebo[0]] if placebo else [])
        for template in templates:
            missing = [s for s in slots if "{" + s + "}" not in template]
            if missing:
                raise ValueError(f"template {template!r} has no slot for {missing}")
        if placebo:
            _check_values("placebo", placebo[1])
        self.templates = list(templates)
        self.attribute = attribute
        self.values = {str(k): list(v) for k, v in values.items()}
        self.placebo = (placebo[0], {str(k): list(v) for k, v in placebo[1].items()}) if placebo else None
        self.contexts = list(contexts) if contexts else [{}]
        self.seed = seed

    def generate(self, n_sets: int, n_placebo_sets: Optional[int] = None) -> list:
        """`n_sets` sets on the probed attribute and `n_placebo_sets` (default: as many) placebo sets"""
        rng = random.Random(self.seed)  # nosec B311
        n_placebo_sets = (n_sets if n_placebo_sets is None else n_placebo_sets) if self.placebo else 0
        probes = []
        for k in range(n_sets + n_placebo_sets):
            placebo_set = k >= n_sets
            template = rng.choice(self.templates)
            context = dict(rng.choice(self.contexts))
            set_id = f"{'placebo' if placebo_set else 'cf'}-{k:05d}"
            if placebo_set:
                slot, groups = self.placebo
                # the probed attribute is held fixed, the placebo varies
                context[self.attribute] = rng.choice(next(iter(self.values.values())))
            else:
                slot, groups = self.attribute, self.values
                if self.placebo:
                    context[self.placebo[0]] = rng.choice(next(iter(self.placebo[1].values())))
            for group, candidates in groups.items():
                value = rng.choice(candidates)
                fill = {**context, slot: value}
                probes.append(
                    Probe(
                        set_id=set_id,
                        variant=group,
                        attribute=slot,
                        value=value,
                        control=PLACEBO if placebo_set else NO_CONTROL,
                        text=template.format(**fill),
                        context=context,
                    )
                )
        return probes

    def run(
        self,
        probes: Sequence[Probe],
        target: Callable[[Probe], Any],
        outcome: Callable[[Any], Any],
        shuffle: bool = True,
        progress: Optional[Callable[[int, int, dict], None]] = None,
    ) -> pd.DataFrame:
        """Send every probe to `target` (in a random order) and map each response to an outcome with `outcome`

        `outcome` returns a label (e.g. the first tool called, "refused", "escalated") or a dict of labels. A failed
        call is kept, with its error, and excluded later by the analysis.
        """
        order = list(probes)
        if shuffle:
            random.Random(self.seed + 1).shuffle(order)  # nosec B311
        rows = []
        for i, probe in enumerate(order, start=1):
            row = {
                "set_id": probe.set_id,
                "variant": probe.variant,
                "attribute": probe.attribute,
                "value": probe.value,
                "control": probe.control,
                "error": None,
            }
            started = time.perf_counter()
            try:
                result = outcome(target(probe))
                if isinstance(result, dict):
                    row.update(result)
                else:
                    row["outcome"] = result
            except Exception as error:  # noqa: BLE001 - one failed request must not stop the probe
                row["error"] = f"{type(error).__name__}: {error}"
            row["latency_s"] = time.perf_counter() - started
            rows.append(row)
            if progress:
                progress(i, len(order), row)
        return pd.DataFrame(rows)


class ComparativeProbe:
    """Ask the target to choose between two profiles that differ only by the probed attribute, in both orders

    Parameters
    ----------
    template : str
        With `{first}` and `{second}` slots (the two profiles) and, if `allow_tie`, a `{tie}` slot for the tie option
    profile : str
        Profile template with a `{<attribute>}` slot and any slot from `contexts`
    attribute, values, contexts, seed : see CounterfactualProbe
    allow_tie : bool, optional
        Offer a tie option (default True); forced choice otherwise
    tie_text : str, optional
        Text put in the `{tie}` slot when ties are allowed
    labels : "neutral" or sequence of (str, str), optional
        Rotate the option labels across the sets (set k uses `labels[k % len(labels)]`, both orders of a set share
        theirs): "neutral" for NEUTRAL_LABELS (A/B, 1/2, X/Y and reversed). The template then needs `{first_label}`
        and `{second_label}` slots. Default None: no rotation, the slots (if any) get "A" and "B".
    """

    def __init__(
        self,
        template: str,
        profile: str,
        attribute: str,
        values: dict,
        contexts: Optional[Sequence[dict]] = None,
        allow_tie: bool = True,
        tie_text: str = "",
        seed: int = 0,
        labels: Any = None,
    ):
        _check_values("values", values)
        if "{first}" not in template or "{second}" not in template:
            raise ValueError("the template needs {first} and {second} slots")
        if "{" + attribute + "}" not in profile:
            raise ValueError(f"the profile needs a {{{attribute}}} slot")
        self.template = template
        self.profile = profile
        self.attribute = attribute
        self.values = {str(k): list(v) for k, v in values.items()}
        self.contexts = list(contexts) if contexts else [{}]
        self.allow_tie = allow_tie
        self.tie_text = tie_text if allow_tie else ""
        self.seed = seed
        if isinstance(labels, str):
            if labels != "neutral":
                raise ValueError('labels must be "neutral" or a sequence of (first, second) label pairs')
            labels = NEUTRAL_LABELS
        if labels is not None:
            labels = [tuple(str(x) for x in pair) for pair in labels]
            if not labels or any(len(pair) != 2 or not all(pair) or pair[0] == pair[1] for pair in labels):
                raise ValueError("labels needs pairs of two distinct, non-empty labels")
            if "{first_label}" not in template or "{second_label}" not in template:
                raise ValueError("label rotation needs {first_label} and {second_label} slots in the template")
        self.labels = labels

    def generate(self, n_sets: int) -> list:
        """Each set: one pair of groups, the two profiles, asked in both orders (control "none" then "order_swapped")"""
        rng = random.Random(self.seed)  # nosec B311
        groups = list(self.values)
        pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i + 1 :]]
        probes = []
        for k in range(n_sets):
            a, b = pairs[k % len(pairs)]
            context = rng.choice(self.contexts)
            profiles = {g: self.profile.format(**{**context, self.attribute: rng.choice(self.values[g])}) for g in (a, b)}
            labels = self.labels[k % len(self.labels)] if self.labels else DEFAULT_LABELS
            for control, (first, second) in (("none", (a, b)), ("order_swapped", (b, a))):
                text = self.template.format(
                    **{
                        **context,
                        "first": profiles[first],
                        "second": profiles[second],
                        "tie": self.tie_text,
                        "first_label": labels[0],
                        "second_label": labels[1],
                    }
                )
                probes.append(
                    Probe(
                        set_id=f"cmp-{k:05d}",
                        variant=f"{first}|{second}",
                        attribute=self.attribute,
                        value=f"{profiles[first]}|{profiles[second]}",
                        control=control,
                        text=text,
                        context=context,
                        profiles=(profiles[first], profiles[second]),
                        labels=tuple(labels),
                    )
                )
        return probes

    def run(
        self,
        probes,
        target,
        choice: Optional[Callable[[Any], Optional[str]]] = None,
        shuffle: bool = True,
        progress=None,
        refusal: Optional[Callable[[Any], bool]] = None,
    ) -> pd.DataFrame:
        """`choice` maps a response to "first", "second" or "tie"; the chosen group is derived from the order

        `choice` may also return a label of the probe (e.g. "X" for the first profile when the labels are X/Y), or
        "refusal". By default (None) the answer is read with `label_choice` and the labels of each probe. An answer
        without choice is a refusal when `refusal(response)` is true (default `is_refusal`: French and English refusal
        phrasings), else unparsed. Columns added to the former ones: `answer_status` ("ok", "refusal", "unparsed",
        "error"), `unusable` (refusal or unparsed), `labels` ("A|B") and `instrument` ("choice"). `choice` is
        "refusal" for a refusal (it was None, like an unparsed answer) and stays None for an unparsed answer;
        `chosen_group` is None for both.
        """
        refusal = refusal or is_refusal

        def read(pair):
            probe, response = pair
            labels = probe.labels or DEFAULT_LABELS
            picked = choice(response) if choice is not None else label_choice(response, labels)
            if isinstance(picked, str) and picked not in ("first", "second", "tie", ANSWER_REFUSAL):
                picked = {labels[0].lower(): "first", labels[1].lower(): "second"}.get(picked.strip().lower())
            if picked in ("first", "second", "tie"):
                return {"choice": picked, "answer_status": ANSWER_OK}
            declined = picked == ANSWER_REFUSAL or bool(refusal(response))
            if declined:
                return {"choice": ANSWER_REFUSAL, "answer_status": ANSWER_REFUSAL}
            return {"choice": None, "answer_status": ANSWER_UNPARSED}

        runner = CounterfactualProbe(["{x}"], "x", {"a": ["a"], "b": ["b"]}, seed=self.seed)
        rows = runner.run(probes, lambda probe: (probe, target(probe)), read, shuffle=shuffle, progress=progress)
        return self._finish(rows, probes, "choice")

    def run_loglik(self, probes, loglik: Optional[Callable[[str], Optional[float]]], progress=None) -> pd.DataFrame:
        """Log-likelihood instrument: the "choice" is the profile the model finds the more likely, nothing is asked

        `loglik(text)` returns the log-likelihood of a profile text under the target model (e.g. the sum of the token
        logprobs of the sentence; normalize it yourself when the profiles differ in length). The two orders of a set
        give the same choice by construction: there is no position effect. When the target exposes no logprobs
        (`loglik` None, or its first call raises NotImplementedError or returns None), the instrument is skipped: an
        empty frame with the same columns is returned, `rows.attrs["skipped"] == "no_logprobs"`. Columns as `run`,
        plus `loglik_first` and `loglik_second`; `instrument` is "loglik".
        """
        cache: dict = {}

        def score(text):
            if text not in cache:
                cache[text] = loglik(text)
            return cache[text]

        probes = list(probes)
        if any(len(p.profiles) != 2 for p in probes):
            raise ValueError("run_loglik needs the profile texts of each probe (Probe.profiles, set by generate)")
        available = loglik is not None and bool(probes)
        if available:
            try:
                available = score(probes[0].profiles[0]) is not None
            except NotImplementedError:
                available = False
        if not available:
            log.info("log-likelihood instrument skipped: the target exposes no logprobs")
            rows = self._finish(pd.DataFrame(columns=_ROW_COLUMNS), [], "loglik")
            rows["loglik_first"], rows["loglik_second"] = [], []
            rows.attrs["skipped"] = "no_logprobs"
            return rows

        def read(probe):
            first, second = score(probe.profiles[0]), score(probe.profiles[1])
            if first is None or second is None:
                raise ValueError("no log-likelihood returned")
            picked = "first" if first > second else "second" if second > first else "tie"
            return {"choice": picked, "answer_status": ANSWER_OK, "loglik_first": float(first), "loglik_second": float(second)}

        runner = CounterfactualProbe(["{x}"], "x", {"a": ["a"], "b": ["b"]}, seed=self.seed)
        rows = runner.run(probes, lambda probe: probe, read, shuffle=False, progress=progress)
        for column in ("loglik_first", "loglik_second"):
            if column not in rows:
                rows[column] = None
        return self._finish(rows, probes, "loglik")

    def _finish(self, rows: pd.DataFrame, probes, instrument: str) -> pd.DataFrame:
        """Columns common to the instruments: answer status, unusable flag, labels, groups and chosen group"""
        for column in ("choice", "answer_status"):
            if column not in rows:
                rows[column] = None
        rows["answer_status"] = [
            ANSWER_ERROR if pd.notna(error) else (status if isinstance(status, str) else ANSWER_UNPARSED)
            for error, status in zip(rows["error"], rows["answer_status"])
        ]
        rows["unusable"] = rows["answer_status"].isin(UNUSABLE_STATUSES)
        labels = {(p.set_id, p.control): "|".join(p.labels) if p.labels else None for p in probes}
        rows["labels"] = [labels.get((s, c)) for s, c in zip(rows["set_id"], rows["control"])]
        rows["instrument"] = instrument
        if not len(rows):
            for column in ("first_group", "second_group", "chosen_group"):
                rows[column] = []
            rows["allow_tie"] = []
            return rows
        firsts = rows["variant"].str.split("|", expand=True)
        rows["first_group"], rows["second_group"] = firsts[0], firsts[1]
        rows["chosen_group"] = [
            {"first": r.first_group, "second": r.second_group, "tie": "tie"}.get(r.choice) for r in rows.itertuples()
        ]
        rows["allow_tie"] = self.allow_tie
        return rows


# columns of a CounterfactualProbe.run row, for the empty frame of a skipped instrument
_ROW_COLUMNS = ["set_id", "variant", "attribute", "value", "control", "error", "choice", "answer_status", "latency_s"]
