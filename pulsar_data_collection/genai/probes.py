"""Counterfactual probes for LLM applications and agents: paired requests that differ only by one attribute.

An audit of a language model can produce its own verdict: answers change with the order of the options, with the
wording, or at random. The probes therefore build the controls in:

- every set holds one request per group of the probed attribute (e.g. a first name typical of each group), everything
  else being identical;
- **placebo** sets vary an attribute that must not matter (e.g. the day the request is written): their flip rate is
  the noise floor of the instrument, the probed attribute is compared with it;
- comparative probes (choose between two profiles) are asked in **both orders**, with or without a tie option, to
  measure and cancel the position effect;
- requests are sent in a random order and look like ordinary requests (nothing tells the model it is audited).

The collected rows are analysed with `pulsar_metrics.fairness.paired`. When the target is an instrumented agent, pass
`probe.span_attributes()` to its agent span: the `pulsar.cf.*` attributes then reach the `agent_run` records.

The value lists are the user's responsibility: lists of names standing for groups are sensitive and must be reviewed
(e.g. by a GBA Plus expert). The ones shipped with the demo are provisional and marked as such.
"""

import random  # reproducible sampling and ordering of the probe requests, never used for security
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from . import semconv as sc

NO_CONTROL = "none"
PLACEBO = "placebo"


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
            for control, (first, second) in (("none", (a, b)), ("order_swapped", (b, a))):
                text = self.template.format(first=profiles[first], second=profiles[second], tie=self.tie_text, **context)
                probes.append(
                    Probe(
                        set_id=f"cmp-{k:05d}",
                        variant=f"{first}|{second}",
                        attribute=self.attribute,
                        value=f"{profiles[first]}|{profiles[second]}",
                        control=control,
                        text=text,
                        context=context,
                    )
                )
        return probes

    def run(self, probes, target, choice: Callable[[Any], str], shuffle: bool = True, progress=None) -> pd.DataFrame:
        """`choice` maps a response to "first", "second" or "tie"; the chosen group is derived from the order"""
        runner = CounterfactualProbe(["{x}"], "x", {"a": ["a"], "b": ["b"]}, seed=self.seed)
        rows = runner.run(probes, target, lambda response: {"choice": choice(response)}, shuffle=shuffle, progress=progress)
        if "choice" not in rows:
            rows["choice"] = None
        firsts = rows["variant"].str.split("|", expand=True)
        rows["first_group"], rows["second_group"] = firsts[0], firsts[1]
        rows["chosen_group"] = [
            {"first": r.first_group, "second": r.second_group, "tie": "tie"}.get(r.choice) for r in rows.itertuples()
        ]
        rows["allow_tie"] = self.allow_tie
        return rows
