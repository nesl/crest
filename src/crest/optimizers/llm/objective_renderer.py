"""Deterministic prose for the supported scoring terms; no model imports."""

import math


def render_objective(config, context):
    """Explain a scoring configuration using known fixed context values.

    Parameters
    ----------
    config : dict
        Original scoring DSL, retained unchanged in the audit context.
    context : dict
        Existing task, runtime, and device semantic context.

    Returns
    -------
    str or None
        Exact equation and configured constraints, or None when unsupported.
        Unresolved limits retain their metric names rather than invented values.
    """
    if not isinstance(config, dict) or config.get("type") != "scoring-function":
        return None
    runtime = context.get("runtime_context", {})
    device = context.get("device_context", {})
    task = context.get("task_context", {})
    values = dict(runtime)
    for name, source in (("max_ram_bytes", "ram_capacity_bytes"),
                         ("max_flash_bytes", "flash_capacity_bytes")):
        if source in device:
            values[name] = device[source]
    definitions = config.get("metrics", {})
    notes = []
    units = {"latency_ms": "ms", "latency_budget_ms": "ms",
             "energy_mj_per_inference": "mJ", "ram_bytes": "bytes",
             "flash_bytes": "bytes", "external_flash_bytes": "bytes"}

    def number(value):
        """Validate a finite scalar.

        Parameters
        ----------
        value : object
            Configured scalar.

        Returns
        -------
        float
            Finite numeric value.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Unknown numeric value")
        return float(value)

    def resolve(ref, seen=()):
        """Resolve fixed references without guessing missing metrics.

        Parameters
        ----------
        ref : dict
            Literal or metric reference from the scoring DSL.
        seen : tuple
            Cycle guard for derived metrics.

        Returns
        -------
        float
            Resolved scalar in the metric's declared units.
        """
        if ref.get("type") == "literal":
            return number(ref["value"])
        if ref.get("type") != "metric":
            raise ValueError("Unsupported reference")
        name = ref["metric"]
        if name in values:
            return number(values[name])
        if name in seen:
            raise ValueError("Cyclic reference")
        definition = definitions.get(name, {})
        if definition.get("type") != "energy-budget-from-power":
            raise ValueError("Unresolved reference")
        power = resolve(definition["power_mw"], (*seen, name))
        duration = resolve(definition["duration_ms"], (*seen, name))
        if power < 0 or duration <= 0:
            raise ValueError("Invalid energy reference")
        value = power * duration / 1000
        note = f"{name} = {power:.15g} mW × {duration:.15g} ms / 1000 = {value:.15g} mJ."
        if note not in notes:
            notes.append(note)
        return value

    try:
        outputs = task.get("objective", {}).get("study_outputs", [])
        if len(outputs) != 1 or outputs[0] != {"direction": "maximize", "metric": "score"}:
            return None
        terms = config["params"]["terms"]
        if not terms:
            return None
        equation, explanations = [], []
        coefficients = {}
        for term in terms:
            weight = number(term["weight"])
            metric = term["metric"]
            if not isinstance(metric, str):
                return None
            expression = metric
            coefficient = weight
            if term["type"] == "normalized-weighted":
                reference = resolve(term["reference"])
                if reference <= 0:
                    return None
                expression = f"({metric} / {reference:.15g})"
                coefficient /= reference
            elif term["type"] != "weighted":
                return None
            magnitude = expression if abs(weight) == 1 else f"{abs(weight):.15g} × {expression}"
            equation.append(("− " if weight < 0 else "+ ") + magnitude)
            coefficients.setdefault(metric, []).append(coefficient)
        # Explain each metric's net effect, including opposing terms in the sum.
        for metric, contributions in coefficients.items():
            coefficient = math.fsum(contributions)
            direction = "increase" if coefficient > 0 else "decrease"
            if coefficient:
                explanations.append(f"Holding other metrics fixed, a {abs(0.01 / coefficient):.15g} "
                    f"{units.get(metric, 'unit')} {direction} in {metric} raises score by 0.01.")
        text = "Maximize score; higher is better. Score = " + " ".join(equation).removeprefix("+ ") + "."
        text += " " + " ".join(notes + explanations)
        if any(t["metric"] == "energy_mj_per_inference" for t in terms):
            text += " Energy units are mJ; use the measurement scope defined in runtime_context, not a different cadence/window energy metric."
        constraints = []
        for rule in task.get("feasibility", {}).get("rules", []):
            operator = {"gt": "≤", "gte": "<", "lt": "≥", "lte": ">"}.get(rule.get("condition"))
            if operator is None:
                return None
            ref = rule["reference"]
            try:
                limit = f"{resolve(ref):.15g}"
            except (KeyError, ValueError, TypeError):
                if ref.get("type") != "metric":
                    return None
                limit = ref["metric"] + " (unresolved here)"
            metric = rule["metric"]
            constraints.append(f"{metric} {operator} {limit} {units.get(metric, '')}".strip())
        if constraints:
            text += " Feasibility additionally requires: " + "; ".join(constraints) + ". These are configured limits, not guarantees that compilation or allocation will succeed."
        return text
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
