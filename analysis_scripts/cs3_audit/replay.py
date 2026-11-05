"""Audit cross-target measurements of score-selected configurations."""

import argparse
import csv
import json
import math
from pathlib import Path


def audit(observations, replays, energy_scale=400.0, weight=0.1):
    """Match replay architecture and inherited quality to source selections.

    Parameters
    ----------
    observations : list of dict
        Feasible endpoint observations from ``campaign.py``.
    replays : sequence of tuple
        Destination-board labels and lists of replay CSV rows.
    energy_scale, weight : float, optional
        Score parameters in ``F1 - weight * energy / energy_scale``.

    Returns
    -------
    list of dict
        Native and replay measurements, including destination energy ranks.
    """
    if (
        energy_scale <= 0
        or weight < 0
        or not all(map(math.isfinite, [energy_scale, weight]))
    ):
        raise ValueError("Invalid score parameters")
    pools = {}
    for row in observations:
        key = (row["board"], row["method"])
        pools.setdefault(key, []).append(row)
    selected = {}
    output = []
    for key, pool in pools.items():
        winner = min(
            pool,
            key=lambda r: (
                -(float(r["f1"]) - weight * float(r["energy_mj"]) / energy_scale),
                -float(r["f1"]),
                float(r["energy_mj"]),
                int(r["trial"]),
            ),
        )
        study = winner["study_name"]
        if study in selected:
            raise ValueError("Source study must identify one campaign")
        selected[study] = winner
        output.append(
            dict(
                source_board=key[0],
                method=key[1],
                trial=int(winner["trial"]),
                board=key[0],
                measurement="native",
                f1=float(winner["f1"]),
                energy_mj=float(winner["energy_mj"]),
                score=float(winner["f1"])
                - weight * float(winner["energy_mj"]) / energy_scale,
                model_variant="source measurement",
            )
        )
    seen = set()
    for destination, rows in replays:
        for row in rows:
            if row["source__study_name"] not in selected:
                raise ValueError("Replay source is absent from the supplied campaigns")
            winner = selected[row["source__study_name"]]
            key = (winner["board"], winner["method"], destination)
            if key in seen or destination == winner["board"]:
                raise ValueError("Duplicate replay or non-cross-target destination")
            seen.add(key)
            source_quality = float(row["source__metric__macro_f1"])
            if not math.isclose(
                source_quality, float(winner["f1"]), rel_tol=0, abs_tol=1e-10
            ):
                raise ValueError(
                    "Replay did not inherit selected-source validation quality"
                )
            params = json.loads(
                winner.get("hyperparameters_json") or "null"
            ) or json.loads(winner["raw_params_json"])
            replay_params = json.loads(row["family_hparams_json"])
            if not replay_params or any(
                k not in params or params[k] != v for k, v in replay_params.items()
            ):
                raise ValueError("Replay architecture differs from source selection")
            if float(row["target__error_code"]) != 1:
                raise ValueError("Replay controller did not record success")
            if row.get("target__cadenced_error_code") not in (
                None,
                "",
                "-1",
                "-1.0",
                "0",
                "0.0",
            ):
                raise ValueError("Replay periodic pass failed")
            energy = float(row["target__energy_mj_per_inference"])
            if not math.isfinite(energy) or energy <= 0:
                raise ValueError("Replay energy must be finite and positive")
            # CSV row positions are not Optuna trial IDs. Identity comes from
            # the selected source's study, architecture and validation quality.
            output.append(
                dict(
                    source_board=winner["board"],
                    method=winner["method"],
                    trial=int(winner["trial"]),
                    board=destination,
                    measurement="replay",
                    f1=source_quality,
                    energy_mj=energy,
                    score=source_quality - weight * energy / energy_scale,
                    model_variant=row.get("model_variant") or "not recorded",
                )
            )
    for row in output:
        peers = [r for r in output if r["board"] == row["board"]]
        row["energy_rank"] = 1 + sum(r["energy_mj"] < row["energy_mj"] for r in peers)
        row["rank_population"] = len(peers)
    return output


def main():
    """Read explicit private inputs and write an external replay audit.

    Returns
    -------
    None
        Writes ``native_and_destination.csv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument(
        "--replay", action="append", required=True, metavar="DESTINATION=CSV"
    )
    parser.add_argument("--energy-scale", type=float, default=400.0)
    parser.add_argument("--weight", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.observations.open(newline="") as stream:
        observations = list(csv.DictReader(stream))
    replays = []
    for value in args.replay:
        board, path = value.split("=", 1)
        with Path(path).open(newline="") as stream:
            replays.append((board, list(csv.DictReader(stream))))
    rows = audit(observations, replays, args.energy_scale, args.weight)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "native_and_destination.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Audited {len(rows)} native/replay measurements.")


if __name__ == "__main__":
    main()
