"""Small synthetic checks for paper analyses, without experiment data."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin


PLOTS = Path(__file__).resolve().parents[1] / "analysis_scripts" / "paper_plots"


def module(name):
    spec = importlib.util.spec_from_file_location(name, PLOTS / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture(scope="module")
def training():
    return module("plot_training_budget")


@pytest.fixture(scope="module")
def proposal():
    return module("plot_case3_proposal_comparison")


def test_image_check_compares_pixels_not_png_metadata(tmp_path):
    checker = module("verify_paper_figures")
    reference, generated = tmp_path / "reference.png", tmp_path / "generated.png"
    image = Image.new("RGBA", (3, 2), (10, 20, 30, 255))
    image.save(reference)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Software", "different metadata, same pixels")
    image.save(generated, pnginfo=metadata)
    result = checker.compare_images(reference, generated)
    assert result["match"]
    assert result["differing_pixels"] == 0
    assert result["reference_sha256"] != result["generated_sha256"]


@pytest.mark.parametrize("change", ["color", "alpha", "dimensions", "missing"])
def test_image_check_rejects_changed_or_missing_assets(tmp_path, change):
    checker = module("verify_paper_figures")
    reference, generated = tmp_path / "reference.png", tmp_path / "generated.png"
    image = Image.new("RGBA", (3, 2), (10, 20, 30, 255))
    image.save(reference)
    if change == "dimensions":
        image = image.resize((4, 2))
    elif change in {"color", "alpha"}:
        image.putpixel((1, 1), (11, 20, 30, 255) if change == "color" else (10, 20, 30, 254))
    if change != "missing":
        image.save(generated)
    result = checker.compare_images(reference, generated)
    assert not result["match"]
    if change in {"color", "alpha"}:
        assert result["differing_pixels"] == 1
        assert result["max_channel_difference"] == 1


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def training_rows():
    return [
        dict(
            candidate_id=cid,
            budget=budget,
            rmse_total=error,
            energy_mj=energy,
            source="external/checkpoint.json",
        )
        for budget in (15, 300)
        for cid, error, energy in [("small", 0.8, 1.0), ("large", 0.3, 4.0)]
    ]


def test_front_retains_tradeoffs_and_equal_objectives(training):
    points = {
        "small": (0.8, 1.0),
        "large": (0.3, 4.0),
        "tied": (0.3, 4.0),
        "dominated": (0.8, 4.0),
        "same_energy_worse": (0.9, 1.0),
    }
    assert training.front(points) == ["small", "large", "tied"]


def test_training_matched_panel_and_orientation(training, tmp_path):
    points = training.load_csv(
        write_csv(tmp_path / "panel.csv", training_rows()), [15, 300]
    )
    assert set(points[15]) == set(points[300]) == {"small", "large"}
    assert points[15]["small"] == (0.8, 1.0)  # RMSE first; energy second.


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "nan", "infinite_energy", "changing_energy"]
)
def test_training_csv_rejects_invalid_scientific_panels(training, tmp_path, defect):
    rows = training_rows()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(rows[0].copy())
    elif defect == "nan":
        rows[0]["rmse_total"] = "nan"
    elif defect == "infinite_energy":
        rows[0]["energy_mj"] = "inf"
    else:
        rows[-1]["energy_mj"] = 5.0
    with pytest.raises(ValueError):
        training.load_csv(write_csv(tmp_path / "panel.csv", rows), [15, 300])


def json_rows():
    return [
        dict(
            candidate_id=r["candidate_id"],
            seed=17,
            budget=r["budget"],
            policy="budget_only",
            status="ok",
            energy_policy="fixed_historical_measurement",
            energy_mj_per_inference=r["energy_mj"],
            metrics={"rmse_total": r["rmse_total"]},
        )
        for r in training_rows()
    ]


@pytest.mark.parametrize(
    "defect", [None, "missing", "duplicate", "nan", "changing_energy", "wrong_policy"]
)
def test_training_json_protocol_validation(training, tmp_path, defect):
    rows = json_rows()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(rows[0].copy())
    elif defect == "nan":
        rows[0]["metrics"]["rmse_total"] = float("nan")
    elif defect == "changing_energy":
        rows[-1]["energy_mj_per_inference"] = 5.0
    elif defect == "wrong_policy":
        rows[0]["energy_policy"] = "new_measurement"
    path = tmp_path / "results.json"
    path.write_text(json.dumps(rows))
    if defect is None:
        assert training.load([path], [15, 300], 17)[300]["large"] == (0.3, 4.0)
    else:
        with pytest.raises(ValueError):
            training.load([path], [15, 300], 17)


def proposal_tables():
    # Highest F1 and lowest energy are deliberately different from score winner.
    rows = []
    endpoints = []
    for board in ("STM32", "Portenta"):
        for method in ("Optuna", "Memory", "History"):
            for trial, f1, energy in [(0, 0.9, 1000.0), (1, 0.8, 100.0)]:
                rows.append(
                    dict(
                        board=board,
                        method=method,
                        trial=trial,
                        f1=f1,
                        energy_mj=energy,
                        score=f1 - energy / 4000,
                        latency_ms=2.0,
                    )
                )
            endpoints.append(
                dict(study=f"{board.lower()}_{method.lower()}", best_individual=0.775)
            )
    return rows, endpoints


def test_proposal_selects_score_maximum_without_dropping_repeats(proposal, tmp_path):
    rows, endpoints = proposal_tables()
    observations, winners = proposal.load(
        write_csv(tmp_path / "points.csv", rows),
        write_csv(tmp_path / "endpoints.csv", endpoints),
        2,
    )
    assert len(observations) == 12
    assert len(winners) == 6
    assert all(
        winner["trial"] == 1 and winner["score"] == pytest.approx(0.775)
        for winner in winners.values()
    )


@pytest.mark.parametrize(
    "defect",
    ["wrong_score", "missing_trial", "duplicate_trial", "nonfinite", "wrong_endpoint"],
)
def test_proposal_rejects_invalid_scores_and_counts(proposal, tmp_path, defect):
    rows, endpoints = proposal_tables()
    if defect == "wrong_score":
        rows[0]["score"] += 0.01
    elif defect == "missing_trial":
        rows.pop()
    elif defect == "duplicate_trial":
        rows[1]["trial"] = rows[0]["trial"]
    elif defect == "nonfinite":
        rows[0]["energy_mj"] = "nan"
    else:
        endpoints[0]["best_individual"] = 0.9
    with pytest.raises(ValueError):
        proposal.load(
            write_csv(tmp_path / "points.csv", rows),
            write_csv(tmp_path / "endpoints.csv", endpoints),
            2,
        )
