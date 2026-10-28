import csv
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("prepare_study_outcomes", Path(__file__).parents[1] / "scripts/prepare_study_outcomes.py")
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


def fixture(tmp_path):
    rows = [{"Patient_ID": pid, "OS.TIME": "730", "OS.EVENT": "1"} for pid in ("001", "NA", "A")]
    for name in ("source.csv", "delivery.csv"):
        with (tmp_path / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for split, pid in zip(("train", "val", "test"), ("001", "NA", "A")):
        (tmp_path / f"{split}_ids.txt").write_text(pid + "\n")
    return rows


def test_preserves_literal_ids_and_does_not_infer_semantics(tmp_path):
    fixture(tmp_path)
    rows, receipt = module.prepare(tmp_path / "source.csv", tmp_path / "delivery.csv", tmp_path)
    assert [row["Patient_ID"] for row in rows] == ["001", "NA", "A"]
    assert receipt["time_origin_inferred"] is False
    assert receipt["scientific_qualification"] is False


@pytest.mark.parametrize("failure", ["overlap", "outcome", "duplicate", "omission", "duplicate_header", "alias_disagreement"])
def test_refuses_identity_or_outcome_changes(tmp_path, failure):
    fixture(tmp_path)
    if failure == "overlap":
        (tmp_path / "test_ids.txt").write_text("001\n")
    elif failure == "omission":
        (tmp_path / "test_ids.txt").write_text("")
    elif failure == "duplicate_header":
        (tmp_path / "delivery.csv").write_text("Patient_ID,OS.TIME,OS.TIME,OS.EVENT\n001,730,731,1\n")
    elif failure == "alias_disagreement":
        (tmp_path / "delivery.csv").write_text("Patient_ID,patient_id,OS.TIME,OS.EVENT\n001,1,730,1\n")
    else:
        path = tmp_path / "delivery.csv"
        content = path.read_text()
        path.write_text(content.replace("730", "731", 1) if failure == "outcome" else content + "001,730,1\n")
    with pytest.raises(ValueError):
        module.prepare(tmp_path / "source.csv", tmp_path / "delivery.csv", tmp_path)
