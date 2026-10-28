"""Synthetic logging checks; no model training or clinical outcomes."""
import csv
from pathlib import Path
import tempfile
import unittest

from trifusesurv2.multimodal_survival.train import append_epoch_metrics


class EpochMeasurementTests(unittest.TestCase):
    def test_close_validation_nll_values_survive_two_epoch_csv_round_trip(self):
        # Console formatting would make these look equal. Checkpoint selection
        # and its audit must retain the actual distinct scalar values.
        values = [0.12345678901234567, 0.12345678901234565]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            for epoch, nll in enumerate(values, 1):
                append_epoch_metrics(path, {"epoch": epoch, "val_nll": nll,
                    "epoch_train_and_evaluation_seconds": 1.234567890123 + epoch})
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["epoch"]) for row in rows], [1, 2])
            self.assertEqual([float(row["val_nll"]) for row in rows], values)
            self.assertLess(float(rows[1]["val_nll"]), float(rows[0]["val_nll"]))
            self.assertTrue(all(float(row["epoch_train_and_evaluation_seconds"]) > 0 for row in rows))

    def test_historical_header_cannot_silently_mislabel_new_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            path.write_text("epoch,train_loss\n1,2.5\n")
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "historical"):
                append_epoch_metrics(path, {"epoch": 2, "val_nll": .1,
                                          "epoch_train_and_evaluation_seconds": 2.})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
