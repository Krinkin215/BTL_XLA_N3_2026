"""Checks for instance corruption, evaluation errors, and validation/test leakage."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from run_experiment import (dataset_path, load_split, otsu_seg, read_manifest,
                            save_prediction, score_instances, watershed_stages)


class ExperimentTests(unittest.TestCase):
    def test_metrics_aggregate_objects_instead_of_averaging_images(self):
        truths = [np.array([[1, 0, 2, 0, 3]], dtype=np.uint16),
                  np.array([[1, 0, 0, 0, 0]], dtype=np.uint16)]
        predictions = [np.array([[1, 0, 0, 0, 0]], dtype=np.uint16),
                       np.array([[1, 0, 2, 0, 0]], dtype=np.uint16)]
        score = score_instances(truths, predictions, (0.5,))[0]
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (2, 1, 2))
        self.assertAlmostEqual(score["precision"], 2 / 3)
        self.assertAlmostEqual(score["recall"], 1 / 2)
        self.assertAlmostEqual(score["f1"], 4 / 7)
        self.assertAlmostEqual(score["matching_accuracy"], 2 / 5)

    def test_export_retains_ids_above_uint8_and_uint16(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value, dtype in [(300, np.uint16), (70000, np.uint32)]:
                labels = np.array([[0, 1], [value, value]], dtype=np.int32)
                path = Path(tmp) / f"{value}.tif"
                save_prediction(path, labels, (2, 2))
                actual = tifffile.imread(path)
                self.assertEqual(actual.dtype, dtype)
                np.testing.assert_array_equal(actual, labels)
            with self.assertRaisesRegex(ValueError, "integer instance map"):
                save_prediction(Path(tmp) / "invalid.tif", np.ones((2, 2), np.float32), (2, 2))

    def test_constant_input_and_border_components(self):
        empty = np.zeros((48, 48), dtype=np.float32)
        self.assertFalse(otsu_seg(empty).any())
        self.assertFalse(watershed_stages(empty)[-1].any())
        image = empty.copy()
        image[:12, :12] = 1
        image[30:40, 30:40] = 1
        foreground, distance, markers, labels = watershed_stages(image, sigma=0, min_size=5, min_distance=15)
        self.assertEqual(np.count_nonzero(np.unique(labels)), 2)
        np.testing.assert_array_equal(labels > 0, foreground)
        self.assertTrue(np.any(markers[:12, :12] > 0))

    def test_manifest_rejects_duplicates_changed_splits_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for sid, split in [("v", "val"), ("t", "test")]:
                row = dict(source_id=sid, split=split, image_path=f"{sid}.png", mask_path=f"{sid}.tif",
                           normalized_path=f"{sid}.npy", width="2", height="2", num_instances="1",
                           appearance_group="grayscale_dark_background")
                for key in ("image_path", "mask_path", "normalized_path"):
                    (root / row[key]).touch()
                rows.append(row)

            def write(path, records):
                with path.open("w", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0])
                    writer.writeheader()
                    writer.writerows(records)

            master = root / "dataset_manifest.csv"
            manifest = root / "manifest.csv"
            write(master, rows)
            write(manifest, rows)
            self.assertEqual(len(read_manifest(root, manifest)), 2)
            write(manifest, [rows[0], rows[0], rows[1]])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_manifest(root, manifest)
            swapped = [{**rows[0], "split": "test"}, {**rows[1], "split": "val"}]
            write(manifest, swapped)
            with self.assertRaisesRegex(ValueError, "differs from master"):
                read_manifest(root, manifest)
            with self.assertRaisesRegex(ValueError, "Invalid dataset path"):
                dataset_path(root, "../outside.npy")

    def test_prepared_manifest_and_input_shapes(self):
        root = Path(__file__).resolve().parent / "StarDist_Dataset"
        rows = read_manifest(root, root / "experiments/pretrained_fluo/manifest.csv")
        val_rows, images, labels = load_split(root, rows, "val")
        self.assertEqual(len(val_rows), 20)
        self.assertEqual(sum(int(r["num_instances"]) for r in val_rows), 862)
        self.assertTrue(all(x.shape == y.shape for x, y in zip(images, labels)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
