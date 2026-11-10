"""Focused checks for data corruption and instance-label failure modes."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_dataset import appearance, merge_masks, normalize_gray, split_records


class DataIntegrityTests(unittest.TestCase):
    def test_source_instances_remain_separate_and_roundtrip_uint16(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp)
            (sample / 'masks').mkdir()
            for filename, column in [('b.png', 2), ('a.png', 0)]:
                mask = np.zeros((3, 4), np.uint8)
                mask[1:, column] = 255
                Image.fromarray(mask).save(sample / 'masks' / filename)
            labels, mapping = merge_masks(sample, (3, 4))
            self.assertEqual(mapping[0]['source_mask'], 'a.png')
            self.assertEqual(labels[1, 0], 1)
            self.assertEqual(labels[1, 2], 2)
            self.assertEqual(labels[0, 0], 0)
            Image.fromarray(labels).save(sample / 'labels.tif')
            reread = np.asarray(Image.open(sample / 'labels.tif'))
            self.assertEqual(reread.dtype, np.uint16)
            np.testing.assert_array_equal(labels, reread)

    def test_rejects_overlap_empty_and_wrong_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp)
            (sample / 'masks').mkdir()
            a = np.zeros((3, 4), np.uint8)
            a[1, 1] = 255
            Image.fromarray(a).save(sample / 'masks/a.png')
            Image.fromarray(a).save(sample / 'masks/b.png')
            with self.assertRaisesRegex(ValueError, 'Overlapping'):
                merge_masks(sample, (3, 4))
            Image.fromarray(np.zeros((3, 4), np.uint8)).save(sample / 'masks/b.png')
            with self.assertRaisesRegex(ValueError, 'Empty'):
                merge_masks(sample, (3, 4))
            Image.fromarray(np.ones((2, 4), np.uint8) * 255).save(sample / 'masks/b.png')
            with self.assertRaisesRegex(ValueError, 'matching 2D'):
                merge_masks(sample, (3, 4))

    def test_normalization_range_and_constant_rejection(self):
        a = np.arange(256, dtype=np.uint8).reshape(16, 16)
        normalized, low, high = normalize_gray(a)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(float(normalized.min()), 0)
        self.assertEqual(float(normalized.max()), 1)
        self.assertLess(low, high)
        with self.assertRaisesRegex(ValueError, 'zero range'):
            normalize_gray(np.ones((4, 4), np.uint8))

    def test_identical_images_stay_together_and_split_is_reproducible(self):
        records = [dict(source_id=f'sample-{i}', pixel_sha256=f'hash-{i // 2}',
                        appearance_group='grayscale_dark_background', num_instances=10,
                        primary_100=0) for i in range(30)]
        second = [r.copy() for r in records]
        split_records(records)
        split_records(second)
        self.assertEqual(records, second)
        for i in range(0, len(records), 2):
            self.assertEqual(records[i]['split'], records[i + 1]['split'])

    def test_color_is_not_assumed_to_be_grayscale(self):
        dark = np.zeros((20, 20, 3), np.uint8)
        dark[10, 10] = [255, 255, 255]
        self.assertEqual(appearance(dark)[0], 'grayscale_dark_background')
        dark[10, 10] = [255, 0, 0]
        self.assertEqual(appearance(dark)[0], 'color')


if __name__ == '__main__':
    unittest.main(verbosity=2)
