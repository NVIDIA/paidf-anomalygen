# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
End-to-end tests for the DAFT v3.0 converters. Only the CLI `main()` entry
points are exercised — helpers are not tested in isolation.

Usage:

pytest tests/test_convert_daft_format.py
"""

import json
import pathlib
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.anomaly_gen import convert_from_daft_format, convert_to_daft_format


def _write_rgb(path: pathlib.Path, size=(64, 48)) -> None:
    arr = np.random.randint(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_mask(path: pathlib.Path, size=(64, 48)) -> None:
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    arr[size[1] // 4 : size[1] // 2, size[0] // 4 : size[0] // 2] = 255
    Image.fromarray(arr, mode="L").save(path)


class TestComponentDefectRoundTrip(unittest.TestCase):
    """Forward then reverse a component/defect dataset and verify bit-identity."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls._tmp.name)
        cls.source = cls.root / "val"

        cls.expected: list[tuple[str, str, str]] = []
        for comp in ("Capacitor", "Resistor"):
            for defect in ("bridge", "less_solder"):
                img_dir = cls.source / comp / "anomaly_image" / defect
                msk_dir = cls.source / comp / "mask" / defect
                img_dir.mkdir(parents=True, exist_ok=True)
                msk_dir.mkdir(parents=True, exist_ok=True)
                for i in range(2):
                    base = f"{comp}_{defect}_{i:03d}"
                    _write_rgb(img_dir / f"{base}.png")
                    _write_mask(msk_dir / f"{base}_mask.png")
                    cls.expected.append((comp, defect, base))

        cls.val_jsonl = cls.root / "validation.jsonl"
        cls.val_jsonl.write_text(
            '{"image_filename": "foo.png", "anomaly_type": "Capacitor+bridge"}\n'
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_round_trip(self):
        convert_to_daft_format.main([
            "--input", str(self.source),
            "--validation-jsonl", str(self.val_jsonl),
        ])

        daft = self.source.parent / "val_daft_v3"
        self.assertTrue((daft / "raw" / "rgb").is_dir())
        self.assertTrue((daft / "raw" / "mask").is_dir())
        self.assertTrue((daft / "contextual").is_dir())

        n = len(self.expected)
        self.assertEqual(len(list((daft / "raw" / "rgb").glob("*.png"))), n)
        self.assertEqual(len(list((daft / "raw" / "mask").glob("*.png"))), n)
        self.assertEqual(len(list((daft / "contextual").glob("*.json"))), n)
        self.assertTrue((daft / "task" / "validation.jsonl").is_file())

        sample = json.loads(next((daft / "contextual").glob("*.json")).read_text())
        self.assertEqual(sample["version"], "3.0")
        self.assertEqual(sample["metadata"]["type"], "image")
        parts = sample["scenario_info"].split(",")
        self.assertEqual(len(parts), 3)
        self.assertIn(parts[0], {"Capacitor", "Resistor"})
        self.assertIn(parts[1], {"bridge", "less_solder"})

        convert_from_daft_format.main(["--input", str(daft)])

        restored = self.source.parent / "val_restored"
        self.assertTrue(restored.is_dir())
        self.assertTrue((restored / "validation.jsonl").is_file())
        # Task file should be flat, not nested under task/.
        self.assertFalse((restored / "task").exists())

        for comp, defect, base in self.expected:
            orig_rgb = self.source / comp / "anomaly_image" / defect / f"{base}.png"
            orig_msk = self.source / comp / "mask" / defect / f"{base}_mask.png"
            dst_rgb = restored / comp / "anomaly_image" / defect / f"{base}.png"
            dst_msk = restored / comp / "mask" / defect / f"{base}_mask.png"
            self.assertTrue(dst_rgb.is_file(), f"missing {dst_rgb}")
            self.assertTrue(dst_msk.is_file(), f"missing {dst_msk}")
            self.assertEqual(orig_rgb.read_bytes(), dst_rgb.read_bytes())
            self.assertEqual(orig_msk.read_bytes(), dst_msk.read_bytes())


class TestInferenceOutputRoundTrip(unittest.TestCase):
    """Forward then reverse an inference-output dataset.

    The reverse produces a component/defect tree keyed off the filename stem
    (e.g. Capacitor+bridge_00000.png), not the original source filenames, so
    the round-trip validates structure + content rather than exact paths.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls._tmp.name)
        cls.source = cls.root / "example_output"

        recon_dir = cls.source / "reconstructed_image"
        orig_msk_dir = cls.source / "original_mask"
        recon_dir.mkdir(parents=True, exist_ok=True)
        orig_msk_dir.mkdir(parents=True, exist_ok=True)

        cls.samples = [
            ("Capacitor+bridge_00000", "Capacitor", "bridge"),
            ("Capacitor+bridge_00001", "Capacitor", "bridge"),
            ("Resistor+less_solder_00000", "Resistor", "less_solder"),
        ]
        for stem, _comp, _defect in cls.samples:
            _write_rgb(recon_dir / f"{stem}.png")
            _write_mask(orig_msk_dir / f"{stem}.png")

        (cls.source / "SDG_result.csv").write_text("a,b,c\n1,2,3\n")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_round_trip(self):
        convert_to_daft_format.main(["--input", str(self.source)])

        daft = self.source.parent / "example_output_daft_v3"
        n = len(self.samples)
        self.assertEqual(len(list((daft / "raw" / "rgb").glob("*.png"))), n)
        self.assertEqual(len(list((daft / "raw" / "mask").glob("*.png"))), n)
        self.assertEqual(len(list((daft / "contextual").glob("*.json"))), n)
        self.assertTrue((daft / "task" / "SDG_result.csv").is_file())

        seen = set()
        for j in (daft / "contextual").glob("*.json"):
            data = json.loads(j.read_text())
            parts = data["scenario_info"].split(",")
            self.assertEqual(len(parts), 3)
            seen.add((parts[0], parts[1]))
        self.assertEqual(
            seen, {("Capacitor", "bridge"), ("Resistor", "less_solder")}
        )

        convert_from_daft_format.main(["--input", str(daft)])

        restored = self.source.parent / "example_output_restored"
        self.assertTrue((restored / "SDG_result.csv").is_file())
        for stem, comp, defect in self.samples:
            self.assertTrue(
                (restored / comp / "anomaly_image" / defect / f"{stem}.png").is_file()
            )
            self.assertTrue(
                (restored / comp / "mask" / defect / f"{stem}_mask.png").is_file()
            )
