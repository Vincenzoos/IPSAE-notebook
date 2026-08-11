"""Bulk-model detection, discovery, and incomplete-input cases."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bulk_eval import discover_af3_models, discover_boltz_models, discover_bulk_models
from model_pairing import ModelType, detect_bulk_model_type, validate_structure_pae_pairing


def _touch(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class BulkDetectionTests(unittest.TestCase):
    def test_detects_af3_and_boltz_recursively(self) -> None:
        cases = [
            (ModelType.AF3, "nested/fold_a_full_data_0.json"),
            (ModelType.BOLTZ, "nested/pae_X_model_0.npz"),
        ]
        for expected, filename in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _touch(root / filename)
                self.assertEqual(detect_bulk_model_type(root), expected)

    def test_mixed_af3_and_boltz_is_rejected_across_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "af3/fold_a_full_data_0.json")
            _touch(root / "boltz/pae_X_model_1.npz")
            with self.assertRaisesRegex(ValueError, "Mixed"):
                detect_bulk_model_type(root)

    def test_af2_only_bulk_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "X_unrelaxed_rank_001_details.pdb")
            _touch(root / "X_scores_rank_001_details.json")
            with self.assertRaisesRegex(ValueError, "AF2"):
                detect_bulk_model_type(root)

    def test_empty_or_unrecognized_folder_is_rejected(self) -> None:
        for add_file in (False, True):
            with self.subTest(add_file=add_file), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                if add_file:
                    _touch(root / "notes.txt")
                with self.assertRaisesRegex(ValueError, "Unrecognized"):
                    detect_bulk_model_type(root)

    def test_missing_folder_and_file_path_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                detect_bulk_model_type(root / "missing")
            file_path = _touch(root / "not-a-folder")
            with self.assertRaises(NotADirectoryError):
                detect_bulk_model_type(file_path)

    def test_auxiliary_files_do_not_change_af3_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "fold_a_full_data_0.json")
            _touch(root / "fold_a_summary_confidences_0.json")
            _touch(root / "confidence_unrelated_model_0.json")
            _touch(root / "plddt_unrelated_model_0.npz")
            self.assertEqual(detect_bulk_model_type(root), ModelType.AF3)


class BulkDiscoveryTests(unittest.TestCase):
    def test_discovers_ready_af3_job_with_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "fold_a"
            _touch(folder / "fold_a_model_0.cif")
            _touch(folder / "fold_a_full_data_0.json")
            _touch(folder / "fold_a_summary_confidences_0.json")

            jobs, preview, detected = discover_bulk_models(root, model_index=0)

            self.assertEqual(detected, ModelType.AF3)
            self.assertEqual(len(jobs), 1)
            self.assertTrue(bool(preview.iloc[0]["Ready"]))
            validation = validate_structure_pae_pairing(
                jobs[0].structure_file,
                jobs[0].pae_file,
                jobs[0].model_type,
                summary_file=jobs[0].summary_file,
            )
            self.assertEqual(validation["summary_file"].name, "fold_a_summary_confidences_0.json")

    def test_discovers_ready_boltz_job_without_optional_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "AURKA_TPX2"
            _touch(folder / "AURKA_TPX2_model_0.cif")
            _touch(folder / "pae_AURKA_TPX2_model_0.npz")

            jobs, preview, detected = discover_bulk_models(root, model_index=0)

            self.assertEqual(detected, ModelType.BOLTZ)
            self.assertEqual(len(jobs), 1)
            self.assertIn("warning", preview.iloc[0]["SummaryStatus"])

    def test_af3_orphans_are_reported_from_either_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "pae_only_full_data_0.json")
            _touch(root / "structure_only_model_0.cif")

            jobs, preview = discover_af3_models(root, model_index=0)

            self.assertEqual(jobs, [])
            errors = " ".join(preview["Error"].tolist())
            self.assertIn("Missing structure", errors)
            self.assertIn("Missing PAE", errors)

    def test_boltz_orphans_are_reported_from_either_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "pae_pae_only_model_0.npz")
            _touch(root / "structure_only_model_0.cif")

            jobs, preview = discover_boltz_models(root, model_index=0)

            self.assertEqual(jobs, [])
            errors = " ".join(preview["Error"].tolist())
            self.assertIn("Missing structure", errors)
            self.assertIn("Missing PAE", errors)

    def test_ambiguous_boltz_pdb_and_cif_is_not_runnable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "X_model_0.cif")
            _touch(root / "X_model_0.pdb")
            _touch(root / "pae_X_model_0.npz")
            jobs, preview = discover_boltz_models(root, model_index=0)
            self.assertEqual(jobs, [])
            self.assertIn("Ambiguous", preview.iloc[0]["Error"])

    def test_model_index_filtering_selects_only_requested_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in (0, 1):
                _touch(root / f"fold_a_model_{index}.cif")
                _touch(root / f"fold_a_full_data_{index}.json")

            jobs0, preview0 = discover_af3_models(root, model_index=0)
            jobs1, preview1 = discover_af3_models(root, model_index=1)

            self.assertEqual(len(jobs0), 1)
            self.assertEqual(len(jobs1), 1)
            self.assertEqual(int(preview0.iloc[0]["ModelIndex"]), 0)
            self.assertEqual(int(preview1.iloc[0]["ModelIndex"]), 1)


if __name__ == "__main__":
    unittest.main()
