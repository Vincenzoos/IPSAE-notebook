"""Single-model input cases: valid pairs, within-type mismatches, and cross-type mixes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_pairing import BOLTZ_MISSING_SUMMARY_WARNING, ModelType, validate_structure_pae_pairing
from single_model_ui_state import (
    clear_incompatible_paths,
    maybe_prefill_boltz_summary,
    path_compatible_with_type,
    selected_model_type,
    single_inputs_locked,
    single_run_ready,
)
from single_model_eval import IpsaeJob, run_ipsae


def _touch(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


# Canonical filenames used across cases.
AF2_STRUCTURE = "X_unrelaxed_rank_001_details.pdb"
AF2_PAE = "X_scores_rank_001_details.json"
AF3_STRUCTURE = "fold_x_model_0.cif"
AF3_PAE = "fold_x_full_data_0.json"
BOLTZ_STRUCTURE_CIF = "X_model_0.cif"
BOLTZ_STRUCTURE_PDB = "X_model_0.pdb"
BOLTZ_PAE = "pae_X_model_0.npz"


class SingleModelValidPairTests(unittest.TestCase):
    def test_valid_pair_for_each_selected_type(self) -> None:
        cases = [
            (ModelType.AF2, AF2_STRUCTURE, AF2_PAE, "X"),
            (ModelType.AF3, AF3_STRUCTURE, AF3_PAE, "fold_x"),
            (ModelType.BOLTZ, BOLTZ_STRUCTURE_CIF, BOLTZ_PAE, "X"),
            (ModelType.BOLTZ, BOLTZ_STRUCTURE_PDB, BOLTZ_PAE, "X"),
        ]
        for model_type, structure_name, pae_name, complex_id in cases:
            with self.subTest(model_type=model_type, structure=structure_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                structure = _touch(root / structure_name)
                pae = _touch(root / pae_name)
                result = validate_structure_pae_pairing(structure, pae, model_type)
                self.assertEqual(result["model_type"], model_type)
                self.assertEqual(result["pair"].complex_id, complex_id)


class SingleModelWithinTypeMismatchTests(unittest.TestCase):
    def test_af2_complex_rank_or_details_mismatch(self) -> None:
        cases = [
            ("A_unrelaxed_rank_001_d.pdb", "B_scores_rank_001_d.json"),
            ("X_unrelaxed_rank_001_d.pdb", "X_scores_rank_002_d.json"),
            ("X_unrelaxed_rank_001_model_4.pdb", "X_scores_rank_001_model_5.json"),
        ]
        for structure_name, pae_name in cases:
            with self.subTest(structure=structure_name, pae=pae_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                structure = _touch(root / structure_name)
                pae = _touch(root / pae_name)
                with self.assertRaises(ValueError):
                    validate_structure_pae_pairing(structure, pae, ModelType.AF2)

    def test_af3_complex_or_index_mismatch(self) -> None:
        cases = [
            ("fold_a_model_0.cif", "fold_b_full_data_0.json"),
            ("fold_x_model_0.cif", "fold_x_full_data_1.json"),
        ]
        for structure_name, pae_name in cases:
            with self.subTest(structure=structure_name, pae=pae_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                structure = _touch(root / structure_name)
                pae = _touch(root / pae_name)
                with self.assertRaises(ValueError):
                    validate_structure_pae_pairing(structure, pae, ModelType.AF3)

    def test_boltz_complex_or_index_mismatch(self) -> None:
        cases = [
            ("A_model_0.cif", "pae_B_model_0.npz"),
            ("X_model_0.cif", "pae_X_model_1.npz"),
        ]
        for structure_name, pae_name in cases:
            with self.subTest(structure=structure_name, pae=pae_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                structure = _touch(root / structure_name)
                pae = _touch(root / pae_name)
                with self.assertRaises(ValueError):
                    validate_structure_pae_pairing(structure, pae, ModelType.BOLTZ)

    def test_same_type_pair_in_different_folders_rejected(self) -> None:
        cases = [
            (ModelType.AF2, AF2_STRUCTURE, AF2_PAE),
            (ModelType.AF3, AF3_STRUCTURE, AF3_PAE),
            (ModelType.BOLTZ, BOLTZ_STRUCTURE_CIF, BOLTZ_PAE),
        ]
        for model_type, structure_name, pae_name in cases:
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                structure = _touch(root / "a" / structure_name)
                pae = _touch(root / "b" / pae_name)
                with self.assertRaises(ValueError) as ctx:
                    validate_structure_pae_pairing(structure, pae, model_type)
                self.assertIn("same parent folder", str(ctx.exception))


class SingleModelMalformedInputTests(unittest.TestCase):
    def test_malformed_names_are_rejected_for_each_type(self) -> None:
        cases = [
            (ModelType.AF2, "X_relaxed_rank_001_d.pdb", "X_scores_rank_001_d.json", "ColabFold-style"),
            (ModelType.AF2, "X_unrelaxed_rank_x_d.pdb", "X_scores_rank_x_d.json", "ColabFold-style"),
            (ModelType.AF3, "X_model_x.cif", "X_full_data_0.json", "expected a structure"),
            (ModelType.AF3, "X_model_0.cif", "X_confidences_0.json", "expected a structure"),
            (ModelType.BOLTZ, "X_structure_0.cif", "pae_X_model_0.npz", "expected structure"),
            (ModelType.BOLTZ, "X_model_0.cif", "X_model_0.npz", "expected structure"),
        ]
        for model_type, structure_name, pae_name, message in cases:
            with self.subTest(model_type=model_type, structure=structure_name, pae=pae_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                structure = _touch(root / structure_name)
                pae = _touch(root / pae_name)
                with self.assertRaisesRegex(ValueError, message):
                    validate_structure_pae_pairing(structure, pae, model_type)

    def test_wrong_and_uppercase_extensions_are_rejected(self) -> None:
        cases = [
            (ModelType.AF2, "X_unrelaxed_rank_001_d.cif", AF2_PAE, "\\.pdb"),
            (ModelType.AF2, AF2_STRUCTURE, "X_scores_rank_001_d.npz", "\\.json"),
            (ModelType.AF3, "X_model_0.CIF", "X_full_data_0.json", "\\.cif"),
            (ModelType.AF3, "X_model_0.cif", "X_full_data_0.JSON", "\\.json"),
            (ModelType.BOLTZ, "X_model_0.xyz", BOLTZ_PAE, "\\.pdb or \\.cif"),
            (ModelType.BOLTZ, BOLTZ_STRUCTURE_CIF, "pae_X_model_0.NPZ", "\\.npz"),
        ]
        for model_type, structure_name, pae_name, message in cases:
            with self.subTest(model_type=model_type, structure=structure_name, pae=pae_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                structure = _touch(root / structure_name)
                pae = _touch(root / pae_name)
                with self.assertRaisesRegex(ValueError, message):
                    validate_structure_pae_pairing(structure, pae, model_type)

    def test_missing_files_and_directories_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing_pae = _touch(root / AF3_PAE)
            with self.assertRaisesRegex(FileNotFoundError, "Structure file not found"):
                validate_structure_pae_pairing(root / AF3_STRUCTURE, existing_pae, ModelType.AF3)

            existing_structure = _touch(root / AF3_STRUCTURE)
            missing_pae = root / "missing_full_data_0.json"
            with self.assertRaisesRegex(FileNotFoundError, "PAE file not found"):
                validate_structure_pae_pairing(existing_structure, missing_pae, ModelType.AF3)

            structure_dir = root / "directory_model_0.cif"
            structure_dir.mkdir()
            pae = _touch(root / "directory_full_data_0.json")
            with self.assertRaisesRegex(ValueError, "regular file"):
                validate_structure_pae_pairing(structure_dir, pae, ModelType.AF3)

    def test_require_exists_false_validates_names_without_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = validate_structure_pae_pairing(
                root / AF3_STRUCTURE,
                root / AF3_PAE,
                ModelType.AF3,
                require_exists=False,
            )
            self.assertEqual(result["pair"].complex_id, "fold_x")


class SingleModelSummaryTests(unittest.TestCase):
    def test_af3_explicit_and_automatic_summary_discovery(self) -> None:
        for explicit in (False, True):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                structure = _touch(root / AF3_STRUCTURE)
                pae = _touch(root / AF3_PAE)
                summary = _touch(root / "fold_x_summary_confidences_0.json")
                result = validate_structure_pae_pairing(
                    structure,
                    pae,
                    ModelType.AF3,
                    summary_file=summary if explicit else None,
                )
                self.assertEqual(result["summary_file"], summary)

    def test_boltz_explicit_and_automatic_summary_discovery(self) -> None:
        for explicit in (False, True):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                structure = _touch(root / BOLTZ_STRUCTURE_CIF)
                pae = _touch(root / BOLTZ_PAE)
                summary = _touch(root / "confidence_X_model_0.json")
                result = validate_structure_pae_pairing(
                    structure,
                    pae,
                    ModelType.BOLTZ,
                    summary_file=summary if explicit else None,
                )
                self.assertEqual(result["summary_file"], summary)
                self.assertIsNone(result["warning"])

    def test_missing_boltz_summary_returns_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = validate_structure_pae_pairing(
                _touch(root / BOLTZ_STRUCTURE_CIF),
                _touch(root / BOLTZ_PAE),
                ModelType.BOLTZ,
            )
            self.assertIsNone(result["summary_file"])
            self.assertEqual(result["warning"], BOLTZ_MISSING_SUMMARY_WARNING)

    def test_mismatched_or_misplaced_summaries_are_rejected(self) -> None:
        cases = [
            (ModelType.AF3, AF3_STRUCTURE, AF3_PAE, "other_summary_confidences_0.json", "summary filename mismatch"),
            (ModelType.BOLTZ, BOLTZ_STRUCTURE_CIF, BOLTZ_PAE, "confidence_X_model_1.json", "summary mismatch"),
        ]
        for model_type, structure_name, pae_name, summary_name, message in cases:
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                structure = _touch(root / structure_name)
                pae = _touch(root / pae_name)
                summary = _touch(root / summary_name)
                with self.assertRaisesRegex(ValueError, message):
                    validate_structure_pae_pairing(structure, pae, model_type, summary_file=summary)

        folder_cases = [
            (ModelType.AF3, AF3_STRUCTURE, AF3_PAE, "fold_x_summary_confidences_0.json"),
            (ModelType.BOLTZ, BOLTZ_STRUCTURE_CIF, BOLTZ_PAE, "confidence_X_model_0.json"),
        ]
        for model_type, structure_name, pae_name, summary_name in folder_cases:
            with self.subTest(model_type=model_type, different_folder=True), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                structure = _touch(root / "inputs" / structure_name)
                pae = _touch(root / "inputs" / pae_name)
                summary = _touch(root / "summary" / summary_name)
                with self.assertRaisesRegex(ValueError, "same parent folder"):
                    validate_structure_pae_pairing(structure, pae, model_type, summary_file=summary)

    def test_summary_extension_and_missing_summary_file_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / BOLTZ_STRUCTURE_CIF)
            pae = _touch(root / BOLTZ_PAE)
            wrong_extension = _touch(root / "confidence_X_model_0.txt")
            with self.assertRaisesRegex(ValueError, "\\.json"):
                validate_structure_pae_pairing(
                    structure, pae, ModelType.BOLTZ, summary_file=wrong_extension
                )
            with self.assertRaisesRegex(FileNotFoundError, "summary file not found"):
                validate_structure_pae_pairing(
                    structure,
                    pae,
                    ModelType.BOLTZ,
                    summary_file=root / "confidence_X_model_0.json",
                )

    def test_af2_rejects_summary_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Summary file"):
                validate_structure_pae_pairing(
                    _touch(root / AF2_STRUCTURE),
                    _touch(root / AF2_PAE),
                    ModelType.AF2,
                    summary_file=_touch(root / "summary.json"),
                )


class SingleModelExecutionIntegrationTests(unittest.TestCase):
    def test_run_ipsae_uses_validated_af3_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "fold_x_model_0.cif")
            pae = _touch(root / "fold_x_full_data_0.json", "{}")
            summary = _touch(root / "fold_x_summary_confidences_0.json", "{}")
            _touch(
                root / "fold_x_model_0_10_10.txt",
                "Chn1 Chn2 Type Model ipSAE\nA B max fold_x 0.5\n",
            )

            scores, warning = run_ipsae(
                IpsaeJob(
                    label="fold_x",
                    structure_file=structure,
                    pae_file=pae,
                    model_type=ModelType.AF3,
                    summary_file=summary,
                ),
                overwrite=False,
            )

            self.assertIsNone(warning)
            self.assertEqual(scores.loc[0, "Model_Type"], "af3")
            self.assertEqual(Path(scores.loc[0, "Summary_File"]).name, summary.name)

    def test_ipsae_script_derives_boltz_companions_from_basename(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        source = (repository / "ipsae.py").read_text()
        self.assertIn('pae_name.startswith("pae_")', source)
        self.assertIn('f"confidence_{boltz_base}.json"', source)
        self.assertIn('f"plddt_{boltz_base}.npz"', source)
        self.assertNotIn('pae_file_path.replace("pae"', source)


class SingleModelCrossTypeTests(unittest.TestCase):
    """Structure from one family + PAE from another, validated under an explicit selected type."""

    def test_selected_af2_rejects_non_af2_partners(self) -> None:
        cases = [
            (AF2_STRUCTURE, AF3_PAE),
            (AF2_STRUCTURE, BOLTZ_PAE),
            (AF3_STRUCTURE, AF2_PAE),
            (BOLTZ_STRUCTURE_CIF, AF2_PAE),
            (BOLTZ_STRUCTURE_PDB, AF2_PAE),
            (AF3_STRUCTURE, AF3_PAE),  # valid AF3 pair, wrong selected type
            (BOLTZ_STRUCTURE_CIF, BOLTZ_PAE),  # valid Boltz pair, wrong selected type
        ]
        for structure_name, pae_name in cases:
            with self.subTest(structure=structure_name, pae=pae_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                structure = _touch(root / structure_name)
                pae = _touch(root / pae_name)
                with self.assertRaises(ValueError):
                    validate_structure_pae_pairing(structure, pae, ModelType.AF2)

    def test_selected_af3_rejects_non_af3_partners(self) -> None:
        cases = [
            (AF3_STRUCTURE, AF2_PAE),
            (AF3_STRUCTURE, BOLTZ_PAE),
            (AF2_STRUCTURE, AF3_PAE),
            (BOLTZ_STRUCTURE_PDB, AF3_PAE),  # Boltz pdb is not valid AF3 structure
            (AF2_STRUCTURE, AF2_PAE),  # valid AF2 pair, wrong selected type
            (BOLTZ_STRUCTURE_PDB, BOLTZ_PAE),  # valid Boltz pdb pair, wrong selected type
        ]
        for structure_name, pae_name in cases:
            with self.subTest(structure=structure_name, pae=pae_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                structure = _touch(root / structure_name)
                pae = _touch(root / pae_name)
                with self.assertRaises(ValueError):
                    validate_structure_pae_pairing(structure, pae, ModelType.AF3)

    def test_selected_boltz_rejects_non_boltz_partners(self) -> None:
        cases = [
            (BOLTZ_STRUCTURE_CIF, AF2_PAE),
            (BOLTZ_STRUCTURE_CIF, AF3_PAE),
            (BOLTZ_STRUCTURE_PDB, AF3_PAE),
            (AF2_STRUCTURE, BOLTZ_PAE),
            (AF2_STRUCTURE, AF2_PAE),  # valid AF2 pair, wrong selected type
            (AF3_STRUCTURE, AF3_PAE),  # valid AF3 pair, wrong selected type (PAE not npz)
        ]
        for structure_name, pae_name in cases:
            with self.subTest(structure=structure_name, pae=pae_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                structure = _touch(root / structure_name)
                pae = _touch(root / pae_name)
                with self.assertRaises(ValueError):
                    validate_structure_pae_pairing(structure, pae, ModelType.BOLTZ)

    def test_af3_shaped_cif_plus_boltz_pae_passes_only_as_boltz(self) -> None:
        """AF3/Boltz share *_model_N.cif naming; PAE decides when selected type is Boltz."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # AF3-looking complex name, but paired with Boltz PAE.
            structure = _touch(root / "fold_x_model_0.cif")
            pae = _touch(root / "pae_fold_x_model_0.npz")
            ok = validate_structure_pae_pairing(structure, pae, ModelType.BOLTZ)
            self.assertEqual(ok["model_type"], ModelType.BOLTZ)
            self.assertEqual(ok["pair"].complex_id, "fold_x")
            with self.assertRaises(ValueError):
                validate_structure_pae_pairing(structure, pae, ModelType.AF3)
            with self.assertRaises(ValueError):
                validate_structure_pae_pairing(structure, pae, ModelType.AF2)

    def test_boltz_shaped_cif_plus_af3_pae_passes_only_as_af3(self) -> None:
        """Same shared cif shape with AF3 PAE is accepted only under AF3 selection."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "X_model_0.cif")
            pae = _touch(root / "X_full_data_0.json")
            ok = validate_structure_pae_pairing(structure, pae, ModelType.AF3)
            self.assertEqual(ok["model_type"], ModelType.AF3)
            self.assertEqual(ok["pair"].complex_id, "X")
            with self.assertRaises(ValueError):
                validate_structure_pae_pairing(structure, pae, ModelType.BOLTZ)
            with self.assertRaises(ValueError):
                validate_structure_pae_pairing(structure, pae, ModelType.AF2)

    def test_swapped_af3_and_boltz_files_fail_for_both_selected_types(self) -> None:
        """True cross-type mix: AF3 structure with Boltz PAE under wrong complex pairing names."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / AF3_STRUCTURE)  # fold_x_model_0.cif
            pae = _touch(root / BOLTZ_PAE)  # pae_X_model_0.npz (different complex)
            with self.assertRaises(ValueError):
                validate_structure_pae_pairing(structure, pae, ModelType.AF3)
            with self.assertRaises(ValueError):
                validate_structure_pae_pairing(structure, pae, ModelType.BOLTZ)


class SingleModelUiGateTests(unittest.TestCase):
    def test_inputs_locked_until_model_type_selected(self) -> None:
        self.assertTrue(single_inputs_locked(model_type=None))
        self.assertFalse(single_run_ready(model_type=None, structure=AF3_STRUCTURE, pae=AF3_PAE))
        for model_type in (ModelType.AF2, ModelType.AF3, ModelType.BOLTZ):
            with self.subTest(model_type=model_type):
                self.assertFalse(single_inputs_locked(model_type=model_type))
                self.assertTrue(
                    single_run_ready(
                        model_type=model_type,
                        structure="structure.file",
                        pae="pae.file",
                    )
                )

    def test_running_or_blank_paths_disable_run(self) -> None:
        self.assertTrue(single_inputs_locked(model_type=ModelType.AF3, running=True))
        self.assertFalse(
            single_run_ready(
                model_type=ModelType.AF3,
                structure=AF3_STRUCTURE,
                pae=AF3_PAE,
                running=True,
            )
        )
        for structure, pae in (("", AF3_PAE), ("   ", AF3_PAE), (AF3_STRUCTURE, ""), (AF3_STRUCTURE, "\t")):
            with self.subTest(structure=structure, pae=pae):
                self.assertFalse(
                    single_run_ready(
                        model_type=ModelType.AF3,
                        structure=structure,
                        pae=pae,
                    )
                )

    def test_path_compatible_only_against_selected_type(self) -> None:
        self.assertTrue(path_compatible_with_type(AF2_STRUCTURE, ModelType.AF2, "structure"))
        self.assertFalse(path_compatible_with_type(AF2_STRUCTURE, ModelType.AF3, "structure"))
        self.assertFalse(path_compatible_with_type(AF2_STRUCTURE, ModelType.BOLTZ, "structure"))

        self.assertTrue(path_compatible_with_type(AF3_STRUCTURE, ModelType.AF3, "structure"))
        self.assertTrue(path_compatible_with_type(AF3_STRUCTURE, ModelType.BOLTZ, "structure"))
        self.assertFalse(path_compatible_with_type(AF3_STRUCTURE, ModelType.AF2, "structure"))

        self.assertTrue(path_compatible_with_type(AF3_PAE, ModelType.AF3, "pae"))
        self.assertFalse(path_compatible_with_type(AF3_PAE, ModelType.BOLTZ, "pae"))
        self.assertFalse(path_compatible_with_type(AF3_PAE, ModelType.AF2, "pae"))

        self.assertTrue(path_compatible_with_type(BOLTZ_PAE, ModelType.BOLTZ, "pae"))
        self.assertFalse(path_compatible_with_type(BOLTZ_PAE, ModelType.AF3, "pae"))
        self.assertFalse(path_compatible_with_type(BOLTZ_PAE, ModelType.AF2, "pae"))

    def test_empty_paths_are_compatible_and_malformed_names_are_not(self) -> None:
        for empty in ("", "  "):
            with self.subTest(empty=empty):
                self.assertTrue(path_compatible_with_type(empty, ModelType.AF3, "structure"))
        self.assertFalse(path_compatible_with_type("X_model_x.cif", ModelType.AF3, "structure"))
        self.assertFalse(path_compatible_with_type("X_full_data_x.json", ModelType.AF3, "pae"))
        self.assertFalse(
            path_compatible_with_type("confidence_X_model_x.json", ModelType.BOLTZ, "summary")
        )

    def test_clear_incompatible_paths_for_cross_type_inputs(self) -> None:
        # AF2 paths cleared when switching to AF3.
        cleared_af3 = clear_incompatible_paths(
            model_type=ModelType.AF3,
            structure=AF2_STRUCTURE,
            pae=AF2_PAE,
            summary="",
        )
        self.assertEqual(cleared_af3["structure"], "")
        self.assertEqual(cleared_af3["pae"], "")

        # AF3 PAE cleared under Boltz; shared cif structure kept.
        cleared_boltz = clear_incompatible_paths(
            model_type=ModelType.BOLTZ,
            structure=AF3_STRUCTURE,
            pae=AF3_PAE,
            summary="",
        )
        self.assertEqual(cleared_boltz["structure"], AF3_STRUCTURE)
        self.assertEqual(cleared_boltz["pae"], "")

        # Boltz PAE cleared under AF3; shared cif structure kept.
        cleared_to_af3 = clear_incompatible_paths(
            model_type=ModelType.AF3,
            structure=BOLTZ_STRUCTURE_CIF,
            pae=BOLTZ_PAE,
            summary="",
        )
        self.assertEqual(cleared_to_af3["structure"], BOLTZ_STRUCTURE_CIF)
        self.assertEqual(cleared_to_af3["pae"], "")

        # No type selected clears everything.
        cleared_none = clear_incompatible_paths(
            model_type=None,
            structure=AF3_STRUCTURE,
            pae=AF3_PAE,
            summary="confidence_X_model_0.json",
        )
        self.assertEqual(cleared_none, {"structure": "", "pae": "", "summary": ""})

    def test_clear_incompatible_paths_retains_valid_boltz_inputs(self) -> None:
        valid = clear_incompatible_paths(
            model_type=ModelType.BOLTZ,
            structure=BOLTZ_STRUCTURE_PDB,
            pae=BOLTZ_PAE,
            summary="confidence_X_model_0.json",
        )
        self.assertEqual(
            valid,
            {
                "structure": BOLTZ_STRUCTURE_PDB,
                "pae": BOLTZ_PAE,
                "summary": "confidence_X_model_0.json",
            },
        )
        invalid_summary = clear_incompatible_paths(
            model_type=ModelType.BOLTZ,
            structure=BOLTZ_STRUCTURE_CIF,
            pae=BOLTZ_PAE,
            summary="X_summary_confidences_0.json",
        )
        self.assertEqual(invalid_summary["summary"], "")

    def test_summary_prefill_requires_empty_value_and_existing_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pae = _touch(root / BOLTZ_PAE)
            summary = _touch(root / "confidence_X_model_0.json")
            self.assertEqual(maybe_prefill_boltz_summary(str(pae), ""), str(summary))
            self.assertIsNone(maybe_prefill_boltz_summary(str(pae), "manual.json"))
            self.assertIsNone(maybe_prefill_boltz_summary("", ""))
            self.assertIsNone(maybe_prefill_boltz_summary(str(root / "pae_Y_model_0.npz"), ""))
            self.assertIsNone(maybe_prefill_boltz_summary(str(root / "wrong.json"), ""))

    def test_selected_model_type_handles_empty_alias_and_unknown_values(self) -> None:
        self.assertIsNone(selected_model_type(None))
        self.assertIsNone(selected_model_type("  "))
        self.assertEqual(selected_model_type("AlphaFold Server"), ModelType.AF3)
        with self.assertRaisesRegex(ValueError, "Unknown model type"):
            selected_model_type("other")


if __name__ == "__main__":
    unittest.main()
