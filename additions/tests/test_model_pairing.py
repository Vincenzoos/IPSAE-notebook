"""Unit tests for AF2/AF3/Boltz pairing, bulk discovery, and UI state helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bulk_eval import discover_af3_models, discover_boltz_models, discover_bulk_models
from model_pairing import (
    MODEL_TYPE_CHOICES,
    ModelType,
    boltz_companion_paths,
    detect_bulk_model_type,
    expected_boltz_summary_path,
    parse_af2_pae_pairing,
    parse_af2_structure_pairing,
    parse_af3_pae_pairing,
    parse_af3_structure_pairing,
    parse_boltz_pae_pairing,
    parse_boltz_structure_pairing,
    validate_structure_pae_pairing,
)
from single_model_ui_state import (
    clear_incompatible_paths,
    maybe_prefill_boltz_summary,
    placeholders_for,
    selected_model_type,
    single_run_ready,
    upload_extensions_for,
)
from single_model_eval import IpsaeJob, run_ipsae


def _touch(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class Af2PairingTests(unittest.TestCase):
    def test_valid_af2_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(
                root / "RAF1_KSR1_unrelaxed_rank_001_alphafold2_multimer_v3_model_4_seed_003.pdb"
            )
            pae = _touch(
                root / "RAF1_KSR1_scores_rank_001_alphafold2_multimer_v3_model_4_seed_003.json"
            )
            result = validate_structure_pae_pairing(structure, pae, ModelType.AF2)
            self.assertEqual(result["pair"].complex_id, "RAF1_KSR1")
            self.assertEqual(result["pair"].rank, "001")

    def test_af2_rank_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "X_unrelaxed_rank_001_details.pdb")
            pae = _touch(root / "X_scores_rank_002_details.json")
            with self.assertRaises(ValueError) as ctx:
                validate_structure_pae_pairing(structure, pae, ModelType.AF2)
            self.assertIn("complex, rank, and details", str(ctx.exception))

    def test_af2_details_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "X_unrelaxed_rank_001_model_4.pdb")
            pae = _touch(root / "X_scores_rank_001_model_5.json")
            with self.assertRaises(ValueError):
                validate_structure_pae_pairing(structure, pae, ModelType.AF2)

    def test_af2_rejects_relaxed_naming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "X_relaxed_rank_001_details.pdb")
            pae = _touch(root / "X_scores_rank_001_details.json")
            with self.assertRaises(ValueError) as ctx:
                validate_structure_pae_pairing(structure, pae, ModelType.AF2)
            self.assertIn("ColabFold-style", str(ctx.exception))

    def test_af2_rejects_different_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "a" / "X_unrelaxed_rank_001_d.pdb")
            pae = _touch(root / "b" / "X_scores_rank_001_d.json")
            with self.assertRaises(ValueError) as ctx:
                validate_structure_pae_pairing(structure, pae, ModelType.AF2)
            self.assertIn("same parent folder", str(ctx.exception))


class Af3PairingTests(unittest.TestCase):
    def test_valid_af3_server_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "fold_aurka_tpx2_model_0.cif")
            pae = _touch(root / "fold_aurka_tpx2_full_data_0.json")
            result = validate_structure_pae_pairing(structure, pae, ModelType.AF3)
            self.assertEqual(result["pair"].complex_id, "fold_aurka_tpx2")
            self.assertEqual(result["pair"].model_index, 0)

    def test_af3_index_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "fold_x_model_0.cif")
            pae = _touch(root / "fold_x_full_data_1.json")
            with self.assertRaises(ValueError) as ctx:
                validate_structure_pae_pairing(structure, pae, ModelType.AF3)
            self.assertIn("different model", str(ctx.exception))

    def test_af3_complex_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "fold_a_model_0.cif")
            pae = _touch(root / "fold_b_full_data_0.json")
            with self.assertRaises(ValueError) as ctx:
                validate_structure_pae_pairing(structure, pae, ModelType.AF3)
            self.assertIn("different complexes", str(ctx.exception))

    def test_af3_rejects_pdb_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "fold_x_model_0.pdb")
            pae = _touch(root / "fold_x_full_data_0.json")
            with self.assertRaises(ValueError) as ctx:
                validate_structure_pae_pairing(structure, pae, ModelType.AF3)
            self.assertIn(".cif", str(ctx.exception))

    def test_af3_summary_is_accepted_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "fold_x_model_0.cif")
            pae = _touch(root / "fold_x_full_data_0.json")
            summary = _touch(root / "fold_x_summary_confidences_0.json")
            result = validate_structure_pae_pairing(
                structure,
                pae,
                ModelType.AF3,
                summary_file=summary,
            )
            self.assertEqual(result["summary_file"], summary)

    def test_uppercase_extensions_are_rejected_before_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "fold_x_model_0.CIF")
            pae = _touch(root / "fold_x_full_data_0.json")
            with self.assertRaises(ValueError):
                validate_structure_pae_pairing(structure, pae, ModelType.AF3)

    def test_run_ipsae_accepts_discovered_af3_summary(self) -> None:
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


class BoltzPairingTests(unittest.TestCase):
    def test_valid_boltz_cif_and_pdb(self) -> None:
        for ext in (".cif", ".pdb"):
            with self.subTest(ext=ext), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                structure = _touch(root / f"AURKA_TPX2_model_0{ext}")
                pae = _touch(root / "pae_AURKA_TPX2_model_0.npz")
                result = validate_structure_pae_pairing(structure, pae, ModelType.BOLTZ)
                self.assertEqual(result["pair"].complex_id, "AURKA_TPX2")
                self.assertIsNotNone(result["warning"])
                self.assertIn("ipTM", result["warning"])

    def test_boltz_structure_matches_af3_shape_but_uses_boltz_pae(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "fold_like_name_model_0.cif")
            pae = _touch(root / "pae_fold_like_name_model_0.npz")
            # Explicit Boltz type accepts AF3-shaped structure names.
            result = validate_structure_pae_pairing(structure, pae, ModelType.BOLTZ)
            self.assertEqual(result["model_type"], ModelType.BOLTZ)
            # Same files rejected as AF3 because PAE is not full_data JSON.
            with self.assertRaises(ValueError):
                validate_structure_pae_pairing(structure, pae, ModelType.AF3)

    def test_boltz_summary_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "AURKA_TPX2_model_0.cif")
            pae = _touch(root / "pae_AURKA_TPX2_model_0.npz")
            summary = _touch(root / "confidence_AURKA_TPX2_model_0.json")
            result = validate_structure_pae_pairing(
                structure, pae, ModelType.BOLTZ, summary_file=summary
            )
            self.assertIsNone(result["warning"])
            self.assertEqual(Path(result["summary_file"]).name, summary.name)

    def test_boltz_summary_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "AURKA_TPX2_model_0.cif")
            pae = _touch(root / "pae_AURKA_TPX2_model_0.npz")
            summary = _touch(root / "confidence_OTHER_model_0.json")
            with self.assertRaises(ValueError):
                validate_structure_pae_pairing(
                    structure, pae, ModelType.BOLTZ, summary_file=summary
                )

    def test_boltz_summary_malformed_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = _touch(root / "AURKA_TPX2_model_0.cif")
            pae = _touch(root / "pae_AURKA_TPX2_model_0.npz")
            summary = _touch(root / "confidence_AURKA_TPX2_model_0.txt")
            with self.assertRaises(ValueError) as ctx:
                validate_structure_pae_pairing(
                    structure, pae, ModelType.BOLTZ, summary_file=summary
                )
            self.assertIn(".json", str(ctx.exception))

    def test_companion_paths_ignore_parent_pae_substring(self) -> None:
        pae = Path("/data/pae_runs/job1/pae_AURKA_TPX2_model_0.npz")
        confidence, plddt = boltz_companion_paths(pae)
        self.assertEqual(confidence, Path("/data/pae_runs/job1/confidence_AURKA_TPX2_model_0.json"))
        self.assertEqual(plddt, Path("/data/pae_runs/job1/plddt_AURKA_TPX2_model_0.npz"))
        # Whole-path replace would corrupt the parent directory name.
        broken = str(pae).replace("pae", "confidence")
        self.assertIn("confidence_runs", broken)
        self.assertNotIn("confidence_runs", str(confidence))


class IpsaeBoltzCompanionTests(unittest.TestCase):
    def test_ipsae_py_derives_companions_from_basename(self) -> None:
        root = Path(__file__).resolve().parents[2]
        ipsae_path = root / "ipsae.py"
        source = ipsae_path.read_text()
        self.assertIn('pae_name.startswith("pae_")', source)
        self.assertIn('f"confidence_{boltz_base}.json"', source)
        self.assertIn('f"plddt_{boltz_base}.npz"', source)
        self.assertNotIn('pae_file_path.replace("pae"', source)


class BulkDiscoveryTests(unittest.TestCase):
    def test_detect_and_discover_af3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "fold_a"
            _touch(folder / "fold_a_model_0.cif")
            _touch(folder / "fold_a_full_data_0.json")
            _touch(folder / "fold_a_summary_confidences_0.json")
            self.assertEqual(detect_bulk_model_type(root), ModelType.AF3)
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
            self.assertEqual(Path(validation["summary_file"]).name, "fold_a_summary_confidences_0.json")

    def test_detect_and_discover_boltz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "AURKA_TPX2"
            _touch(folder / "AURKA_TPX2_model_0.cif")
            _touch(folder / "pae_AURKA_TPX2_model_0.npz")
            jobs, preview, detected = discover_bulk_models(root, model_index=0)
            self.assertEqual(detected, ModelType.BOLTZ)
            self.assertEqual(len(jobs), 1)
            self.assertIn("warning", preview.iloc[0]["SummaryStatus"])

    def test_mixed_af3_boltz_rejected_even_at_different_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "af3" / "fold_a_full_data_0.json")
            _touch(root / "af3" / "fold_a_model_0.cif")
            _touch(root / "boltz" / "pae_X_model_1.npz")
            _touch(root / "boltz" / "X_model_1.cif")
            with self.assertRaises(ValueError) as ctx:
                detect_bulk_model_type(root)
            self.assertIn("Mixed", str(ctx.exception))

    def test_unsupported_af2_bulk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "X_unrelaxed_rank_001_details.pdb")
            _touch(root / "X_scores_rank_001_details.json")
            with self.assertRaises(ValueError) as ctx:
                detect_bulk_model_type(root)
            self.assertIn("AF2", str(ctx.exception))

    def test_unrecognized_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "notes.txt")
            with self.assertRaises(ValueError) as ctx:
                detect_bulk_model_type(root)
            self.assertIn("Unrecognized", str(ctx.exception))

    def test_incomplete_af3_pair_in_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "fold_a_full_data_0.json")
            jobs, preview = discover_af3_models(root, model_index=0)
            self.assertEqual(jobs, [])
            self.assertFalse(bool(preview.iloc[0]["Ready"]))
            self.assertIn("Missing structure", preview.iloc[0]["Error"])

    def test_af3_structure_without_pae_is_in_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "ready_model_0.cif")
            _touch(root / "ready_full_data_0.json")
            _touch(root / "orphan_model_0.cif")
            jobs, preview = discover_af3_models(root, model_index=0)
            self.assertEqual(len(jobs), 1)
            orphan = preview.loc[preview["Model"] == "orphan_model_0"].iloc[0]
            self.assertFalse(bool(orphan["Ready"]))
            self.assertIn("Missing PAE", orphan["Error"])

    def test_boltz_structure_without_pae_is_in_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "ready_model_0.cif")
            _touch(root / "pae_ready_model_0.npz")
            _touch(root / "orphan_model_0.cif")
            jobs, preview = discover_boltz_models(root, model_index=0)
            self.assertEqual(len(jobs), 1)
            orphan = preview.loc[preview["Model"] == "orphan_model_0"].iloc[0]
            self.assertFalse(bool(orphan["Ready"]))
            self.assertIn("Missing PAE", orphan["Error"])

    def test_ambiguous_boltz_pdb_and_cif(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "X_model_0.cif")
            _touch(root / "X_model_0.pdb")
            _touch(root / "pae_X_model_0.npz")
            jobs, preview = discover_boltz_models(root, model_index=0)
            self.assertEqual(jobs, [])
            self.assertIn("Ambiguous", preview.iloc[0]["Error"])

    def test_model_index_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "fold_a_model_0.cif")
            _touch(root / "fold_a_full_data_0.json")
            _touch(root / "fold_a_model_1.cif")
            _touch(root / "fold_a_full_data_1.json")
            jobs0, preview0 = discover_af3_models(root, model_index=0)
            jobs1, preview1 = discover_af3_models(root, model_index=1)
            self.assertEqual(len(jobs0), 1)
            self.assertEqual(len(jobs1), 1)
            self.assertEqual(int(preview0.iloc[0]["ModelIndex"]), 0)
            self.assertEqual(int(preview1.iloc[0]["ModelIndex"]), 1)

    def test_aux_files_do_not_cause_mixed_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "fold_a_model_0.cif")
            _touch(root / "fold_a_full_data_0.json")
            _touch(root / "fold_a_summary_confidences_0.json")
            _touch(root / "confidence_unrelated_model_0.json")
            _touch(root / "plddt_unrelated_model_0.npz")
            self.assertEqual(detect_bulk_model_type(root), ModelType.AF3)


class UiStateTests(unittest.TestCase):
    def test_dropdown_choices_use_label_value_order(self) -> None:
        import ipywidgets as widgets

        dropdown = widgets.Dropdown(options=list(MODEL_TYPE_CHOICES), value="")
        self.assertEqual(dropdown.label, "Select model type…")
        dropdown.value = "af2"
        self.assertEqual(dropdown.label, "AlphaFold2")

    def test_placeholders_and_extensions(self) -> None:
        self.assertIn("unrelaxed_rank", placeholders_for(ModelType.AF2)["structure"])
        self.assertEqual(upload_extensions_for(ModelType.AF2)["structure"], frozenset({".pdb"}))
        self.assertEqual(upload_extensions_for(ModelType.AF3)["structure"], frozenset({".cif"}))
        self.assertEqual(upload_extensions_for(ModelType.BOLTZ)["pae"], frozenset({".npz"}))

    def test_clear_incompatible_on_type_change(self) -> None:
        cleared = clear_incompatible_paths(
            model_type=ModelType.AF3,
            structure="x.pdb",
            pae="y.json",
            summary="z.json",
        )
        self.assertEqual(cleared["structure"], "")
        self.assertEqual(cleared["pae"], "")
        self.assertEqual(cleared["summary"], "")

        boltz = clear_incompatible_paths(
            model_type=ModelType.BOLTZ,
            structure="x_model_0.cif",
            pae="pae_x_model_0.npz",
            summary="confidence_x_model_0.json",
        )
        self.assertEqual(boltz["structure"], "x_model_0.cif")
        self.assertEqual(boltz["pae"], "pae_x_model_0.npz")
        self.assertEqual(boltz["summary"], "confidence_x_model_0.json")

    def test_run_ready_requires_type_and_paths(self) -> None:
        self.assertFalse(single_run_ready(model_type=None, structure="a.cif", pae="b.json"))
        self.assertFalse(
            single_run_ready(model_type=ModelType.AF3, structure="", pae="b.json")
        )
        self.assertTrue(
            single_run_ready(model_type=ModelType.AF3, structure="a.cif", pae="b.json")
        )

    def test_summary_prefill_only_when_empty_and_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pae = _touch(root / "pae_X_model_0.npz")
            summary = _touch(root / "confidence_X_model_0.json")
            self.assertEqual(
                maybe_prefill_boltz_summary(str(pae), ""),
                str(summary),
            )
            self.assertIsNone(maybe_prefill_boltz_summary(str(pae), "already.json"))

    def test_selected_model_type(self) -> None:
        self.assertIsNone(selected_model_type(""))
        self.assertEqual(selected_model_type("af2"), ModelType.AF2)


class ParseHelpersTests(unittest.TestCase):
    def test_parsers(self) -> None:
        self.assertIsNotNone(
            parse_af2_structure_pairing("c_unrelaxed_rank_001_d.pdb")
        )
        self.assertIsNotNone(parse_af2_pae_pairing("c_scores_rank_001_d.json"))
        self.assertIsNotNone(parse_af3_structure_pairing("c_model_0.cif"))
        self.assertIsNotNone(parse_af3_pae_pairing("c_full_data_0.json"))
        self.assertIsNotNone(parse_boltz_structure_pairing("c_model_0.pdb"))
        self.assertIsNotNone(parse_boltz_pae_pairing("pae_c_model_0.npz"))
        self.assertEqual(
            expected_boltz_summary_path("pae_c_model_0.npz").name,
            "confidence_c_model_0.json",
        )


if __name__ == "__main__":
    unittest.main()
