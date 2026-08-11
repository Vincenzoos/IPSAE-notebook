"""Focused unit tests for shared model-pairing parsers and path helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from model_pairing import (
    MODEL_TYPE_CHOICES,
    ModelType,
    af3_summary_path_for,
    boltz_companion_paths,
    expected_boltz_summary_path,
    parse_af2_pae_pairing,
    parse_af2_structure_pairing,
    parse_af3_pae_pairing,
    parse_af3_structure_pairing,
    parse_af3_summary_pairing,
    parse_boltz_pae_pairing,
    parse_boltz_structure_pairing,
    parse_boltz_summary_pairing,
    parse_model_type,
)


class ModelTypeTests(unittest.TestCase):
    def test_choices_use_label_value_order(self) -> None:
        self.assertEqual(MODEL_TYPE_CHOICES[0], ("Select model type…", ""))
        self.assertIn(("AlphaFold2", "af2"), MODEL_TYPE_CHOICES)
        self.assertIn(("AlphaFold3", "af3"), MODEL_TYPE_CHOICES)
        self.assertIn(("Boltz", "boltz"), MODEL_TYPE_CHOICES)

    def test_model_type_aliases(self) -> None:
        cases = {
            "af2": ModelType.AF2,
            " AlphaFold2 ": ModelType.AF2,
            "af3": ModelType.AF3,
            "AlphaFold Server": ModelType.AF3,
            "boltz": ModelType.BOLTZ,
            "Boltz2": ModelType.BOLTZ,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(parse_model_type(value), expected)

    def test_missing_or_unknown_model_type_is_rejected(self) -> None:
        for value in (None, "", "  ", "unknown"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_model_type(value)


class ParseHelpersTests(unittest.TestCase):
    def test_valid_parsers_return_pair_metadata(self) -> None:
        af2_structure = parse_af2_structure_pairing("c_unrelaxed_rank_001_d.pdb")
        af2_pae = parse_af2_pae_pairing("c_scores_rank_001_d.json")
        af3_structure = parse_af3_structure_pairing("c_model_12.cif")
        af3_pae = parse_af3_pae_pairing("c_full_data_12.json")
        af3_summary = parse_af3_summary_pairing("c_summary_confidences_12.json")
        boltz_structure = parse_boltz_structure_pairing("c_model_12.pdb")
        boltz_pae = parse_boltz_pae_pairing("pae_c_model_12.npz")
        boltz_summary = parse_boltz_summary_pairing("confidence_c_model_12.json")

        self.assertEqual((af2_structure.complex_id, af2_structure.rank, af2_structure.details), ("c", "001", "d"))
        self.assertEqual(af2_structure, af2_pae)
        self.assertEqual(af3_structure, af3_pae)
        self.assertEqual(af3_structure, af3_summary)
        self.assertEqual(boltz_structure, boltz_pae)
        self.assertEqual(boltz_structure, boltz_summary)

    def test_parsers_reject_wrong_extensions_or_malformed_names(self) -> None:
        cases = [
            (parse_af2_structure_pairing, "c_unrelaxed_rank_001_d.cif"),
            (parse_af2_structure_pairing, "c_relaxed_rank_001_d.pdb"),
            (parse_af2_pae_pairing, "c_scores_rank_x_d.json"),
            (parse_af3_structure_pairing, "c_model_x.cif"),
            (parse_af3_pae_pairing, "c_full_data_0.npz"),
            (parse_af3_summary_pairing, "c_summary_confidence_0.json"),
            (parse_boltz_structure_pairing, "c_model_0.xyz"),
            (parse_boltz_pae_pairing, "c_model_0.npz"),
            (parse_boltz_summary_pairing, "confidence_c_model_x.json"),
        ]
        for parser, filename in cases:
            with self.subTest(parser=parser.__name__, filename=filename):
                self.assertIsNone(parser(filename))


class CompanionPathTests(unittest.TestCase):
    def test_boltz_companion_paths_only_transform_basename(self) -> None:
        pae = Path("/data/pae_runs/job1/pae_AURKA_TPX2_model_0.npz")
        confidence, plddt = boltz_companion_paths(pae)
        self.assertEqual(confidence, Path("/data/pae_runs/job1/confidence_AURKA_TPX2_model_0.json"))
        self.assertEqual(plddt, Path("/data/pae_runs/job1/plddt_AURKA_TPX2_model_0.npz"))
        self.assertEqual(expected_boltz_summary_path(pae), confidence)

    def test_invalid_boltz_pae_cannot_produce_companions(self) -> None:
        for filename in ("X_model_0.npz", "pae_X_model_0.json", "pae_.npz"):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                boltz_companion_paths(filename)

    def test_af3_summary_path_is_a_sibling(self) -> None:
        source = Path("/data/fold_x/fold_x_full_data_7.json")
        self.assertEqual(
            af3_summary_path_for(source, "fold_x", 7),
            Path("/data/fold_x/fold_x_summary_confidences_7.json"),
        )


if __name__ == "__main__":
    unittest.main()
