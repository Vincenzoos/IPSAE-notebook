"""Regression tests for ipSAE score parsing (Binder/pandas 3 compatibility)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ipsae_evals_summary import summarize_ipsae_folder
from single_model_eval import parse_ipsae_scores

SAMPLE_SCORE = """\
Chn1 Chn2 PAE Dist Type ipSAE ipTM_af pDockQ LIS Model
A B 10 10 max 0.42 0.55 0.31 0.20 demo_model
A B 10 10 asym 0.38 0.51 0.28 0.18 demo_model
"""


class ParseScoresTests(unittest.TestCase):
    def test_parse_ipsae_scores_coerces_numeric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            score_file = Path(tmp) / "demo_10_10.txt"
            score_file.write_text(SAMPLE_SCORE)
            scores = parse_ipsae_scores(score_file)
            self.assertEqual(len(scores), 2)
            self.assertAlmostEqual(float(scores.iloc[0]["ipSAE"]), 0.42)
            self.assertEqual(scores.iloc[0]["Chn1"], "A")

    def test_summarize_folder_parses_copied_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_dir = root / "fold_demo_model_0"
            job_dir.mkdir()
            (job_dir / "fold_demo_model_0_10_10.txt").write_text(SAMPLE_SCORE)
            (job_dir / "fold_demo_model_0_10_10_byres.txt").write_text("skip\n")
            summary, all_scores = summarize_ipsae_folder(root)
            self.assertFalse(all_scores.empty)
            self.assertFalse(summary.empty)
            self.assertIn("ipSAE", summary.columns)


if __name__ == "__main__":
    unittest.main()
