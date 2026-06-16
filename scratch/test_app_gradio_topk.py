import unittest

import pandas as pd
from PIL import Image

from scratch.app_gradio import (
    _draw_mcs_change_highlight,
    _ppm_pass_column,
    _rank_candidates_for_display,
    _select_primary_prediction,
)


def _count_highlight_pixels(image: Image.Image) -> tuple[int, int]:
    red = green = 0
    rgb_image = image.convert("RGB")
    pixels = (
        rgb_image.get_flattened_data()
        if hasattr(rgb_image, "get_flattened_data")
        else rgb_image.getdata()
    )
    for r, g, b in pixels:
        if r > 180 and g < 140 and b < 140:
            red += 1
        if g > 150 and r < 170 and b < 170:
            green += 1
    return red, green


class TopKCandidateDisplayTest(unittest.TestCase):
    def test_rank_candidates_deduplicates_and_sorts_by_ppm(self):
        table = _rank_candidates_for_display(
            parent="CC",
            delta=15.9949,
            candidates=["CCC", "CCO", "OCC", "not_a_smiles"],
            target_smiles="CCO",
            limit=10,
        )

        self.assertEqual(list(table["rank"])[:2], [1, 2])
        self.assertEqual(table.iloc[0]["candidate_smiles"], "CCO")
        self.assertLess(table.iloc[0]["ppm_error"], table.iloc[1]["ppm_error"])
        self.assertEqual(table.iloc[0]["exact_match"], True)
        self.assertEqual(table["candidate_smiles"].tolist().count("CCO"), 1)
        self.assertEqual(table.iloc[-1]["validity"], False)

    def test_rank_candidates_only_ppm_pass_filters_and_keeps_all(self):
        table = _rank_candidates_for_display(
            parent="CC",
            delta=15.9949,
            candidates=["CCC", "CCO", "OCC", "CCN"],
            limit=None,
            only_ppm_pass=True,
        )
        ppm_col = _ppm_pass_column()

        self.assertFalse(table.empty)
        self.assertTrue((table[ppm_col]).all())
        self.assertIn("CCO", table["candidate_smiles"].tolist())

    def test_select_primary_prediction_prefers_ppm_pass_table(self):
        ppm_pass = _rank_candidates_for_display(
            parent="CC",
            delta=15.9949,
            candidates=["CCO"],
            limit=None,
            only_ppm_pass=True,
        )
        top10 = _rank_candidates_for_display(
            parent="CC",
            delta=15.9949,
            candidates=["CCC", "CCO"],
            limit=10,
        )

        self.assertEqual(_select_primary_prediction(ppm_pass, top10), "CCO")
        self.assertEqual(_select_primary_prediction(pd.DataFrame(), top10), "CCO")

    def test_draw_mcs_change_highlight_returns_image_for_valid_pair(self):
        image = _draw_mcs_change_highlight("CC", "CCO")

        self.assertIsInstance(image, Image.Image)
        self.assertGreater(image.size[0], 0)
        self.assertGreater(image.size[1], 0)

    def test_draw_mcs_change_highlight_returns_none_for_invalid_product(self):
        self.assertIsNone(_draw_mcs_change_highlight("CC", "not_a_smiles"))

    def test_draw_mcs_change_highlight_uses_green_for_common_scaffold(self):
        image = _draw_mcs_change_highlight(
            "CCNC1=NC(NC(C)C)=NC(Cl)=N1",
            "CCNC1=NC(NC(C)C)=NC(O)=N1",
        )
        self.assertIsNotNone(image)
        red, green = _count_highlight_pixels(image)

        self.assertGreater(green, red)


if __name__ == "__main__":
    unittest.main()
