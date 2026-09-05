import numpy as np
import unittest

from envs.modules.HRL_Coordinator import HRL_Coordinator


class LowCandidateMaskAlignmentTests(unittest.TestCase):
    def test_filters_candidates_and_features_with_final_mask(self):
        candidate_info = {
            "indices": [2, 4, 6],
            "current_node": 1,
            "features": np.asarray(
                [[2.0] * 6, [4.0] * 6, [6.0] * 6], dtype=np.float32
            ),
        }
        mask = np.zeros(8, dtype=np.float32)
        mask[[4, 7]] = 1.0

        aligned = HRL_Coordinator._align_low_candidates_with_mask(
            candidate_info, mask, current_node=1
        )

        self.assertEqual(aligned["indices"], [4])
        np.testing.assert_array_equal(
            aligned["features"], np.asarray([[4.0] * 6], dtype=np.float32)
        )

    def test_rebuilds_candidates_when_path_shield_excludes_old_topk(self):
        candidate_info = {
            "indices": [2, 4],
            "current_node": 1,
            "features": np.ones((2, 6), dtype=np.float32),
        }
        mask = np.zeros(8, dtype=np.float32)
        mask[7] = 1.0

        aligned = HRL_Coordinator._align_low_candidates_with_mask(
            candidate_info, mask, current_node=1
        )

        self.assertEqual(aligned["indices"], [7])
        self.assertIsNone(aligned["features"])
        self.assertIs(aligned["realigned_from_mask"], True)

    def test_returns_none_for_dead_end_mask(self):
        aligned = HRL_Coordinator._align_low_candidates_with_mask(
            {"indices": [1], "current_node": 0, "features": None},
            np.zeros(3, dtype=np.float32),
            current_node=0,
        )

        self.assertIsNone(aligned)


if __name__ == "__main__":
    unittest.main()
