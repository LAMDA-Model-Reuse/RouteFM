from __future__ import annotations

import unittest

from routefm.data.episode_sampler import EpisodeSampler
from routefm.data.schema import RoutingRecord


class EpisodeCostTest(unittest.TestCase):
    def test_uses_recorded_cost_for_routerbench(self) -> None:
        record = RoutingRecord(
            source="routerbench",
            dataset_id="task",
            query_id="query",
            model_id="WizardLM/WizardLM-13B-V1.2",
            query_embedding=[1.0],
            score=0.8,
            cost=0.003,
            metadata={"cost_available": True},
        )
        self.assertAlmostEqual(EpisodeSampler._point_cost(record), 0.003)

    def test_unavailable_cost_stays_unavailable(self) -> None:
        record = RoutingRecord(
            source="routerbench",
            dataset_id="task",
            query_id="query",
            model_id="claude-v1",
            query_embedding=[1.0],
            score=0.8,
            cost=0.0,
            metadata={"cost_available": False},
        )
        self.assertEqual(EpisodeSampler._point_cost(record), 0.0)


if __name__ == "__main__":
    unittest.main()
