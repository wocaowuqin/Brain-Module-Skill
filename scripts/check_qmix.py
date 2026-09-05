#!/usr/bin/env python3
"""Small deterministic shape/update smoke test for the QMIX learner."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.qmix import QMIXLearner
from core.marl.trainable_role_agents import (
    TrainableSFTSelectionAgent, TrainableTreeRerouteAgent, TrainableVNFMigrationAgent,
)


def main() -> int:
    agents = [TrainableSFTSelectionAgent(top_k=3), TrainableVNFMigrationAgent(), TrainableTreeRerouteAgent()]
    learner = QMIXLearner(agents, mixer_embed_dim=16, target_update_interval=2)
    obs = [[0.1] * agent.config.obs_dim for agent in agents]
    next_obs = [[0.2] * agent.config.obs_dim for agent in agents]
    for i in range(8):
        learner.push_transition(obs, [i % agent.config.action_dim for agent in agents], 1.0, next_obs,
                                [[0], [0], [0]], False)
    result = learner.update_from_replay(batch_size=4)
    assert result["updated"] == 1.0 and result["loss"] >= 0.0
    print(json.dumps({"ok": True, "state_dim": learner.state_dim, "replay": len(learner.replay), **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
