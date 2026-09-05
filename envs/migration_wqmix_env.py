"""Offline migration environment using the online joint decoder semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence
import warnings

import torch

from core.marl.batch_deployment_wqmix import ResourceSnapshot
from core.marl.deployment_topk import deserialize_footprint
from core.marl.joint_candidate_decoder import JointDecodeResult, decode_joint_candidates
from core.marl.migration_dataset import (
    MigrationFeatureNormalizer,
    MigrationTransitionDataset,
)
from core.marl.migration_reward import migration_action_reward


def snapshot_from_record(record: Mapping[str, Any]) -> ResourceSnapshot:
    raw = record["snapshot"]

    def edges(values: Mapping[str, Any]) -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for key, value in values.items():
            if isinstance(key, str):
                u, v = key.split(",", 1)
                result[(int(u), int(v))] = float(value)
            else:
                result[(int(key[0]), int(key[1]))] = float(value)
        return result

    return ResourceSnapshot(
        version=int(raw.get("version", 0)),
        cpu_remaining={int(key): float(value) for key, value in raw["cpu_remaining"].items()},
        memory_remaining={int(key): float(value) for key, value in raw["memory_remaining"].items()},
        bandwidth_remaining=edges(raw["bandwidth_remaining"]),
    )


@dataclass(frozen=True)
class MigrationStepResult:
    observation: dict[str, torch.Tensor]
    reward: float
    done: bool
    info: dict[str, Any]


class MigrationReplayEnv:
    """Static trace replay for policy evaluation and counterfactual decoding.

    This class intentionally replays serialized migration records.  Actions do
    not mutate a resource ledger or future records, so it must not be used as
    a dynamic training environment.  Use :class:`DynamicMigrationEnv` for
    action-dependent rollouts.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        normalizer: MigrationFeatureNormalizer,
        max_agents: int = 8,
        max_candidates: int = 8,
        decoder_top_r: int = 4,
        decoder_time_budget_ms: float = 2.0,
    ) -> None:
        self.records = list(records)
        self.dataset = MigrationTransitionDataset(
            self.records, normalizer, max_agents, max_candidates
        )
        self.decoder_top_r = int(decoder_top_r)
        self.decoder_time_budget_ms = float(decoder_time_budget_ms)
        self.index = 0

    def reset(self) -> dict[str, torch.Tensor]:
        self.index = 0
        return self._observation()

    def _observation(self) -> dict[str, torch.Tensor]:
        if self.index >= len(self.dataset):
            return self.dataset._tensorize(None)
        item = self.dataset._tensorize(self.records[self.index])
        return {
            key: value.unsqueeze(0)
            for key, value in item.items()
            if key != "actions"
        }

    def decode_rankings(
        self,
        rankings: Sequence[Sequence[int]],
        *,
        scores: Optional[Sequence[Sequence[float]]] = None,
    ) -> JointDecodeResult:
        if self.index >= len(self.records):
            raise RuntimeError("migration environment is terminated")
        record = self.records[self.index]
        agents = list(record["agents"])
        if len(rankings) != len(agents):
            raise ValueError("one ranking is required per migration agent")
        # In deployment, reject is only a feasibility fallback.  In migration,
        # however, noop is a genuine policy action.  Treat its rank as a cutoff:
        # candidates below noop must not be resurrected by the acceptance-first
        # joint decoder.
        bounded_rankings: list[list[int]] = []
        for agent, ranking in zip(agents, rankings):
            reject = int(agent["reject_action"])
            bounded: list[int] = []
            for raw_action in ranking:
                action = int(raw_action)
                bounded.append(action)
                if action == reject:
                    break
            if reject not in bounded:
                bounded.append(reject)
            bounded_rankings.append(bounded)
        footprints = [
            [
                None
                if candidate.get("resource_footprint") is None
                else deserialize_footprint(candidate["resource_footprint"])
                for candidate in agent["candidates"]
            ]
            for agent in agents
        ]
        return decode_joint_candidates(
            footprints,
            snapshot_from_record(record),
            bounded_rankings,
            reject_actions=[int(agent["reject_action"]) for agent in agents],
            action_mask=[list(map(bool, agent["action_mask"])) for agent in agents],
            scores=scores,
            priorities=[
                (-float(agent["task"]["priority"]), agent["task"]["task_id"])
                for agent in agents
            ],
            top_r=self.decoder_top_r,
            time_budget_ms=self.decoder_time_budget_ms,
        )

    @staticmethod
    def action_reward(agent: Mapping[str, Any], action: int) -> float:
        return migration_action_reward(agent, action)

    def step_rankings(
        self,
        rankings: Sequence[Sequence[int]],
        *,
        scores: Optional[Sequence[Sequence[float]]] = None,
    ) -> MigrationStepResult:
        decode = self.decode_rankings(rankings, scores=scores)
        record = self.records[self.index]
        reward = sum(
            self.action_reward(agent, action)
            for agent, action in zip(record["agents"], decode.actions)
        )
        self.index += 1
        done = self.index >= len(self.records) or (
            self.index > 0
            and self.index < len(self.records)
            and str(self.records[self.index].get("trace_id"))
            != str(record.get("trace_id"))
        )
        return MigrationStepResult(
            observation=self._observation(),
            reward=float(reward),
            done=bool(done),
            info={"decoder": decode.to_dict(), "actions": list(decode.actions)},
        )


class MigrationWQMIXEnv(MigrationReplayEnv):
    """Deprecated compatibility alias for the old static replay environment.

    New code should import :class:`MigrationReplayEnv` explicitly.  Keeping
    this alias avoids breaking historical fixtures while making accidental use
    visible during development and CI.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "MigrationWQMIXEnv is a deprecated static replay environment; "
            "use MigrationReplayEnv for offline records or "
            "DynamicMigrationEnv for action-dependent training.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
