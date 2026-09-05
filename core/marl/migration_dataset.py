"""Dataset and normalization utilities for migration-specific WQMIX."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from core.marl.migration_candidates import (
    MIGRATION_CANDIDATE_DIM,
    MIGRATION_REQUEST_DIM,
    MIGRATION_STATE_DIM,
)
from core.marl.migration_reward import migration_action_rewards


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_migration_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        source = path / "batches.jsonl" if path.is_dir() else path
        for record in read_jsonl(source):
            record["_source"] = str(source.resolve())
            records.append(record)
    records.sort(key=lambda row: (
        str(row.get("trace_id", row.get("_source", ""))),
        int(row.get("batch_id", 0)),
    ))
    return records


@dataclass(frozen=True)
class MigrationFeatureNormalizer:
    request_mean: list[float]
    request_std: list[float]
    candidate_mean: list[float]
    candidate_std: list[float]
    state_mean: list[float]
    state_std: list[float]

    @staticmethod
    def _stats(rows: Sequence[Sequence[float]], dimension: int) -> tuple[list[float], list[float]]:
        if not rows:
            return [0.0] * dimension, [1.0] * dimension
        values = np.asarray(rows, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != dimension:
            raise ValueError(f"expected feature dimension {dimension}, got {values.shape}")
        deviation = values.std(axis=0)
        deviation[deviation < 1e-6] = 1.0
        return values.mean(axis=0).tolist(), deviation.tolist()

    @classmethod
    def fit(cls, records: Sequence[Mapping[str, Any]]) -> "MigrationFeatureNormalizer":
        requests: list[list[float]] = []
        candidates: list[list[float]] = []
        states: list[list[float]] = []
        for record in records:
            states.append(list(map(float, record["state_features"])))
            for agent in record["agents"]:
                requests.append(list(map(float, agent["request_features"])))
                candidates.extend(
                    list(map(float, values))
                    for values in agent["candidate_features"]
                )
        request_mean, request_std = cls._stats(requests, MIGRATION_REQUEST_DIM)
        candidate_mean, candidate_std = cls._stats(candidates, MIGRATION_CANDIDATE_DIM)
        state_mean, state_std = cls._stats(states, MIGRATION_STATE_DIM)
        return cls(
            request_mean, request_std,
            candidate_mean, candidate_std,
            state_mean, state_std,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MigrationFeatureNormalizer":
        return cls(**{key: list(map(float, value[key])) for key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def _normalize(values: Sequence[float], mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
        return (
            np.asarray(values, dtype=np.float32)
            - np.asarray(mean, dtype=np.float32)
        ) / np.asarray(std, dtype=np.float32)

    def normalize_request(self, values: Sequence[float]) -> np.ndarray:
        return self._normalize(values, self.request_mean, self.request_std)

    def normalize_candidate(self, values: Sequence[float]) -> np.ndarray:
        return self._normalize(values, self.candidate_mean, self.candidate_std)

    def normalize_state(self, values: Sequence[float]) -> np.ndarray:
        return self._normalize(values, self.state_mean, self.state_std)


class MigrationTransitionDataset(Dataset):
    """Padded consecutive batch transitions with oracle teacher actions."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        normalizer: MigrationFeatureNormalizer,
        max_agents: int = 8,
        max_candidates: int = 8,
    ) -> None:
        self.records = list(records)
        self.normalizer = normalizer
        self.max_agents = int(max_agents)
        self.max_candidates = int(max_candidates)
        if self.max_agents <= 0 or self.max_candidates <= 0:
            raise ValueError("max_agents and max_candidates must be positive")
        self.next_index: list[int | None] = []
        for index, record in enumerate(self.records):
            next_value: int | None = None
            if index + 1 < len(self.records):
                next_record = self.records[index + 1]
                if str(next_record.get("trace_id")) == str(record.get("trace_id")):
                    next_value = index + 1
            self.next_index.append(next_value)

    def __len__(self) -> int:
        return len(self.records)

    def _tensorize(self, record: Mapping[str, Any] | None) -> dict[str, torch.Tensor]:
        request = np.zeros((self.max_agents, MIGRATION_REQUEST_DIM), dtype=np.float32)
        candidate = np.zeros(
            (self.max_agents, self.max_candidates + 1, MIGRATION_CANDIDATE_DIM),
            dtype=np.float32,
        )
        action_mask = np.zeros((self.max_agents, self.max_candidates + 1), dtype=np.bool_)
        action_rewards = np.zeros(
            (self.max_agents, self.max_candidates + 1), dtype=np.float32
        )
        agent_mask = np.zeros(self.max_agents, dtype=np.bool_)
        actions = np.zeros(self.max_agents, dtype=np.int64)
        state = np.zeros(MIGRATION_STATE_DIM, dtype=np.float32)
        if record is not None:
            agents = list(record["agents"])
            if len(agents) > self.max_agents:
                raise ValueError("migration record exceeds max_agents")
            state[:] = self.normalizer.normalize_state(record["state_features"])
            for agent_index, agent in enumerate(agents):
                features = list(agent["candidate_features"])
                masks = list(agent["action_mask"])
                rewards = migration_action_rewards(agent)
                if len(features) > self.max_candidates + 1:
                    raise ValueError("migration record exceeds max_candidates + reject")
                if len(features) != len(masks):
                    raise ValueError("candidate features and action mask must align")
                if len(features) != len(rewards):
                    raise ValueError("candidate features and action rewards must align")
                agent_mask[agent_index] = True
                request[agent_index] = self.normalizer.normalize_request(
                    agent["request_features"]
                )
                for candidate_index, values in enumerate(features):
                    candidate[agent_index, candidate_index] = (
                        self.normalizer.normalize_candidate(values)
                    )
                    action_mask[agent_index, candidate_index] = bool(masks[candidate_index])
                    action_rewards[agent_index, candidate_index] = float(
                        rewards[candidate_index]
                    )
                action = int(agent["oracle_action"])
                if not 0 <= action < len(features) or not action_mask[agent_index, action]:
                    raise ValueError("oracle migration action is invalid or masked")
                actions[agent_index] = action
        return {
            "request_observations": torch.from_numpy(request),
            "candidate_features": torch.from_numpy(candidate),
            "action_mask": torch.from_numpy(action_mask),
            "action_rewards": torch.from_numpy(action_rewards),
            "agent_mask": torch.from_numpy(agent_mask),
            "states": torch.from_numpy(state),
            "actions": torch.from_numpy(actions),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        current = self._tensorize(record)
        next_index = self.next_index[index]
        following = self._tensorize(
            self.records[next_index] if next_index is not None else None
        )
        return {
            **current,
            "record_index": torch.tensor(index, dtype=torch.int64),
            "teacher_actions": current["actions"].clone(),
            "rewards": torch.tensor(float(record.get("reward", 0.0)), dtype=torch.float32),
            "next_request_observations": following["request_observations"],
            "next_candidate_features": following["candidate_features"],
            "next_action_mask": following["action_mask"],
            "next_agent_mask": following["agent_mask"],
            "next_states": following["states"],
            "dones": torch.tensor(float(next_index is None), dtype=torch.float32),
        }
