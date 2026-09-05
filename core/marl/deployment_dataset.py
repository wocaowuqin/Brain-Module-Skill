"""Padded tensor loading for deployment_topk_v3 and Oracle labels."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


CANDIDATE_CONTINUOUS_INDICES = tuple(range(8)) + tuple(range(12, 15)) + tuple(range(16, 23))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass
class FeatureNormalizer:
    request_mean: List[float]
    request_std: List[float]
    candidate_mean: List[float]
    candidate_std: List[float]
    state_mean: List[float]
    state_std: List[float]

    @classmethod
    def fit(cls, batches: Sequence[Mapping[str, Any]]) -> "FeatureNormalizer":
        request_rows = []
        candidate_rows = []
        state_rows = []
        for batch in batches:
            state_rows.append(list(map(float, batch["global_state_features"])))
            for agent in batch["agents"]:
                request_rows.append(list(map(float, agent["request_features"])))
                for candidate in agent["candidates"]:
                    candidate_rows.append(list(map(float, candidate["candidate_features"])))
        if not request_rows or not candidate_rows or not state_rows:
            raise ValueError("cannot fit feature normalization on an empty dataset")

        def stats(rows):
            array = np.asarray(rows, dtype=np.float64)
            return array.mean(axis=0), np.maximum(array.std(axis=0), 1e-6)

        request_mean, request_std = stats(request_rows)
        candidate_mean, candidate_std = stats(candidate_rows)
        state_mean, state_std = stats(state_rows)
        candidate_mean_masked = np.zeros_like(candidate_mean)
        candidate_std_masked = np.ones_like(candidate_std)
        for index in CANDIDATE_CONTINUOUS_INDICES:
            if index < len(candidate_mean):
                candidate_mean_masked[index] = candidate_mean[index]
                candidate_std_masked[index] = candidate_std[index]
        return cls(
            request_mean.tolist(),
            request_std.tolist(),
            candidate_mean_masked.tolist(),
            candidate_std_masked.tolist(),
            state_mean.tolist(),
            state_std.tolist(),
        )

    def normalize_request(self, values: Sequence[float]) -> np.ndarray:
        return (
            np.asarray(values, dtype=np.float32) - np.asarray(self.request_mean, dtype=np.float32)
        ) / np.asarray(self.request_std, dtype=np.float32)

    def normalize_candidate(self, values: Sequence[float]) -> np.ndarray:
        return (
            np.asarray(values, dtype=np.float32)
            - np.asarray(self.candidate_mean, dtype=np.float32)
        ) / np.asarray(self.candidate_std, dtype=np.float32)

    def normalize_state(self, values: Sequence[float]) -> np.ndarray:
        return (
            np.asarray(values, dtype=np.float32) - np.asarray(self.state_mean, dtype=np.float32)
        ) / np.asarray(self.state_std, dtype=np.float32)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "request_mean": self.request_mean,
            "request_std": self.request_std,
            "candidate_mean": self.candidate_mean,
            "candidate_std": self.candidate_std,
            "state_mean": self.state_mean,
            "state_std": self.state_std,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Sequence[float]]) -> "FeatureNormalizer":
        return cls(**{key: list(map(float, raw[key])) for key in cls.__annotations__})


def load_labeled_batches(folders: Iterable[str | Path]) -> List[Dict[str, Any]]:
    combined = []
    for raw_folder in folders:
        folder = Path(raw_folder)
        batches_path = folder / "batches.jsonl" if folder.is_dir() else folder
        labels_path = batches_path.parent / "oracle_labels.jsonl"
        batches = read_jsonl(batches_path)
        labels = read_jsonl(labels_path)
        if len(batches) != len(labels):
            raise ValueError(f"batch/label count mismatch in {batches_path.parent}")
        source_id = str(batches_path.parent.resolve())
        for batch, label in zip(batches, labels):
            if (
                int(batch["batch_id"]) != int(label["batch_id"])
                or int(batch["snapshot_version"]) != int(label["snapshot_version"])
            ):
                raise ValueError(f"batch/label identity mismatch in {batches_path.parent}")
            combined.append({"batch": batch, "label": label, "source_id": source_id})
    return combined


def source_request_files(folders: Iterable[str | Path]) -> set[str]:
    results = set()
    for raw_folder in folders:
        folder = Path(raw_folder)
        spec_path = folder / "dataset_spec.json"
        if spec_path.exists():
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if spec.get("requests"):
                results.add(str(Path(spec["requests"]).resolve()))
    return results


class DeploymentOracleDataset(Dataset):
    """Fixed-shape request-agent batches with Oracle action targets."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        normalizer: FeatureNormalizer,
        max_agents: int = 32,
        max_candidates: int = 9,
    ) -> None:
        self.records = list(records)
        self.normalizer = normalizer
        self.max_agents = int(max_agents)
        self.max_candidates = int(max_candidates)
        if not self.records:
            raise ValueError("deployment dataset is empty")
        first_agent = self.records[0]["batch"]["agents"][0]
        self.request_dim = len(first_agent["request_features"])
        self.candidate_dim = len(first_agent["candidates"][0]["candidate_features"])
        self.state_dim = len(self.records[0]["batch"]["global_state_features"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        batch = record["batch"]
        label = record["label"]
        agents = batch["agents"]
        if len(agents) > self.max_agents:
            raise ValueError(f"batch has {len(agents)} agents; max is {self.max_agents}")
        request_obs = np.zeros((self.max_agents, self.request_dim), dtype=np.float32)
        candidate_features = np.zeros(
            (self.max_agents, self.max_candidates, self.candidate_dim), dtype=np.float32
        )
        action_mask = np.zeros((self.max_agents, self.max_candidates), dtype=np.bool_)
        agent_mask = np.zeros(self.max_agents, dtype=np.bool_)
        actions = np.zeros(self.max_agents, dtype=np.int64)
        original_actions = np.zeros(self.max_agents, dtype=np.int64)
        reject_actions = np.zeros(self.max_agents, dtype=np.int64)
        request_ids = np.full(self.max_agents, -1, dtype=np.int64)
        for agent_index, agent in enumerate(agents):
            candidates = agent["candidates"]
            if len(candidates) > self.max_candidates:
                raise ValueError(
                    f"request {agent['request_id']} has {len(candidates)} actions; "
                    f"max is {self.max_candidates}"
                )
            agent_mask[agent_index] = True
            request_ids[agent_index] = int(agent["request_id"])
            request_obs[agent_index] = self.normalizer.normalize_request(
                agent["request_features"]
            )
            for candidate_index, candidate in enumerate(candidates):
                values = list(map(float, candidate["candidate_features"]))
                if values:
                    # v3.0 stored ``source == legacy_hrl`` in the final
                    # feature.  Keep the dimension stable while preventing
                    # generator identity leakage into BC/WQMIX.
                    values[-1] = 0.0
                candidate_features[agent_index, candidate_index] = (
                    self.normalizer.normalize_candidate(values)
                )
                action_mask[agent_index, candidate_index] = bool(candidate["action_valid"])
            actions[agent_index] = int(label["oracle_actions"][agent_index])
            original_actions[agent_index] = int(label["original_actions"][agent_index])
            reject_actions[agent_index] = int(agent["reject_action"])
        return {
            "request_observations": torch.from_numpy(request_obs),
            "candidate_features": torch.from_numpy(candidate_features),
            "action_mask": torch.from_numpy(action_mask),
            "agent_mask": torch.from_numpy(agent_mask),
            "states": torch.from_numpy(
                self.normalizer.normalize_state(batch["global_state_features"])
            ),
            "actions": torch.from_numpy(actions),
            "original_actions": torch.from_numpy(original_actions),
            "reject_actions": torch.from_numpy(reject_actions),
            "request_ids": torch.from_numpy(request_ids),
            "oracle_accepted": torch.tensor(float(label["oracle_accepted"])),
            "training_reward": torch.tensor(float(label["training_reward"])),
            "batch_id": torch.tensor(int(batch["batch_id"])),
        }
