#!/usr/bin/env python3
"""Stable checkpoint-loading API used by the HRL export and online planner.

This module intentionally contains no dataset generation or long evaluation
CLI.  It builds the current project stack, loads frozen weights, and returns
the environment, agent, and coordinator required by runtime callers.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.hrl.agent import create_goal_conditioned_agent  # noqa: E402
from envs.modules.HRL_Coordinator import HRL_Coordinator  # noqa: E402
from envs.sfc_env import SFC_HIRL_Env  # noqa: E402
from train_tahrl import (  # noqa: E402
    inject_dynamic_dimensions,
    load_topology,
    setup_hrl_config,
    validate_config,
)
from utils.config_utils import load_config  # noqa: E402


logger = logging.getLogger("evaluate_checkpoint")

__all__ = ["set_seed", "load_eval_checkpoint", "build_eval_stack"]


def set_seed(seed: int) -> None:
    """Set deterministic evaluation seeds without changing model weights."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def _load_payload(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    # NumPy 2 serializes scalar reconstruction helpers under ``numpy._core``.
    # The Mininet/Ryu environment intentionally remains on NumPy 1.x, where
    # the same modules are public as ``numpy.core``. Register import aliases
    # for pickle without changing arrays, dtypes, or checkpoint values.
    if int(np.__version__.split('.', 1)[0]) < 2:
        numpy_core = importlib.import_module('numpy.core')
        sys.modules.setdefault('numpy._core', numpy_core)
        for module_name in (
            'multiarray', 'numeric', 'umath', '_multiarray_umath',
        ):
            module = importlib.import_module(f'numpy.core.{module_name}')
            sys.modules.setdefault(f'numpy._core.{module_name}', module)
    try:
        payload = torch.load(
            str(path), map_location=device, weights_only=False
        )
    except TypeError:
        payload = torch.load(str(path), map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    return payload


def _load_module(module: Any, state: Any, name: str) -> None:
    if module is None:
        raise RuntimeError(f"current agent has no module named {name}")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is missing state for {name}")
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"checkpoint {name} is incompatible: {exc}") from exc


def load_eval_checkpoint(agent: Any, checkpoint_path: str | Path) -> dict[str, Any]:
    """Strictly load a trainer-style or raw HRL agent checkpoint."""
    path = Path(checkpoint_path).expanduser().resolve()
    payload = _load_payload(path, agent.device)
    state = payload.get("agent_state", payload)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint agent_state must be a mapping: {path}")

    _load_module(agent.high_policy, state.get("high_policy"), "high_policy")
    _load_module(agent.low_policy, state.get("low_policy"), "low_policy")
    _load_module(agent.encoder, state.get("encoder"), "encoder")

    target_high = state.get("target_high_policy")
    if target_high is None:
        agent.target_high_policy.load_state_dict(agent.high_policy.state_dict())
    else:
        _load_module(
            agent.target_high_policy, target_high, "target_high_policy"
        )

    target_low = state.get("target_low_policy")
    if target_low is None:
        agent.target_low_policy.load_state_dict(agent.low_policy.state_dict())
    else:
        _load_module(agent.target_low_policy, target_low, "target_low_policy")

    target_encoder = getattr(agent, "target_encoder", None)
    if target_encoder is not None:
        target_state = state.get("target_encoder")
        if target_state is None:
            target_encoder.load_state_dict(agent.encoder.state_dict())
        else:
            _load_module(target_encoder, target_state, "target_encoder")

    agent.steps_done = int(state.get("steps_done", agent.steps_done))
    logger.info("Loaded frozen HRL checkpoint: %s", path)
    return payload


def _required_arg(args: Any, name: str) -> Any:
    if not hasattr(args, name):
        raise ValueError(f"evaluation argument is missing: {name}")
    value = getattr(args, name)
    if value is None and name not in {"config", "bw_cap", "cap_cpu", "cap_mem"}:
        raise ValueError(f"evaluation argument cannot be None: {name}")
    return value


def build_eval_stack(args: Any, seed: int, data_path: str | Path):
    """Build the current HRL environment and load one frozen checkpoint."""
    required = (
        "config",
        "topo",
        "goal_strategy",
        "epsilon",
        "bw_cap",
        "cap_cpu",
        "cap_mem",
        "ablation_variant",
        "ablation_hop",
        "ablation_reach",
        "zero_candidate_feats",
        "minimal_mlp_state",
        "checkpoint",
        "model_mode",
        "start_episode",
    )
    values = {name: _required_arg(args, name) for name in required}

    checkpoint_path = Path(values["checkpoint"]).expanduser().resolve()
    dataset_path = Path(data_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if values["model_mode"] not in {"eval", "train"}:
        raise ValueError("model_mode must be 'eval' or 'train'")
    epsilon = float(values["epsilon"])
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    for name in ("bw_cap", "cap_cpu", "cap_mem"):
        value = values[name]
        if value is not None and float(value) <= 0.0:
            raise ValueError(f"{name} must be positive")

    set_seed(seed)
    config = load_config(values["config"] or "phase3")
    setup_hrl_config(config)
    config["topo"] = str(values["topo"])
    config.setdefault("hrl", {})["goal_strategy"] = str(
        values["goal_strategy"]
    )
    epsilon_config = config.setdefault("training", {}).setdefault("epsilon", {})
    for name in ("initial_high", "initial_low", "final_high", "final_low"):
        epsilon_config[name] = epsilon

    topology_path = PROJECT_ROOT / "configs" / "topology.yaml"
    with topology_path.open(encoding="utf-8") as handle:
        topology_registry = yaml.safe_load(handle)
    topologies = (topology_registry or {}).get("topologies", {})
    if values["topo"] not in topologies:
        raise ValueError(f"unknown topology: {values['topo']}")
    topology = topologies[values["topo"]]
    matrix_path = (PROJECT_ROOT / topology["file"]).resolve()
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)
    config.setdefault("topology", {})["file"] = str(matrix_path)
    config["topology"]["dc_nodes"] = list(topology["dc_nodes"])
    config.setdefault("environment", {})["dc_nodes"] = list(
        topology["dc_nodes"]
    )

    if values["bw_cap"] is not None:
        config.setdefault("env", {})["link_capacity"] = float(values["bw_cap"])
        config.setdefault("capacities", {})["bandwidth"] = float(
            values["bw_cap"]
        )
    if values["cap_cpu"] is not None:
        config.setdefault("capacities", {})["cpu"] = float(values["cap_cpu"])
    if values["cap_mem"] is not None:
        config.setdefault("capacities", {})["memory"] = float(values["cap_mem"])
    variant = str(values["ablation_variant"])
    config["ablation_variant"] = variant

    validate_config(config, "phase3")
    if not load_topology(config):
        raise RuntimeError(f"failed to load topology matrix: {matrix_path}")
    node_count = int(config["topology"]["matrix"].shape[0])
    config.setdefault("environment", {})["num_nodes"] = node_count
    config["environment"]["nb_high_level_goals"] = node_count
    config["environment"]["nb_low_level_actions"] = node_count

    env = SFC_HIRL_Env(config, use_gnn=True)
    env._ablation_variant = variant
    env.ablation_variant = variant
    inject_dynamic_dimensions(config, env)
    minimal_mlp = bool(values["minimal_mlp_state"] or variant == "mlp")
    env._minimal_mlp_state = minimal_mlp
    env._ablation_hop = bool(values["ablation_hop"] or minimal_mlp)
    env._ablation_reach = bool(values["ablation_reach"] or minimal_mlp)
    env._ablation_candidate_feats = bool(
        values["zero_candidate_feats"] or minimal_mlp
    )

    agent = create_goal_conditioned_agent(
        config=config,
        phase=3,
        goal_strategy=str(values["goal_strategy"]),
        env=env,
        ablation_variant=variant,
    )
    payload = load_eval_checkpoint(agent, checkpoint_path)
    checkpoint_variant = payload.get("ablation_variant")
    if checkpoint_variant is not None and str(checkpoint_variant) != variant:
        raise ValueError(
            f"checkpoint variant {checkpoint_variant!r} does not match {variant!r}"
        )
    checkpoint_encoder = payload.get("encoder_class")
    current_encoder = agent.encoder.__class__.__name__
    if checkpoint_encoder is not None and str(checkpoint_encoder) != current_encoder:
        raise ValueError(
            f"checkpoint encoder {checkpoint_encoder!r} does not match "
            f"{current_encoder!r}"
        )

    if values["model_mode"] == "train":
        agent.train()
    else:
        agent.eval()
    agent.epsilon_high = epsilon
    agent.epsilon_low = epsilon

    coordinator = HRL_Coordinator(
        env=env, high_agent=agent, low_agent=agent, config=config
    )
    coordinator.current_episode = int(values["start_episode"])
    if not env.load_dataset(str(dataset_path)):
        raise RuntimeError(f"failed to load dataset: {dataset_path}")
    return env, agent, coordinator
