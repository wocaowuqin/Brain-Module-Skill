#!/usr/bin/env python3
"""Run real-data evaluation for role-based SFT reconfiguration."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hrl.agent import create_goal_conditioned_agent
from envs.modules.HRL_Coordinator import HRL_Coordinator
from envs.sfc_env import SFC_HIRL_Env
from envs.sft_role_marl_env import SFTRoleMARLEnv
from trainer.role_reconfig_eval import RoleReconfigEvaluator
from train_tahrl import (
    inject_dynamic_dimensions,
    load_topology,
    setup_hrl_config,
    validate_config,
)
from utils.config_utils import load_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate role-based SFT reconfiguration on real request data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="phase3", help="config key or yaml path")
    parser.add_argument("--data", default="data/us_backbone_rate8/phase3_requests.pkl")
    parser.add_argument(
        "--topo",
        default="us_backbone",
        choices=["us_backbone", "13node", "23node", "50node", "cogentco"],
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--reconfig-interval", type=int, default=10)
    parser.add_argument("--warmup-episodes", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true", help="plan role actions without applying them")
    parser.add_argument(
        "--output", default="artifacts/runs/reconfiguration/role_eval"
    )
    parser.add_argument("--gpu", type=int, default=-1, help="-1 forces CPU")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--goal-strategy", default="adaptive", choices=["adaptive", "relative", "hybrid"])
    parser.add_argument(
        "--ablation-variant",
        default="single_dqn",
        choices=["full", "no_tree", "no_dest_mask", "single_stream", "single_dqn", "mlp", "gat"],
    )
    parser.add_argument("--checkpoint", default=None, help="optional HRL/IL checkpoint path")
    parser.add_argument(
        "--deployment-executor",
        default="policy",
        choices=["policy", "bw_planner"],
        help="low-level executor used for online request deployment",
    )
    parser.add_argument("--bw-cap", type=float, default=None)
    parser.add_argument("--cap-cpu", type=float, default=None)
    parser.add_argument("--cap-mem", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--node-util-threshold", type=float, default=0.80)
    parser.add_argument("--link-util-threshold", type=float, default=0.80)
    parser.add_argument("--delay-threshold", type=float, default=None)
    parser.add_argument("--min-risk-score", type=float, default=0.0)
    parser.add_argument("--min-action-gain", type=float, default=0.0)
    parser.add_argument(
        "--max-tree-edge-growth",
        type=int,
        default=0,
        help="maximum extra physical tree edges per online reroute; zero protects admission bandwidth",
    )
    parser.add_argument(
        "--max-reroute-link-utilization",
        type=float,
        default=0.50,
        help="reject reroutes whose newly allocated edge would exceed this utilization",
    )
    parser.add_argument(
        "--min-connectivity-ratio",
        type=float,
        default=1.0,
        help="minimum post/pre all-pairs residual widest-path score for an online reroute",
    )
    parser.add_argument("--role-agent-mode", default="rule", choices=["rule", "trainable"])
    parser.add_argument("--role-hidden-dim", type=int, default=64)
    parser.add_argument("--role-lr", type=float, default=1e-3)
    parser.add_argument(
        "--role-epsilon",
        type=float,
        default=0.0,
        help="role-policy exploration; evaluation defaults to deterministic greedy actions",
    )
    parser.add_argument("--role-checkpoint", default=None, help="trained role DQN/QMIX checkpoint")
    parser.add_argument("--disable-migration", action="store_true")
    parser.add_argument("--disable-reroute", action="store_true")
    parser.add_argument(
        "--migration-benefit-predictor",
        default=None,
        help="validated counterfactual migration-benefit checkpoint",
    )
    parser.add_argument(
        "--migration-benefit-threshold",
        type=float,
        default=None,
        help="optional probability threshold override",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_topology(config: dict, topo_name: str) -> None:
    topo_yaml = ROOT / "configs" / "topology.yaml"
    with topo_yaml.open("r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)
    topologies = registry.get("topologies", {})
    if topo_name not in topologies:
        raise ValueError(f"unknown topology {topo_name}; available={list(topologies)}")
    topo_info = topologies[topo_name]

    config["topo"] = topo_name
    config.setdefault("topology", {})["file"] = str(ROOT / topo_info["file"])
    config.setdefault("topology", {})["dc_nodes"] = topo_info["dc_nodes"]
    config.setdefault("environment", {})["dc_nodes"] = topo_info["dc_nodes"]

    if not load_topology(config):
        raise RuntimeError(f"failed to load topology: {topo_name}")

    num_nodes = int(config["topology"]["matrix"].shape[0])
    config.setdefault("environment", {})["num_nodes"] = num_nodes
    config["environment"]["nb_high_level_goals"] = num_nodes
    config["environment"]["nb_low_level_actions"] = num_nodes


def apply_overrides(config: dict, args: argparse.Namespace) -> None:
    use_cuda = bool(torch.cuda.is_available() and args.gpu >= 0)
    config["use_cuda"] = use_cuda
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    config.setdefault("training", {})
    config["training"].setdefault("lr_high", 1e-5)
    config["training"].setdefault("lr_low", 1e-5)
    config["training"].setdefault("batch_size", 64)
    config["training"].setdefault("gamma", 0.95)
    config["training"].setdefault("target_update_freq", 500)
    config["training"].setdefault("buffer_size", 100000)
    if not isinstance(config["training"].get("epsilon"), dict):
        config["training"]["epsilon"] = {
            "initial_high": 0.05,
            "final_high": 0.05,
            "initial_low": 0.05,
            "final_low": 0.05,
            "decay_episodes": 999999,
        }

    config["ablation_variant"] = args.ablation_variant
    config["deployment_executor"] = args.deployment_executor
    config.setdefault("hrl", {})["goal_strategy"] = args.goal_strategy
    config.setdefault("phase3", {})["max_steps"] = args.max_steps

    if args.bw_cap is not None:
        config.setdefault("env", {})["link_capacity"] = float(args.bw_cap)
        config.setdefault("capacities", {})["bandwidth"] = float(args.bw_cap)
    if args.cap_cpu is not None:
        config.setdefault("capacities", {})["cpu"] = float(args.cap_cpu)
    if args.cap_mem is not None:
        config.setdefault("capacities", {})["memory"] = float(args.cap_mem)


def build_runtime(args: argparse.Namespace):
    set_seed(args.seed)
    config = load_config(args.config)
    setup_hrl_config(config)
    configure_topology(config, args.topo)
    apply_overrides(config, args)
    validate_config(config, "phase3")

    env = SFC_HIRL_Env(config, use_gnn=True)
    inject_dynamic_dimensions(config, env)

    env._ablation_variant = args.ablation_variant
    env.ablation_variant = args.ablation_variant
    minimal_mlp_state = bool(args.ablation_variant == "mlp")
    env._minimal_mlp_state = minimal_mlp_state
    env._ablation_hop = minimal_mlp_state
    env._ablation_reach = minimal_mlp_state
    env._ablation_candidate_feats = minimal_mlp_state

    agent = create_goal_conditioned_agent(
        config=config,
        phase=3,
        goal_strategy=args.goal_strategy,
        env=env,
        ablation_variant=args.ablation_variant,
    )
    if args.ablation_variant == "single_dqn":
        for param in agent.high_policy.parameters():
            param.requires_grad = False

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            agent.load(str(ckpt_path))
        else:
            logger.warning("checkpoint not found, continue with initialized weights: %s", ckpt_path)
    agent.eval()
    # Role-policy experiments compare reconfiguration methods, so the shared
    # underlying deployment policy must not introduce method-dependent noise.
    agent.epsilon_high = 0.0
    agent.epsilon_low = 0.0

    coordinator = HRL_Coordinator(env=env, high_agent=agent, low_agent=agent, config=config)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"request data not found: {data_path}")
    if not env.load_dataset(str(data_path)):
        raise RuntimeError(f"env.load_dataset failed: {data_path}")

    migration_benefit_predictor = None
    predictor_path = getattr(args, "migration_benefit_predictor", None)
    if predictor_path:
        from core.marl.migration_benefit_predictor import MigrationBenefitPredictor

        migration_benefit_predictor = MigrationBenefitPredictor.from_path(
            predictor_path,
            threshold=getattr(args, "migration_benefit_threshold", None),
        )

    role_env = SFTRoleMARLEnv(
        env,
        top_k=args.top_k,
        node_util_threshold=args.node_util_threshold,
        link_util_threshold=args.link_util_threshold,
        delay_threshold=args.delay_threshold,
        min_risk_score=args.min_risk_score,
        min_action_gain=args.min_action_gain,
        agent_mode=args.role_agent_mode,
        trainable_config={
            "hidden_dim": args.role_hidden_dim,
            "lr": args.role_lr,
            "epsilon": args.role_epsilon,
            "device": "cuda" if torch.cuda.is_available() and args.gpu >= 0 else "cpu",
        },
        enable_migration=not getattr(args, "disable_migration", False),
        enable_reroute=not getattr(args, "disable_reroute", False),
        max_tree_edge_growth=args.max_tree_edge_growth,
        max_projected_link_utilization=args.max_reroute_link_utilization,
        min_residual_connectivity_ratio=args.min_connectivity_ratio,
        migration_benefit_predictor=migration_benefit_predictor,
    )
    if args.role_checkpoint:
        role_state = torch.load(str(Path(args.role_checkpoint)), map_location=role_env.selector.device, weights_only=False)
        role_env.selector.load_state_dict(role_state["selector"])
        role_env.migrator.load_state_dict(role_state["migration"])
        role_env.rerouter.load_state_dict(role_state["reroute"])
    return config, env, agent, coordinator, role_env


def main() -> int:
    os.chdir(ROOT)
    args = parse_args()
    _, env, _, coordinator, role_env = build_runtime(args)

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    evaluator = RoleReconfigEvaluator(
        env=env,
        coordinator=coordinator,
        role_env=role_env,
        output_dir=output_dir,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        reconfig_interval=args.reconfig_interval,
        warmup_episodes=args.warmup_episodes,
        apply_reconfig=not args.plan_only,
    )
    summary = evaluator.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
