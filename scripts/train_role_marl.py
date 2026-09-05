#!/usr/bin/env python3
"""Train role-based DQN agents for SFT reconfiguration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_role_reconfig_eval import build_runtime
from trainer.role_marl_trainer import RoleMARLTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train role-DQN agents for low-disturbance SFT reconfiguration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="phase3")
    parser.add_argument("--data", default="data/us_backbone_rate8/phase3_requests.pkl")
    parser.add_argument(
        "--topo",
        default="us_backbone",
        choices=["us_backbone", "13node", "23node", "50node", "cogentco"],
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--reconfig-interval", type=int, default=10)
    parser.add_argument("--warmup-episodes", type=int, default=0)
    parser.add_argument(
        "--output", default="artifacts/runs/reconfiguration/role_marl_train"
    )
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--goal-strategy", default="adaptive", choices=["adaptive", "relative", "hybrid"])
    parser.add_argument(
        "--ablation-variant",
        default="single_dqn",
        choices=["full", "no_tree", "no_dest_mask", "single_stream", "single_dqn", "mlp", "gat"],
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--bw-cap", type=float, default=None)
    parser.add_argument("--cap-cpu", type=float, default=None)
    parser.add_argument("--cap-mem", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--node-util-threshold", type=float, default=0.80)
    parser.add_argument("--link-util-threshold", type=float, default=0.80)
    parser.add_argument("--delay-threshold", type=float, default=None)
    parser.add_argument("--min-risk-score", type=float, default=0.0)
    parser.add_argument("--min-action-gain", type=float, default=0.0)
    parser.add_argument("--role-hidden-dim", type=int, default=64)
    parser.add_argument("--role-lr", type=float, default=1e-3)
    parser.add_argument("--role-epsilon", type=float, default=0.20)
    parser.add_argument("--role-epsilon-final", type=float, default=0.05)
    parser.add_argument("--role-epsilon-decay-episodes", type=int, default=200)
    parser.add_argument("--training-algorithm", default="independent_dqn", choices=["independent_dqn", "qmix"])
    parser.add_argument("--qmix-embed-dim", type=int, default=32)
    parser.add_argument("--qmix-lr", type=float, default=None)
    parser.add_argument("--role-checkpoint", default=None, help="resume role DQN/QMIX weights and optimizers")
    parser.add_argument("--disable-migration", action="store_true")
    parser.add_argument("--disable-reroute", action="store_true")
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--updates-per-train", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--plan-only", action="store_true", help="train on planned actions without applying them")
    parser.add_argument("--reward-hotspot", type=float, default=2.0)
    parser.add_argument("--reward-delay", type=float, default=1.0)
    parser.add_argument("--reward-delay-scale", type=float, default=10.0)
    parser.add_argument("--penalty-migration", type=float, default=0.50)
    parser.add_argument("--penalty-reroute", type=float, default=0.30)
    parser.add_argument("--penalty-tree", type=float, default=0.10)
    parser.add_argument("--penalty-failure", type=float, default=2.0)
    parser.add_argument("--penalty-noop", type=float, default=0.05)
    parser.add_argument("--reward-coord", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    os.chdir(ROOT)
    args = parse_args()
    args.role_agent_mode = "trainable"
    _, env, _, coordinator, role_env = build_runtime(args)

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    trainer = RoleMARLTrainer(
        env=env,
        coordinator=coordinator,
        role_env=role_env,
        output_dir=output_dir,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        reconfig_interval=args.reconfig_interval,
        warmup_episodes=args.warmup_episodes,
        train_every=args.train_every,
        updates_per_train=args.updates_per_train,
        batch_size=args.batch_size,
        save_every=args.save_every,
        apply_reconfig=not args.plan_only,
        epsilon_final=args.role_epsilon_final,
        epsilon_decay_episodes=args.role_epsilon_decay_episodes,
        training_algorithm=args.training_algorithm,
        qmix_embed_dim=args.qmix_embed_dim,
        qmix_lr=args.qmix_lr,
        role_checkpoint=args.role_checkpoint,
        reward_weights={
            "hotspot": args.reward_hotspot,
            "delay": args.reward_delay,
            "delay_scale": args.reward_delay_scale,
            "migration": args.penalty_migration,
            "reroute": args.penalty_reroute,
            "tree": args.penalty_tree,
            "failure": args.penalty_failure,
            "noop": args.penalty_noop,
            "coord": args.reward_coord,
        },
    )
    summary = trainer.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
