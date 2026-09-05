#!/usr/bin/env python3
"""Online QMIX or Weighted-QMIX training in the deployment environment."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.batch_deployment_wqmix import WeightedQMIXLearner  # noqa: E402
from core.marl.deployment_dataset import FeatureNormalizer  # noqa: E402
from core.marl.deployment_env import BatchDeploymentEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", required=True)
    parser.add_argument("--train-trace", action="append", required=True)
    parser.add_argument("--algorithm", choices=("qmix", "wqmix"), default="wqmix")
    parser.add_argument("--profile", default="sdn/topologies/us_backbone_28_bw90.json")
    parser.add_argument(
        "--output", default="artifacts/runs/deployment/wqmix_train"
    )
    parser.add_argument("--episodes-per-trace", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--mixer-embed-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon-start", type=float, default=0.20)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--target-update-interval", type=int, default=200)
    parser.add_argument("--bandwidth-utilization-limit", type=float, default=0.8)
    parser.add_argument("--queue-safety-factor", type=float, default=2.0)
    parser.add_argument("--sla-risk-weight", type=float, default=8.0)
    parser.add_argument("--sla-violation-penalty", type=float, default=20.0)
    parser.add_argument(
        "--bc-anchor-weight",
        type=float,
        default=1.0,
        help="cross-entropy weight that keeps fine-tuning close to the frozen BC policy",
    )
    parser.add_argument("--decoder-top-r", type=int, default=4)
    parser.add_argument("--decoder-time-budget-ms", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--resume",
        default=None,
        help="resume from a qmix_latest.pt or wqmix_latest.pt checkpoint",
    )
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def choose_rankings(
    learner: WeightedQMIXLearner,
    observation: Dict[str, torch.Tensor],
    epsilon: float,
    generator: random.Random,
) -> tuple[list[list[int]], list[list[float]]]:
    active = observation["agent_mask"][0].bool()
    mask = observation["action_mask"][0].bool()
    with torch.inference_mode():
        q_values = learner.q_network(
            observation["request_observations"].to(learner.device),
            observation["candidate_features"].to(learner.device),
            observation["agent_mask"].to(learner.device),
        )[0]
    rankings: list[list[int]] = []
    scores: list[list[float]] = []
    for agent_index in range(int(active.sum())):
        valid = torch.where(mask[agent_index])[0].tolist()
        if not valid:
            raise RuntimeError(f"active agent {agent_index} has no valid action")
        ranking = sorted(
            map(int, valid),
            key=lambda action: float(q_values[agent_index, action]),
            reverse=True,
        )
        if generator.random() < epsilon:
            exploratory = int(generator.choice(valid))
            ranking.remove(exploratory)
            ranking.insert(0, exploratory)
        rankings.append(ranking)
        scores.append([float(value) for value in q_values[agent_index].tolist()])
    return rankings, scores


def transition(
    observation: Dict[str, torch.Tensor],
    actions: Sequence[int],
    next_observation: Dict[str, torch.Tensor],
    reward: float,
    done: bool,
    teacher_actions: Sequence[int] | None = None,
) -> Dict[str, torch.Tensor]:
    padded_actions = torch.zeros(1, observation["agent_mask"].shape[1], dtype=torch.long)
    padded_actions[0, : len(actions)] = torch.tensor(list(actions), dtype=torch.long)
    result = {
        "request_observations": observation["request_observations"],
        "candidate_features": observation["candidate_features"],
        "action_mask": observation["action_mask"],
        "agent_mask": observation["agent_mask"],
        "states": observation["states"],
        "actions": padded_actions,
        "rewards": torch.tensor([float(reward)], dtype=torch.float32),
        "next_request_observations": next_observation["request_observations"],
        "next_candidate_features": next_observation["candidate_features"],
        "next_action_mask": next_observation["action_mask"],
        "next_agent_mask": next_observation["agent_mask"],
        "next_states": next_observation["states"],
        "dones": torch.tensor([bool(done)], dtype=torch.bool),
    }
    if teacher_actions is not None:
        padded_teacher = torch.zeros_like(padded_actions)
        padded_teacher[0, : len(teacher_actions)] = torch.tensor(
            list(teacher_actions), dtype=torch.long
        )
        result["teacher_actions"] = padded_teacher
    return result


def main() -> int:
    args = parse_args()
    effective_alpha = 1.0 if args.algorithm == "qmix" else float(args.alpha)
    if (
        args.episodes_per_trace <= 0
        or args.lr <= 0.0
        or not 0.0 < args.gamma <= 1.0
        or not 0.0 < args.bandwidth_utilization_limit <= 1.0
        or min(
            args.queue_safety_factor,
            args.sla_risk_weight,
            args.sla_violation_penalty,
            args.bc_anchor_weight,
            args.decoder_time_budget_ms,
        )
        < 0.0
        or args.decoder_top_r <= 0
    ):
        raise ValueError("invalid training parameter")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    bc_path = resolve(args.bc_checkpoint)
    bc_checkpoint = torch.load(bc_path, map_location=device, weights_only=False)
    normalizer = FeatureNormalizer.from_dict(bc_checkpoint["normalizer"])
    hidden_dim = int(args.hidden_dim or bc_checkpoint["hidden_dim"])
    learner = WeightedQMIXLearner(
        int(bc_checkpoint["request_dim"]),
        int(bc_checkpoint["candidate_dim"]),
        int(bc_checkpoint["state_dim"]),
        max_agents=int(bc_checkpoint["max_agents"]),
        hidden_dim=hidden_dim,
        mixer_embed_dim=args.mixer_embed_dim,
        alpha=effective_alpha,
        lr=args.lr,
        gamma=args.gamma,
        target_update_interval=args.target_update_interval,
        imitation_weight=args.bc_anchor_weight,
        device=device,
    )
    learner.q_network.load_state_dict(bc_checkpoint["state_dict"])
    learner.target_q_network.load_state_dict(learner.q_network.state_dict())
    bc_teacher = copy.deepcopy(learner.q_network).eval()
    for parameter in bc_teacher.parameters():
        parameter.requires_grad_(False)
    rng = random.Random(args.seed)
    output_dir = resolve(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    total_updates = 0
    completed_runs: set[tuple[str, int]] = set()

    if args.resume:
        resume_path = resolve(args.resume)
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        learner.q_network.load_state_dict(resume["state_dict"])
        learner.mixer.load_state_dict(resume["qmix_state_dict"])
        learner.target_q_network.load_state_dict(resume["target_state_dict"])
        learner.target_mixer.load_state_dict(resume["target_qmix_state_dict"])
        learner.optimizer.load_state_dict(resume["optimizer_state_dict"])
        learner.update_steps = int(resume.get("learner_update_steps", 0))
        total_updates = int(resume.get("total_updates", learner.update_steps))
        episodes = list(resume.get("episodes", []))
        completed_runs = {
            (str(row["trace"]), int(row["episode"]))
            for row in episodes
        }
        rng_state = resume.get("python_rng_state")
        if rng_state is not None:
            rng.setstate(rng_state)

    def checkpoint_payload(include_training_state: bool) -> dict[str, Any]:
        payload = {
            "model": f"deployment_candidate_{args.algorithm}",
            "dataset_version": "deployment_topk_oracle_v3",
            "candidate_feature_schema": "v3.1_source_neutral",
            "state_dict": copy.deepcopy(learner.q_network.state_dict()),
            "qmix_state_dict": copy.deepcopy(learner.mixer.state_dict()),
            "normalizer": normalizer.to_dict(),
            "request_dim": int(bc_checkpoint["request_dim"]),
            "candidate_dim": int(bc_checkpoint["candidate_dim"]),
            "state_dim": int(bc_checkpoint["state_dim"]),
            "hidden_dim": hidden_dim,
            "max_agents": int(bc_checkpoint["max_agents"]),
            "max_candidates": int(bc_checkpoint["max_candidates"]),
            "training": {
                "algorithm": (
                    "online_qmix"
                    if args.algorithm == "qmix"
                    else "online_optimistic_weighted_qmix"
                ),
                "updates": total_updates,
                "train_traces": [str(resolve(value)) for value in args.train_trace],
                "episodes_per_trace": args.episodes_per_trace,
                "epsilon_start": args.epsilon_start,
                "epsilon_end": args.epsilon_end,
                "gamma": args.gamma,
                "alpha": effective_alpha,
                "bandwidth_utilization_limit": args.bandwidth_utilization_limit,
                "queue_safety_factor": args.queue_safety_factor,
                "sla_risk_weight": args.sla_risk_weight,
                "sla_violation_penalty": args.sla_violation_penalty,
                "bc_anchor_weight": args.bc_anchor_weight,
                "decoder_top_r": args.decoder_top_r,
                "decoder_time_budget_ms": args.decoder_time_budget_ms,
            },
        }
        if include_training_state:
            payload.update({
                "target_state_dict": copy.deepcopy(
                    learner.target_q_network.state_dict()
                ),
                "target_qmix_state_dict": copy.deepcopy(
                    learner.target_mixer.state_dict()
                ),
                "optimizer_state_dict": copy.deepcopy(
                    learner.optimizer.state_dict()
                ),
                "learner_update_steps": int(learner.update_steps),
                "total_updates": int(total_updates),
                "episodes": copy.deepcopy(episodes),
                "python_rng_state": rng.getstate(),
            })
        return payload

    for trace_value in args.train_trace:
        trace = resolve(trace_value)
        for episode_index in range(args.episodes_per_trace):
            run_key = (str(trace.resolve()), int(episode_index))
            if run_key in completed_runs:
                continue
            env = BatchDeploymentEnv(
                trace,
                resolve(args.profile),
                normalizer,
                max_agents=int(bc_checkpoint["max_agents"]),
                top_k=int(bc_checkpoint["max_candidates"]) - 1,
                max_requests=100,
                bandwidth_utilization_limit=args.bandwidth_utilization_limit,
                queue_safety_factor=args.queue_safety_factor,
                sla_risk_weight=args.sla_risk_weight,
                sla_violation_penalty=args.sla_violation_penalty,
            )
            observation = env.reset()
            episode_reward = 0.0
            accepted = 0
            modeled_sla_violations = 0
            modeled_sla_risk_total = 0.0
            decoder_timeouts = 0
            decoder_greedy_budget_exhaustions = 0
            decoder_repair_budget_exhaustions = 0
            decoder_elapsed_ms = 0.0
            decoder_adjusted_requests = 0
            steps = 0
            while env.active_agents:
                progress = total_updates / max(1.0, len(args.train_trace) * args.episodes_per_trace * 90.0)
                epsilon = max(
                    args.epsilon_end,
                    args.epsilon_start
                    - (args.epsilon_start - args.epsilon_end) * min(1.0, progress),
                )
                rankings, q_scores = choose_rankings(
                    learner, observation, epsilon, rng
                )
                decode = env.decode_rankings(
                    rankings,
                    scores=q_scores,
                    top_r=args.decoder_top_r,
                    time_budget_ms=args.decoder_time_budget_ms,
                )
                actions = list(decode.actions)
                decoder_timeouts += int(decode.timed_out)
                decoder_greedy_budget_exhaustions += int(
                    decode.greedy_budget_exhausted
                )
                decoder_repair_budget_exhaustions += int(
                    decode.repair_budget_exhausted
                )
                decoder_elapsed_ms += float(decode.elapsed_ms)
                decoder_adjusted_requests += sum(
                    int(action != ranking[0])
                    for action, ranking in zip(actions, rankings)
                    if ranking
                )
                with torch.inference_mode():
                    teacher_q = bc_teacher(
                        observation["request_observations"].to(device),
                        observation["candidate_features"].to(device),
                        observation["agent_mask"].to(device),
                    )[0]
                    teacher_q = teacher_q.masked_fill(
                        ~observation["action_mask"][0].bool().to(device),
                        torch.finfo(teacher_q.dtype).min,
                    )
                    teacher_actions = teacher_q.argmax(dim=-1)[: len(actions)].tolist()
                next_observation, reward, done, info = env.step(
                    actions, expected_version=decode.snapshot_version
                )
                batch = transition(
                    observation,
                    actions,
                    next_observation,
                    reward,
                    done,
                    teacher_actions=teacher_actions,
                )
                metrics = learner.train_step(batch)
                total_updates += 1
                episode_reward += reward
                accepted += int(info["accepted"])
                modeled_sla_violations += int(info["modeled_sla_violations"])
                modeled_sla_risk_total += float(info["mean_modeled_sla_risk"])
                steps += 1
                observation = next_observation
                if done:
                    break
            episodes.append({
                "trace": str(trace.resolve()),
                "episode": episode_index,
                "steps": steps,
                "accepted": accepted,
                "modeled_sla_violations": modeled_sla_violations,
                "mean_batch_modeled_sla_risk": (
                    modeled_sla_risk_total / max(1, steps)
                ),
                "decoder_timeouts": decoder_timeouts,
                "decoder_greedy_budget_exhaustions": (
                    decoder_greedy_budget_exhaustions
                ),
                "decoder_repair_budget_exhaustions": (
                    decoder_repair_budget_exhaustions
                ),
                "mean_decoder_ms": decoder_elapsed_ms / max(1, steps),
                "decoder_adjusted_requests": decoder_adjusted_requests,
                "reward": episode_reward,
                "last_loss": float(metrics["loss"]),
            })
            completed_runs.add(run_key)
            torch.save(
                checkpoint_payload(include_training_state=True),
                output_dir / f"{args.algorithm}_latest.pt",
            )
            print(
                f"completed trace={trace.name} episode={episode_index + 1}/"
                f"{args.episodes_per_trace} updates={total_updates} "
                f"accepted={accepted}",
                flush=True,
            )

    checkpoint_path = output_dir / f"{args.algorithm}_final.pt"
    checkpoint = checkpoint_payload(include_training_state=False)
    torch.save(checkpoint, checkpoint_path)
    report = {
        "valid": True,
        "checkpoint": str(checkpoint_path.resolve()),
        "updates": total_updates,
        "episodes": episodes,
        "warning": "training traces are not an independent test; evaluate on held-out seeds",
    }
    (output_dir / f"{args.algorithm}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
