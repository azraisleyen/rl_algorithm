from __future__ import annotations
import argparse
from pathlib import Path
from rl_airfoil.config.schema import ExperimentConfig, TD3Hyperparameters
from rl_airfoil.training.runner import train_td3, evaluate_td3


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--algorithm", default="td3", choices=["td3"])
        sp.add_argument("--evaluator", default="surrogate", choices=["surrogate"])
        sp.add_argument("--surrogate-model-name", default="S-1D", choices=["S-1D", "S-2D", "S-3D"])
        sp.add_argument("--surrogate-checkpoint-path", default="checkpoints/surrogate_s1d.pt")
        sp.add_argument("--scaler-json-path", default="checkpoints/scalers.json")
        sp.add_argument("--rl-checkpoint-path", default="checkpoints/td3_surrogate_s-1d.zip")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--total-timesteps", type=int, default=200000)
        sp.add_argument("--aoa", type=float, default=2.0)
        sp.add_argument("--re", type=float, default=1e6)
        sp.add_argument("--action-scale", type=float, default=0.02)
        sp.add_argument("--episode-max-steps", type=int, default=25)

        sp.add_argument("--td3-learning-rate", type=float, default=1e-3)
        sp.add_argument("--td3-buffer-size", type=int, default=100000)
        sp.add_argument("--td3-learning-starts", type=int, default=1000)
        sp.add_argument("--td3-batch-size", type=int, default=256)
        sp.add_argument("--td3-tau", type=float, default=0.005)
        sp.add_argument("--td3-gamma", type=float, default=0.99)
        sp.add_argument("--td3-policy-delay", type=int, default=2)
        sp.add_argument("--td3-target-policy-noise", type=float, default=0.2)
        sp.add_argument("--td3-target-noise-clip", type=float, default=0.5)
        sp.add_argument("--td3-action-noise-sigma", type=float, default=0.1)
        # Reward içinde action saturation'ı azaltmak için kullanılan ceza katsayısı
        sp.add_argument("--w-action", type=float, default=0.01)

    t = sub.add_parser("train")
    add_common(t)

    e = sub.add_parser("evaluate")
    add_common(e)
    e.add_argument("--run-dir", required=False, default=None)
    e.add_argument("--episodes", type=int, default=10)
    e.add_argument("--aoa-sweep", default="-2,0,2,4,6,8")
    return p


def to_cfg(args):
    cfg = ExperimentConfig(
        algorithm=args.algorithm,
        evaluator=args.evaluator,
        surrogate_model_name=args.surrogate_model_name,
        surrogate_checkpoint_path=args.surrogate_checkpoint_path,
        scaler_json_path=args.scaler_json_path,
        rl_checkpoint_path=args.rl_checkpoint_path,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        aoa=args.aoa,
        re=args.re,
        episode_max_steps=args.episode_max_steps,
        action_scale=args.action_scale,
        td3=TD3Hyperparameters(
            learning_rate=args.td3_learning_rate,
            buffer_size=args.td3_buffer_size,
            learning_starts=args.td3_learning_starts,
            batch_size=args.td3_batch_size,
            tau=args.td3_tau,
            gamma=args.td3_gamma,
            policy_delay=args.td3_policy_delay,
            target_policy_noise=args.td3_target_policy_noise,
            target_noise_clip=args.td3_target_noise_clip,
            action_noise_sigma=args.td3_action_noise_sigma,
        ),
    )

    cfg.reward_weights.w_action = args.w_action

    return cfg


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = to_cfg(args)
    if args.command == "train":
        run_dir = train_td3(cfg)
        print(run_dir)
    else:
        run_dir = Path(args.run_dir) if args.run_dir else None
        evaluate_td3(cfg, run_dir, episodes=args.episodes, aoa_sweep=args.aoa_sweep)
        print(args.run_dir if args.run_dir else "auto-generated")
