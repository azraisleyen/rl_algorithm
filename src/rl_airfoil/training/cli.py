from __future__ import annotations
import argparse
from pathlib import Path
from rl_airfoil.config.schema import ExperimentConfig
from rl_airfoil.training.runner import train_td3, evaluate_td3


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--algorithm", default="td3", choices=["td3"])
        sp.add_argument("--evaluator", default="surrogate", choices=["surrogate", "xfoil"])
        sp.add_argument("--surrogate-model-name", default="S-1D", choices=["S-1D", "S-2D", "S-3D"])
        sp.add_argument("--surrogate-checkpoint-path", default="checkpoints/surrogate_s1d.pt")
        sp.add_argument("--scaler-x-path", default="checkpoints/scaler_x.pkl")
        sp.add_argument("--scaler-y-path", default="checkpoints/scaler_y.pkl")
        sp.add_argument("--rl-checkpoint-path", default="checkpoints/td3_surrogate_s-1d.zip")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--total-timesteps", type=int, default=200000)
        sp.add_argument("--aoa", type=float, default=2.0)
        sp.add_argument("--re", type=float, default=1e6)

    t = sub.add_parser("train")
    add_common(t)

    e = sub.add_parser("evaluate")
    add_common(e)
    e.add_argument("--run-dir", required=True)
    e.add_argument("--episodes", type=int, default=10)
    e.add_argument("--aoa-sweep", default="-2,0,2,4,6,8")
    return p


def to_cfg(args):
    return ExperimentConfig(
        algorithm=args.algorithm,
        evaluator=args.evaluator,
        surrogate_model_name=args.surrogate_model_name,
        surrogate_checkpoint_path=args.surrogate_checkpoint_path,
        scaler_x_path=args.scaler_x_path,
        scaler_y_path=args.scaler_y_path,
        rl_checkpoint_path=args.rl_checkpoint_path,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        aoa=args.aoa,
        re=args.re,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = to_cfg(args)
    if args.command == "train":
        run_dir = train_td3(cfg)
        print(run_dir)
    else:
        evaluate_td3(cfg, Path(args.run_dir), episodes=args.episodes, aoa_sweep=args.aoa_sweep)
        print(args.run_dir)
