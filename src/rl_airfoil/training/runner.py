from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from rl_airfoil.config.schema import ExperimentConfig, create_run_dir, write_experiment_metadata
from rl_airfoil.core.env import AirfoilEnv
from rl_airfoil.evaluators.surrogate import SurrogateEvaluator
from rl_airfoil.evaluators.xfoil import XFOILEvaluator
from rl_airfoil.logging.xai_logger import CSVLogger


def _make_evaluator(cfg: ExperimentConfig):
    if cfg.evaluator.lower() == "surrogate":
        return SurrogateEvaluator(cfg.surrogate_checkpoint_path, cfg.surrogate_model_name)
    if cfg.evaluator.lower() == "xfoil":
        return XFOILEvaluator()
    raise ValueError(f"Unsupported evaluator: {cfg.evaluator}")


def train_td3(cfg: ExperimentConfig, base_logs: Path = Path("logs"), checkpoints_dir: Path = Path("checkpoints")) -> Path:
    run_dir = create_run_dir(base_logs, "td3")
    checkpoints_dir.mkdir(exist_ok=True)

    env = AirfoilEnv(cfg, _make_evaluator(cfg))
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3("MlpPolicy", env, action_noise=action_noise, seed=cfg.seed, verbose=0)
    model.learn(total_timesteps=cfg.total_timesteps)

    ckpt = checkpoints_dir / "td3_model.zip"
    model.save(str(ckpt))

    write_experiment_metadata(cfg, run_dir, str(ckpt), normalization_stats={"input": "none", "target": "none"})
    (run_dir / "training_update_logs.csv").write_text("update_id,critic_loss_Q1,critic_loss_Q2,actor_loss\n", encoding="utf-8")
    (run_dir / "replay_sample_logs.csv").write_text("buffer_index,state_t,action_t,reward_t,next_state_t,done_t,sampling_weight\n", encoding="utf-8")
    return run_dir


def evaluate_td3(cfg: ExperimentConfig, run_dir: Path, episodes: int = 10):
    env = AirfoilEnv(cfg, _make_evaluator(cfg))
    model = TD3.load("checkpoints/td3_model.zip")

    rollout_cols = ["experiment_id","algorithm","seed","episode_id","step_id","global_step",
                    "state_CST_u1","state_CST_u2","state_CST_u3","state_CST_u4","state_CST_l1","state_CST_l2","state_CST_l3","state_CST_l4",
                    "AoA","Re","log10_Re","CL","CD","CM","CL_CD","t_c",
                    "action_u1","action_u2","action_u3","action_u4","action_l1","action_l2","action_l3","action_l4",
                    "action_norm","reward_total","reward_objective_term","reward_CM_penalty","reward_tc_penalty","done","done_reason"]
    policy_cols = ["episode_id","step_id","actor_action_u1","actor_action_u2","actor_action_u3","actor_action_u4","actor_action_l1","actor_action_l2","actor_action_l3","actor_action_l4","Q1","Q2","Q_min","Q_disagreement","target_Q","td_error_Q1","td_error_Q2"]

    rollout = CSVLogger(run_dir / "rollout_step_logs.csv", rollout_cols)
    policy = CSVLogger(run_dir / "policy_outputs.csv", policy_cols)

    summaries = []
    global_step = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = False
        step = 0
        total_reward = 0.0
        best_clcd = -1e9
        initial_clcd = float(obs[13])
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            q1, q2 = model.critic(obs_tensor=model.policy.obs_to_tensor(obs)[0], actions=model.policy.obs_to_tensor(action)[0])
            nxt, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            row = {"experiment_id": run_dir.name, "algorithm": "TD3", "seed": cfg.seed, "episode_id": ep, "step_id": step, "global_step": global_step,
                   "AoA": cfg.aoa, "Re": cfg.re, "log10_Re": np.log10(cfg.re), "CL": info["CL"], "CD": info["CD"], "CM": info["CM"], "CL_CD": info["CL_CD"], "t_c": info["t_c"],
                   "reward_total": reward, "reward_objective_term": info["reward_objective_term"], "reward_CM_penalty": info["reward_CM_penalty"], "reward_tc_penalty": info["reward_tc_penalty"],
                   "done": done, "done_reason": info["done_reason"], "action_norm": float(np.linalg.norm(action))}
            for i in range(4): row[f"state_CST_u{i+1}"] = float(obs[i])
            for i in range(4): row[f"state_CST_l{i+1}"] = float(obs[4+i])
            for i in range(4): row[f"action_u{i+1}"] = float(action[i])
            for i in range(4): row[f"action_l{i+1}"] = float(action[4+i])
            rollout.log(row)

            q1v = float(q1.detach().cpu().numpy().ravel()[0]); q2v = float(q2.detach().cpu().numpy().ravel()[0])
            policy.log({"episode_id": ep, "step_id": step, **{f"actor_action_u{i+1}": float(action[i]) for i in range(4)}, **{f"actor_action_l{i+1}": float(action[4+i]) for i in range(4)},
                        "Q1": q1v, "Q2": q2v, "Q_min": min(q1v,q2v), "Q_disagreement": abs(q1v-q2v), "target_Q": "", "td_error_Q1": "", "td_error_Q2": ""})

            total_reward += reward
            best_clcd = max(best_clcd, info["CL_CD"])
            obs = nxt
            step += 1
            global_step += 1

        summaries.append({"experiment_id": run_dir.name, "algorithm": "TD3", "seed": cfg.seed, "episode_id": ep,
                          "initial_CL_CD": initial_clcd, "final_CL_CD": info["CL_CD"], "best_CL_CD": best_clcd,
                          "final_CL": info["CL"], "final_CD": info["CD"], "final_CM": info["CM"], "final_t_c": info["t_c"],
                          "total_reward": total_reward, "total_penalty": info["reward_CM_penalty"] + info["reward_tc_penalty"],
                          "mean_action_norm": "", "max_action_norm": "", "constraint_violation_count": int((not info['is_CM_feasible']) or (not info['is_tc_feasible'])),
                          "CM_violation_count": int(not info['is_CM_feasible']), "tc_violation_count": int(not info['is_tc_feasible']),
                          "invalid_geometry_count": 0, "solver_error_count": 0, "done_reason": info["done_reason"], "episode_length": step,
                          "is_final_feasible": bool(info['is_CM_feasible'] and info['is_tc_feasible'])})

    rollout.close(); policy.close()
    pd.DataFrame(summaries).to_csv(run_dir / "episode_summary.csv", index=False)
    (run_dir / "xfoil_validation_logs.csv").write_text("algorithm,episode_id,step_id\n", encoding="utf-8")
