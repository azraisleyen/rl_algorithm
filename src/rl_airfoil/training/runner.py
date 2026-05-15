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
        return SurrogateEvaluator(
            checkpoint_path=cfg.surrogate_checkpoint_path,
            model_name=cfg.surrogate_model_name,
            scaler_x_path=cfg.scaler_x_path,
            scaler_y_path=cfg.scaler_y_path,
        )
    if cfg.evaluator.lower() == "xfoil":
        return XFOILEvaluator()
    raise ValueError(f"Unsupported evaluator: {cfg.evaluator}")


def train_td3(cfg: ExperimentConfig, base_logs: Path = Path("logs"), checkpoints_dir: Path = Path("checkpoints")) -> Path:
    run_dir = create_run_dir(base_logs, "td3")
    checkpoints_dir.mkdir(exist_ok=True)

    env = AirfoilEnv(cfg, _make_evaluator(cfg))
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3("MlpPolicy", env, action_noise=action_noise, seed=cfg.seed, verbose=1)
    model.learn(total_timesteps=cfg.total_timesteps)

    ckpt = checkpoints_dir / f"td3_{cfg.evaluator}_{cfg.surrogate_model_name.lower()}.zip"
    model.save(str(ckpt))
    cfg.rl_checkpoint_path = str(ckpt)

    write_experiment_metadata(cfg, run_dir, normalization_stats={"input": "scaler_x.pkl", "target": "scaler_y.pkl"})
    (run_dir / "training_update_logs.csv").write_text(
        "update_id,critic_loss_Q1,critic_loss_Q2,actor_loss,target_Q_mean,target_Q_std,Q1_mean,Q2_mean,policy_delay_step,learning_rate_actor,learning_rate_critic,gradient_norm_actor,gradient_norm_critic\n",
        encoding="utf-8",
    )
    return run_dir


def evaluate_td3(cfg: ExperimentConfig, run_dir: Path, episodes: int = 10, aoa_sweep: str = "-2,0,2,4,6,8"):
    env = AirfoilEnv(cfg, _make_evaluator(cfg))
    model = TD3.load(cfg.rl_checkpoint_path)

    rollout_cols = ["experiment_id","algorithm","seed","episode_id","step_id","global_step",
                    "state_CST_u1","state_CST_u2","state_CST_u3","state_CST_u4","state_CST_l1","state_CST_l2","state_CST_l3","state_CST_l4",
                    "AoA","Re","log10_Re","CL","CD","CM","CL_CD","t_c",
                    "action_u1","action_u2","action_u3","action_u4","action_l1","action_l2","action_l3","action_l4",
                    "action_norm","reward_total","reward_objective_term","reward_CM_penalty","reward_tc_penalty","reward_local_thickness_penalty","penalty_total",
                    "done","truncated","terminated","done_reason"]
    policy_cols = ["episode_id","step_id","actor_action_u1","actor_action_u2","actor_action_u3","actor_action_u4","actor_action_l1","actor_action_l2","actor_action_l3","actor_action_l4","Q1","Q2","Q_min","Q_disagreement","target_Q","td_error_Q1","td_error_Q2"]

    rollout = CSVLogger(run_dir / "rollout_step_logs.csv", rollout_cols)
    policy = CSVLogger(run_dir / "policy_outputs.csv", policy_cols)

    summaries = []
    global_step = 0
    best_record = None

    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = False
        step = 0
        total_reward = 0.0
        penalties = 0.0
        action_norms = []
        best_clcd = -1e9
        initial_clcd = float(obs[13])
        last_info = None
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs_t = model.policy.obs_to_tensor(obs)[0]
            act_t = model.policy.obs_to_tensor(action)[0]
            q1, q2 = model.critic(obs_t, act_t)
            nxt, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            pen_total = info["reward_CM_penalty"] + info["reward_tc_penalty"]
            row = {"experiment_id": run_dir.name, "algorithm": "TD3", "seed": cfg.seed, "episode_id": ep, "step_id": step, "global_step": global_step,
                   "AoA": cfg.aoa, "Re": cfg.re, "log10_Re": np.log10(cfg.re), "CL": info["CL"], "CD": info["CD"], "CM": info["CM"], "CL_CD": info["CL_CD"], "t_c": info["t_c"],
                   "reward_total": reward, "reward_objective_term": info["reward_objective_term"], "reward_CM_penalty": info["reward_CM_penalty"], "reward_tc_penalty": info["reward_tc_penalty"],
                   "reward_local_thickness_penalty": 0.0, "penalty_total": pen_total,
                   "done": done, "truncated": truncated, "terminated": terminated, "done_reason": info["done_reason"], "action_norm": float(np.linalg.norm(action))}
            for i in range(4): row[f"state_CST_u{i+1}"] = float(obs[i])
            for i in range(4): row[f"state_CST_l{i+1}"] = float(obs[4+i])
            for i in range(4): row[f"action_u{i+1}"] = float(action[i])
            for i in range(4): row[f"action_l{i+1}"] = float(action[4+i])
            rollout.log(row)

            q1v = float(q1.detach().cpu().numpy().ravel()[0]); q2v = float(q2.detach().cpu().numpy().ravel()[0])
            policy.log({"episode_id": ep, "step_id": step, **{f"actor_action_u{i+1}": float(action[i]) for i in range(4)}, **{f"actor_action_l{i+1}": float(action[4+i]) for i in range(4)},
                        "Q1": q1v, "Q2": q2v, "Q_min": min(q1v,q2v), "Q_disagreement": abs(q1v-q2v), "target_Q": "", "td_error_Q1": "", "td_error_Q2": ""})

            total_reward += reward
            penalties += pen_total
            action_norms.append(float(np.linalg.norm(action)))
            best_clcd = max(best_clcd, info["CL_CD"])
            if best_record is None or info["CL_CD"] > best_record["CL_CD"]:
                best_record = {"episode_id": ep, "step_id": step, "cst": info["next_cst"].copy(), **info}
            obs = nxt
            step += 1
            global_step += 1
            last_info = info

        summaries.append({"experiment_id": run_dir.name, "algorithm": "TD3", "seed": cfg.seed, "episode_id": ep,
                          "initial_CL_CD": initial_clcd, "final_CL_CD": last_info["CL_CD"], "best_CL_CD": best_clcd,
                          "final_CL": last_info["CL"], "final_CD": last_info["CD"], "final_CM": last_info["CM"], "final_t_c": last_info["t_c"],
                          "total_reward": total_reward, "total_penalty": penalties,
                          "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
                          "max_action_norm": float(np.max(action_norms)) if action_norms else 0.0,
                          "constraint_violation_count": int((not last_info['is_CM_feasible']) or (not last_info['is_tc_feasible'])),
                          "CM_violation_count": int(not last_info['is_CM_feasible']), "tc_violation_count": int(not last_info['is_tc_feasible']),
                          "invalid_geometry_count": 0, "solver_error_count": 0, "done_reason": last_info["done_reason"], "episode_length": step,
                          "is_final_feasible": bool(last_info['is_CM_feasible'] and last_info['is_tc_feasible'])})

    rollout.close(); policy.close()
    pd.DataFrame(summaries).to_csv(run_dir / "episode_summary.csv", index=False)

    # AoA sweep and optional surrogate-xfoil validation
    xeval = XFOILEvaluator()
    sweep_rows = []
    if best_record is not None:
        aoa_values = [float(x) for x in aoa_sweep.split(",") if x.strip()]
        cst = best_record["cst"]
        surrogate_eval = _make_evaluator(cfg)
        for a in aoa_values:
            s = surrogate_eval.evaluate(cst, a, cfg.re)
            x = xeval.evaluate(cst, a, cfg.re)
            s_ratio = s.cl / max(s.cd, cfg.cd_lower_bound)
            x_ratio = x.cl / max(x.cd, cfg.cd_lower_bound)
            sweep_rows.append({
                "algorithm": "TD3", "episode_id": best_record["episode_id"], "step_id": best_record["step_id"],
                "AoA": a, "Re": cfg.re,
                **{f"CST_u{i+1}": float(cst[i]) for i in range(4)}, **{f"CST_l{i+1}": float(cst[4+i]) for i in range(4)},
                "CL_surrogate": s.cl, "CD_surrogate": s.cd, "CM_surrogate": s.cm, "CL_CD_surrogate": s_ratio,
                "CL_xfoil": x.cl, "CD_xfoil": x.cd, "CM_xfoil": x.cm, "CL_CD_xfoil": x_ratio,
                "abs_error_CL": abs(s.cl - x.cl), "abs_error_CD": abs(s.cd - x.cd), "abs_error_CM": abs(s.cm - x.cm), "abs_error_CL_CD": abs(s_ratio - x_ratio),
                "xfoil_converged": True, "xfoil_runtime_ms": 0.0,
            })
    pd.DataFrame(sweep_rows).to_csv(run_dir / "xfoil_validation_logs.csv", index=False)
