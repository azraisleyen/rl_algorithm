from __future__ import annotations
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from rl_airfoil.config.schema import ExperimentConfig, create_run_dir, write_experiment_metadata
from rl_airfoil.core.env import AirfoilEnv
from rl_airfoil.evaluators.surrogate import SurrogateEvaluator
from rl_airfoil.evaluators.xfoil import XFOILEvaluator
from rl_airfoil.logging.xai_logger import CSVLogger


class TrainingMetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.rows = []
        self.update_id = 0
        self.train_steps = []
        self.ep_rows = []
        self._ep_id = 0
        self._ep_reward = 0.0
        self._ep_len = 0

    def _on_step(self) -> bool:
        self.update_id += 1
        logs = self.model.logger.name_to_value
        rewards = float(np.asarray(self.locals.get("rewards", [0.0])).reshape(-1)[0])
        dones = bool(np.asarray(self.locals.get("dones", [False])).reshape(-1)[0])
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}
        obs = np.asarray(self.locals.get("new_obs", np.zeros((1,16), dtype=np.float32))).reshape(1,-1)[0]
        actions = np.asarray(self.locals.get("actions", np.zeros((1,8), dtype=np.float32))).reshape(1,-1)[0]
        self.train_steps.append({
            "episode_id": self._ep_id, "step_id": self._ep_len,
            "reward_total": rewards, "done_reason": info.get("done_reason", "running"),
            "CL": info.get("CL", np.nan), "CD": info.get("CD", np.nan), "CM": info.get("CM", np.nan), "CL_CD": info.get("CL_CD", np.nan), "t_c": info.get("t_c", np.nan),
            "action_norm": float(np.linalg.norm(actions)),
            "state": obs.tolist(), "action": actions.tolist()
        })
        self._ep_reward += rewards
        self._ep_len += 1
        if dones:
            self.ep_rows.append({"episode_id": self._ep_id, "total_reward": self._ep_reward, "episode_length": self._ep_len, "done_reason": info.get("done_reason", "")})
            self._ep_id += 1; self._ep_reward = 0.0; self._ep_len = 0

        self.rows.append({
            "update_id": self.update_id,
            "critic_loss_Q1": logs.get("train/critic_loss", ""),
            "critic_loss_Q2": logs.get("train/critic_loss", ""),
            "actor_loss": logs.get("train/actor_loss", ""),
            "target_Q_mean": "",
            "target_Q_std": "",
            "Q1_mean": "",
            "Q2_mean": "",
            "policy_delay_step": "",
            "learning_rate_actor": logs.get("train/learning_rate", ""),
            "learning_rate_critic": logs.get("train/learning_rate", ""),
            "gradient_norm_actor": "",
            "gradient_norm_critic": "",
        })
        return True


def _make_evaluator(cfg: ExperimentConfig):
    if cfg.evaluator.lower() == "surrogate":
        return SurrogateEvaluator(cfg.surrogate_checkpoint_path, cfg.surrogate_model_name, cfg.scaler_json_path)
    if cfg.evaluator.lower() == "xfoil":
        return XFOILEvaluator()
    raise ValueError(f"Unsupported evaluator: {cfg.evaluator}")


def _write_replay_sample(model: TD3, run_dir: Path, sample_n: int = 5000):
    rb = model.replay_buffer
    if rb is None or rb.size() == 0:
        pd.DataFrame(columns=["buffer_index","state_t","action_t","reward_t","next_state_t","done_t","sampling_weight"]).to_csv(run_dir / "replay_sample_logs.csv", index=False)
        return
    n = min(sample_n, rb.size())
    idxs = np.linspace(0, rb.size() - 1, n, dtype=int)
    rows = []
    for i in idxs:
        rows.append({
            "buffer_index": int(i),
            "state_t": np.asarray(rb.observations[i]).reshape(-1).tolist(),
            "action_t": np.asarray(rb.actions[i]).reshape(-1).tolist(),
            "reward_t": float(np.asarray(rb.rewards[i]).reshape(-1)[0]),
            "next_state_t": np.asarray(rb.next_observations[i]).reshape(-1).tolist(),
            "done_t": float(np.asarray(rb.dones[i]).reshape(-1)[0]),
            "sampling_weight": 1.0,
        })
    pd.DataFrame(rows).to_csv(run_dir / "replay_sample_logs.csv", index=False)


def train_td3(cfg: ExperimentConfig, base_logs: Path = Path("logs"), checkpoints_dir: Path = Path("checkpoints")) -> Path:
    run_dir = create_run_dir(base_logs, "td3")
    checkpoints_dir.mkdir(exist_ok=True)

    env = AirfoilEnv(cfg, _make_evaluator(cfg))
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3("MlpPolicy", env, action_noise=action_noise, seed=cfg.seed, verbose=1)

    cb = TrainingMetricsCallback()
    t0 = time.time()
    model.learn(total_timesteps=cfg.total_timesteps, callback=cb)
    train_wall_time = time.time() - t0

    ckpt = checkpoints_dir / f"td3_{cfg.evaluator}_{cfg.surrogate_model_name.lower()}.zip"
    model.save(str(ckpt))
    cfg.rl_checkpoint_path = str(ckpt)

    write_experiment_metadata(cfg, run_dir, normalization_stats={"source": cfg.scaler_json_path, "train_wall_time_sec": train_wall_time})
    pd.DataFrame(cb.rows).to_csv(run_dir / "training_update_logs.csv", index=False)
    pd.DataFrame(cb.train_steps).to_csv(run_dir / "train_rollout_step_logs.csv", index=False)
    pd.DataFrame(cb.ep_rows).to_csv(run_dir / "train_episode_summary.csv", index=False)
    _write_replay_sample(model, run_dir)

    # evaluate ayrı komutla çalıştırılacak; train sadece eğitim artefact'larını üretir.
    return run_dir


def evaluate_td3(cfg: ExperimentConfig, run_dir: Path, episodes: int = 10, aoa_sweep: str = "-2,0,2,4,6,8"):
    env = AirfoilEnv(cfg, _make_evaluator(cfg))
    model = TD3.load(cfg.rl_checkpoint_path)

    rollout_cols = ["experiment_id","algorithm","evaluator","seed","episode_id","step_id","global_step",
                    "state_CST_u1","state_CST_u2","state_CST_u3","state_CST_u4","state_CST_l1","state_CST_l2","state_CST_l3","state_CST_l4",
                    "AoA","Re","log10_Re","CL","CD","CM","CL_CD","t_c",
                    "action_u1","action_u2","action_u3","action_u4","action_l1","action_l2","action_l3","action_l4",
                    "action_norm","upper_action_norm","lower_action_norm","action_max_abs","action_saturation_count",
                    "next_CST_u1","next_CST_u2","next_CST_u3","next_CST_u4","next_CST_l1","next_CST_l2","next_CST_l3","next_CST_l4",
                    "delta_CST_u1","delta_CST_u2","delta_CST_u3","delta_CST_u4","delta_CST_l1","delta_CST_l2","delta_CST_l3","delta_CST_l4",
                    "CL_pred","CD_pred","CM_pred","CL_CD_pred","is_CM_feasible","is_tc_feasible","is_geometry_valid",
                    "reward_total","reward_objective_term","reward_CL_CD_term","reward_CM_penalty","reward_tc_penalty","reward_local_thickness_penalty","reward_invalid_geometry_penalty","reward_solver_error_penalty","penalty_total",
                    "CM_lower_violation","CM_upper_violation","tc_lower_violation","tc_upper_violation","local_thickness_violation",
                    "done","truncated","terminated","done_reason"]
    policy_cols = ["episode_id","step_id","actor_action_u1","actor_action_u2","actor_action_u3","actor_action_u4","actor_action_l1","actor_action_l2","actor_action_l3","actor_action_l4","Q1","Q2","Q_min","Q_disagreement","target_Q","td_error_Q1","td_error_Q2"]

    rollout = CSVLogger(run_dir / "rollout_step_logs.csv", rollout_cols)
    policy = CSVLogger(run_dir / "policy_outputs.csv", policy_cols)

    summaries = []
    global_step = 0
    best_record = None
    eval_start = time.time()

    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = False
        step = 0
        total_reward = 0.0
        penalties = 0.0
        action_norms = []
        best_clcd = -1e9
        initial_clcd = float(obs[14])
        last_info = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs_t = model.policy.obs_to_tensor(obs)[0]
            # NOTE:
            # obs_to_tensor() validates input against *observation_space*.
            # Passing an action vector here causes a shape mismatch on SB3
            # (expected obs shape=(16,), got action shape=(8,)).
            # Build action tensor manually for critic forward pass.
            act_t = torch.as_tensor(action, dtype=torch.float32, device=obs_t.device).reshape(1, -1)
            q1, q2 = model.critic(obs_t, act_t)
            nxt, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            norm = float(np.linalg.norm(action)); upper = float(np.linalg.norm(action[:4])); lower = float(np.linalg.norm(action[4:]))
            row = {"experiment_id": run_dir.name, "algorithm": "TD3", "evaluator": cfg.evaluator, "seed": cfg.seed, "episode_id": ep, "step_id": step, "global_step": global_step,
                   "AoA": cfg.aoa, "Re": cfg.re, "log10_Re": np.log10(cfg.re), "CL": info["CL"], "CD": info["CD"], "CM": info["CM"], "CL_CD": info["CL_CD"], "t_c": info["t_c"],
                   "CL_pred": info["CL"], "CD_pred": info["CD"], "CM_pred": info["CM"], "CL_CD_pred": info["CL_CD"],
                   "is_CM_feasible": info["is_CM_feasible"], "is_tc_feasible": info["is_tc_feasible"], "is_geometry_valid": info["is_geometry_valid"],
                   "reward_total": reward, "reward_objective_term": info["reward_objective_term"], "reward_CL_CD_term": info["reward_objective_term"], "reward_CM_penalty": info["reward_CM_penalty"], "reward_tc_penalty": info["reward_tc_penalty"],
                   "reward_local_thickness_penalty": 0.0, "reward_invalid_geometry_penalty": 0.0, "reward_solver_error_penalty": 0.0,
                   "penalty_total": info["reward_CM_penalty"] + info["reward_tc_penalty"],
                   "CM_lower_violation": info["CM_lower_violation"], "CM_upper_violation": info["CM_upper_violation"], "tc_lower_violation": info["tc_lower_violation"], "tc_upper_violation": info["tc_upper_violation"], "local_thickness_violation": 0.0,
                   "done": done, "truncated": truncated, "terminated": terminated, "done_reason": info["done_reason"],
                   "action_norm": norm, "upper_action_norm": upper, "lower_action_norm": lower, "action_max_abs": float(np.max(np.abs(action))), "action_saturation_count": int(np.sum(np.abs(action) >= 0.999))}
            for i in range(4):
                row[f"state_CST_u{i+1}"] = float(obs[i]); row[f"state_CST_l{i+1}"] = float(obs[4+i])
                row[f"action_u{i+1}"] = float(action[i]); row[f"action_l{i+1}"] = float(action[4+i])
                row[f"next_CST_u{i+1}"] = float(info["next_cst"][i]); row[f"next_CST_l{i+1}"] = float(info["next_cst"][4+i])
                row[f"delta_CST_u{i+1}"] = float(info["delta_cst"][i]); row[f"delta_CST_l{i+1}"] = float(info["delta_cst"][4+i])
            rollout.log(row)

            q1v = float(q1.detach().cpu().numpy().ravel()[0]); q2v = float(q2.detach().cpu().numpy().ravel()[0])
            policy.log({"episode_id": ep, "step_id": step, **{f"actor_action_u{i+1}": float(action[i]) for i in range(4)}, **{f"actor_action_l{i+1}": float(action[4+i]) for i in range(4)},
                        "Q1": q1v, "Q2": q2v, "Q_min": min(q1v, q2v), "Q_disagreement": abs(q1v-q2v), "target_Q": "", "td_error_Q1": "", "td_error_Q2": ""})

            total_reward += reward
            penalties += info["reward_CM_penalty"] + info["reward_tc_penalty"]
            action_norms.append(norm)
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
                          "invalid_geometry_count": int(not last_info['is_geometry_valid']), "solver_error_count": 0, "done_reason": last_info["done_reason"], "episode_length": step,
                          "is_final_feasible": bool(last_info['is_CM_feasible'] and last_info['is_tc_feasible'])})

    rollout.close(); policy.close()
    pd.DataFrame(summaries).to_csv(run_dir / "episode_summary.csv", index=False)

    # only current evaluator AoA sweep (no cross-validation)
    sweep_rows = []
    if best_record is not None:
        aoa_values = [float(x) for x in aoa_sweep.split(",") if x.strip()]
        cst = best_record["cst"]
        evaltor = _make_evaluator(cfg)
        for a in aoa_values:
            t0 = time.time()
            out = evaltor.evaluate(cst, a, cfg.re)
            runtime_ms = (time.time() - t0) * 1000.0
            ratio = out.cl / max(out.cd, cfg.cd_lower_bound)
            sweep_rows.append({
                "algorithm": "TD3", "evaluator": cfg.evaluator, "episode_id": best_record["episode_id"], "step_id": best_record["step_id"],
                "AoA": a, "Re": cfg.re,
                **{f"CST_u{i+1}": float(cst[i]) for i in range(4)}, **{f"CST_l{i+1}": float(cst[4+i]) for i in range(4)},
                "CL_pred": out.cl, "CD_pred": out.cd, "CM_pred": out.cm, "CL_CD_pred": ratio,
                "xfoil_converged": True if cfg.evaluator == "xfoil" else "", "xfoil_iterations": "", "xfoil_error_message": "", "runtime_ms": runtime_ms,
            })
    pd.DataFrame(sweep_rows).to_csv(run_dir / "xfoil_validation_logs.csv", index=False)

    with open(run_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump({"eval_wall_time_sec": time.time() - eval_start, "episodes": episodes, "aoa_sweep": aoa_sweep}, f, indent=2)
