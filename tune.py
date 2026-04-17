"""
Sequential hyperparameter tuning for the Q-learning market maker agent.

Tunes in three phases (one param at a time, carrying the best forward):
  1. epsilon_decay
  2. env_alpha (inventory penalty)
  3. learning_rate

Each candidate is trained for 5,000 episodes and evaluated greedily over
500 episodes. Results are logged to results/tuning_log.csv and the best
config is saved to results/best_config.json.
"""

import csv
import json
import os
import numpy as np
from market_maker_env import MarketMakingEnv
from q_learning_agent import QLearningAgent

os.makedirs("results", exist_ok=True)

# Fixed env params throughout tuning (DQN env config)
FIXED_ENV = {
    "max_steps":                   500,
    "buyer_arrival_rate":          1,
    "seller_arrival_rate":         1,
    "tick_size":                   0.05,
    "max_ticks":                   5,
    "max_inventory":               30,
    "initial_price":               100.0,
    "stock_volatility":            0.0015,
    "adverse_selection_strength":  0.0025,
    "maker_fee":                   0.015,
    "terminal_inventory_penalty":  1.0,
    "time_risk_start_frac":        0.25,
    "time_risk_k":                 3.0,
    "vol_ewma_beta":               0.94,
}
N_TRAIN = 5_000
N_EVAL  = 500

log_rows = []


def run_trial(env_alpha, epsilon_decay, learning_rate):
    env = MarketMakingEnv(config={**FIXED_ENV, "alpha": env_alpha})
    agent = QLearningAgent(env, config={
        "learning_rate":  learning_rate,
        "gamma":          0.99,
        "epsilon_start":  1.0,
        "epsilon_min":    0.05,
        "epsilon_decay":  epsilon_decay,
    })
    agent.train(n_episodes=N_TRAIN, log_every=N_TRAIN + 1)  # silent
    stats = agent.evaluate(n_episodes=N_EVAL)
    return stats["mean_reward"], stats["mean_abs_final_inventory"]


print("=" * 60)
print("Phase 1 — Tuning epsilon_decay")
print("=" * 60)

# Baseline values for other params
best_env_alpha   = 0.01   # DQN env default
best_lr          = 0.1
best_eps_decay   = None
best_mean_reward = -np.inf

for ed in [0.9995, 0.9990, 0.9985]:
    print(f"  epsilon_decay={ed} ...", flush=True)
    mean_r, mean_inv = run_trial(best_env_alpha, ed, best_lr)
    print(f"    mean_reward={mean_r:.2f}  mean_abs_inv={mean_inv:.2f}")
    log_rows.append({
        "phase": 1, "param": "epsilon_decay", "value": ed,
        "env_alpha": best_env_alpha, "learning_rate": best_lr,
        "mean_reward": mean_r, "mean_abs_inventory": mean_inv,
    })
    if mean_r > best_mean_reward:
        best_mean_reward = mean_r
        best_eps_decay   = ed

print(f"  Best epsilon_decay: {best_eps_decay}  (mean_reward={best_mean_reward:.2f})\n")

# ── Phase 2: env_alpha ───────────────────────────────────────────────────────
print("=" * 60)
print("Phase 2 — Tuning env_alpha (inventory penalty)")
print("=" * 60)

best_mean_reward = -np.inf
best_env_alpha_final = None

for ea in [0.005, 0.01, 0.02]:
    print(f"  env_alpha={ea} ...", flush=True)
    mean_r, mean_inv = run_trial(ea, best_eps_decay, best_lr)
    print(f"    mean_reward={mean_r:.2f}  mean_abs_inv={mean_inv:.2f}")
    log_rows.append({
        "phase": 2, "param": "env_alpha", "value": ea,
        "env_alpha": ea, "learning_rate": best_lr,
        "mean_reward": mean_r, "mean_abs_inventory": mean_inv,
    })
    if mean_r > best_mean_reward:
        best_mean_reward     = mean_r
        best_env_alpha_final = ea

best_env_alpha = best_env_alpha_final
print(f"  Best env_alpha: {best_env_alpha}  (mean_reward={best_mean_reward:.2f})\n")

# ── Phase 3: learning_rate ───────────────────────────────────────────────────
print("=" * 60)
print("Phase 3 — Tuning learning_rate")
print("=" * 60)

best_mean_reward = -np.inf
best_lr_final    = None

for lr in [0.1, 0.05, 0.01]:
    print(f"  learning_rate={lr} ...", flush=True)
    mean_r, mean_inv = run_trial(best_env_alpha, best_eps_decay, lr)
    print(f"    mean_reward={mean_r:.2f}  mean_abs_inv={mean_inv:.2f}")
    log_rows.append({
        "phase": 3, "param": "learning_rate", "value": lr,
        "env_alpha": best_env_alpha, "learning_rate": lr,
        "mean_reward": mean_r, "mean_abs_inventory": mean_inv,
    })
    if mean_r > best_mean_reward:
        best_mean_reward = mean_r
        best_lr_final    = lr

best_lr = best_lr_final
print(f"  Best learning_rate: {best_lr}  (mean_reward={best_mean_reward:.2f})\n")

# ── Save results ─────────────────────────────────────────────────────────────
csv_path = "results/tuning_log.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
    writer.writeheader()
    writer.writerows(log_rows)
print(f"Tuning log saved to {csv_path}")

best_config = {
    "env_alpha":                  best_env_alpha,
    "epsilon_decay":              best_eps_decay,
    "learning_rate":              best_lr,
    "gamma":                      0.99,
    "epsilon_start":              1.0,
    "epsilon_min":                0.05,
    "max_steps":                  500,
    "buyer_arrival_rate":         1,
    "seller_arrival_rate":        1,
    "tick_size":                  0.05,
    "max_ticks":                  5,
    "max_inventory":              30,
    "initial_price":              100.0,
    "stock_volatility":           0.0015,
    "adverse_selection_strength": 0.0025,
    "maker_fee":                  0.015,
    "terminal_inventory_penalty": 1.0,
    "time_risk_start_frac":       0.25,
    "time_risk_k":                3.0,
    "vol_ewma_beta":              0.94,
}
config_path = "results/best_config.json"
with open(config_path, "w") as f:
    json.dump(best_config, f, indent=2)
print(f"Best config saved to {config_path}")
print(f"\nBest config: {best_config}")
