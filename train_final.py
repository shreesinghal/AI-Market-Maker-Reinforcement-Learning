"""
Final training run using the best hyperparameters from tune.py.

Trains for 10,000 episodes, then evaluates both the Q-learning agent and
the random baseline over 1,000 episodes using the same seeds (fair comparison).

Output files (all in results/):
  q_table.pkl             — trained Q-table
  training_curve.csv      — episode, reward, smoothed_reward
  eval_qlearning.csv      — episode, reward, final_equity, final_inventory
  eval_baseline.csv       — episode, reward, final_equity, final_inventory
  comparison_summary.json — summary stats for the GUI
"""

import csv
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from market_maker_env import MarketMakingEnv
from q_learning_agent import QLearningAgent

os.makedirs("results", exist_ok=True)

# ── Load best config ─────────────────────────────────────────────────────────
config_path = "results/best_config.json"
if not os.path.exists(config_path):
    print(f"'{config_path}' not found. Run tune.py first, or using defaults.")
    best_config = {
        "env_alpha":                  0.01,
        "epsilon_decay":              0.9995,
        "learning_rate":              0.1,
        "gamma":                      0.99,
        "epsilon_start":              1.0,
        "epsilon_min":                0.05,
        "max_steps":                  500,
        "buyer_arrival_rate":         1,
        "seller_arrival_rate":        1,
        "tick_size":                  0.05,
        "max_ticks":                  5,
        "max_inventory":              100,
        "initial_price":              100.0,
        "stock_volatility":           0.0015,
        "adverse_selection_strength": 0.0025,
        "maker_fee":                  0.015,
        "terminal_inventory_penalty": 1.0,
        "time_risk_start_frac":       0.25,
        "time_risk_k":                3.0,
        "vol_ewma_beta":              0.94,
    }
else:
    with open(config_path) as f:
        best_config = json.load(f)
    print(f"Loaded best config: {best_config}")

env_cfg = {k: best_config[k] for k in best_config
           if k not in ("env_alpha", "epsilon_decay", "learning_rate",
                        "gamma", "epsilon_start", "epsilon_min")}
env_cfg["alpha"] = best_config["env_alpha"]

# ── Train ─────────────────────────────────────────────────────────────────────
N_TRAIN = 10_000
print(f"\n=== Training for {N_TRAIN} episodes ===")

env = MarketMakingEnv(config=env_cfg)
agent = QLearningAgent(env, config={
    "learning_rate":  best_config["learning_rate"],
    "gamma":          best_config["gamma"],
    "epsilon_start":  best_config["epsilon_start"],
    "epsilon_min":    best_config["epsilon_min"],
    "epsilon_decay":  best_config["epsilon_decay"],
})

rewards = agent.train(n_episodes=N_TRAIN, log_every=500)

# Save Q-table
qtable_path = "results/q_table.pkl"
with open(qtable_path, "wb") as f:
    pickle.dump({"q_table": agent.q_table, "epsilon": agent.epsilon}, f)
print(f"Q-table saved to {qtable_path}")

# Save training curve CSV
window = 100
smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
smoothed_full = [None] * (window - 1) + list(smoothed)

curve_path = "results/training_curve.csv"
with open(curve_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "reward", "smoothed_reward"])
    for i, r in enumerate(rewards):
        s = smoothed_full[i]
        writer.writerow([i + 1, round(r, 4), round(s, 4) if s is not None else ""])
print(f"Training curve saved to {curve_path}")

# Save training curve PNG
plt.figure(figsize=(10, 4))
plt.plot(rewards, alpha=0.3, label="Episode reward")
plt.plot(range(window - 1, len(rewards)), smoothed, label=f"{window}-ep moving avg")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Q-Learning — Market Maker (optimized)")
plt.legend()
plt.tight_layout()
plt.savefig("results/training_curve.png", dpi=150)
print("Training curve PNG saved to results/training_curve.png")
plt.close()

# ── Evaluate Q-learning and baseline on same seeds ────────────────────────────
N_EVAL = 1_000
seeds  = list(range(N_EVAL))
print(f"\n=== Evaluating Q-learning (greedy, {N_EVAL} episodes) ===")

ql_stats = agent.evaluate(n_episodes=N_EVAL, seeds=seeds)
print(f"  mean_reward:              {ql_stats['mean_reward']:.4f}")
print(f"  std_reward:               {ql_stats['std_reward']:.4f}")
print(f"  mean_equity:              {ql_stats['mean_equity']:.4f}")
print(f"  mean_abs_final_inventory: {ql_stats['mean_abs_final_inventory']:.4f}")

ql_csv_path = "results/eval_qlearning.csv"
with open(ql_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "reward", "final_equity", "final_inventory"])
    for i in range(N_EVAL):
        writer.writerow([
            i + 1,
            round(ql_stats["rewards"][i], 4),
            round(ql_stats["equities"][i], 4),
            round(ql_stats["inventories"][i], 4),
        ])
print(f"Q-learning eval saved to {ql_csv_path}")

print(f"\n=== Evaluating random baseline ({N_EVAL} episodes) ===")
baseline_env = MarketMakingEnv(config=env_cfg)
bl_rewards, bl_equities, bl_inventories = [], [], []

for seed in seeds:
    obs, _ = baseline_env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        action = baseline_env.action_space.sample()
        obs, reward, terminated, truncated, info = baseline_env.step(action)
        total_reward += reward
        done = terminated or truncated
    bl_rewards.append(total_reward)
    bl_equities.append(info.get("equity", 0.0))
    bl_inventories.append(float(info.get("inventory", obs[0])))

print(f"  mean_reward:              {np.mean(bl_rewards):.4f}")
print(f"  std_reward:               {np.std(bl_rewards):.4f}")
print(f"  mean_equity:              {np.mean(bl_equities):.4f}")
print(f"  mean_abs_final_inventory: {np.mean(np.abs(bl_inventories)):.4f}")

bl_csv_path = "results/eval_baseline.csv"
with open(bl_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "reward", "final_equity", "final_inventory"])
    for i in range(N_EVAL):
        writer.writerow([
            i + 1,
            round(bl_rewards[i], 4),
            round(bl_equities[i], 4),
            round(bl_inventories[i], 4),
        ])
print(f"Baseline eval saved to {bl_csv_path}")

# ── Win rate (Q-learning reward > baseline reward, same seed) ─────────────────
wins = sum(1 for q, b in zip(ql_stats["rewards"], bl_rewards) if q > b)
win_rate = wins / N_EVAL

# ── Comparison summary ────────────────────────────────────────────────────────
summary = {
    "qlearning": {
        "mean_reward":              round(ql_stats["mean_reward"], 4),
        "std_reward":               round(ql_stats["std_reward"], 4),
        "mean_final_equity":        round(ql_stats["mean_equity"], 4),
        "mean_abs_final_inventory": round(ql_stats["mean_abs_final_inventory"], 4),
        "num_eval_episodes":        N_EVAL,
    },
    "baseline": {
        "mean_reward":              round(float(np.mean(bl_rewards)), 4),
        "std_reward":               round(float(np.std(bl_rewards)), 4),
        "mean_final_equity":        round(float(np.mean(bl_equities)), 4),
        "mean_abs_final_inventory": round(float(np.mean(np.abs(bl_inventories))), 4),
        "num_eval_episodes":        N_EVAL,
    },
    "win_rate": round(win_rate, 4),
    "seed":     0,
}

summary_path = "results/comparison_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nComparison summary saved to {summary_path}")
print(f"Win rate (Q-learning > baseline): {win_rate:.1%}")
