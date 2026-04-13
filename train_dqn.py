"""
Final DQN training run using the best hyperparameters from tune.py.

Trains for 10,000 episodes, then evaluates both the DQN agent and
the random baseline over 1,000 episodes using the same seeds (fair comparison).

Output files (all in results/):
  dqn_model.pt                — trained DQN model
  dqn_training_curve.csv      — episode, reward, smoothed_reward
  dqn_training_curve.png      — learning curve plot
  eval_dqn.csv                — episode, reward, final_equity, final_inventory
  eval_dqn_baseline.csv       — episode, reward, final_equity, final_inventory
  dqn_comparison_summary.json — summary stats for comparison with Q-learning
"""

import csv
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from market_maker_env import MarketMakingEnv, DQN, ReplayBuffer

os.makedirs("results", exist_ok=True)

# ── Load best config ──────────────────────────────────────────────────────────
config_path = "results/best_config.json"
if not os.path.exists(config_path):
    print(f"'{config_path}' not found. Run tune.py first, or using defaults.")
    best_config = {
        "env_alpha":                  0.01,
        "epsilon_decay":              0.9995,
        "learning_rate":              1e-3,
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

# Separate env config from agent hyperparams
env_cfg = {k: best_config[k] for k in best_config
           if k not in ("env_alpha", "epsilon_decay", "learning_rate",
                        "gamma", "epsilon_start", "epsilon_min")}
env_cfg["alpha"] = best_config["env_alpha"]

# ── DQN hyperparameters ──────────────────────────────────────────────────────
GAMMA              = best_config.get("gamma", 0.99)
LR                 = 1e-3   # DQN learning rate (Adam); separate from Q-learning's tabular LR
EPS_START          = best_config.get("epsilon_start", 1.0)
EPS_END            = best_config.get("epsilon_min", 0.05)
EPS_DECAY          = best_config.get("epsilon_decay", 0.9995)
BATCH_SIZE         = 128
BUFFER_CAPACITY    = 50_000
TARGET_UPDATE_FREQ = 200
TRAIN_EVERY        = 4
GRAD_CLIP          = 1.0
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Train ─────────────────────────────────────────────────────────────────────
N_TRAIN = 2000
print(f"\n=== DQN Training for {N_TRAIN} episodes ===")

env = MarketMakingEnv(config=env_cfg)
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim).to(DEVICE)
target_net = DQN(state_dim, action_dim).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer    = ReplayBuffer(BUFFER_CAPACITY)

epsilon     = EPS_START
global_step = 0
rewards     = []

for episode in range(1, N_TRAIN + 1):
    state, _ = env.reset()
    total_reward = 0.0

    for _ in range(env_cfg["max_steps"]):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randrange(action_dim)
        else:
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                action = int(torch.argmax(policy_net(state_t), dim=1).item())

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, done)

        # Train every N steps once buffer is warm
        if global_step % TRAIN_EVERY == 0 and len(buffer) >= BATCH_SIZE:
            states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(BATCH_SIZE)
            states_b      = torch.tensor(states_b,      dtype=torch.float32, device=DEVICE)
            actions_b     = torch.tensor(actions_b,     dtype=torch.long,    device=DEVICE).unsqueeze(1)
            rewards_b     = torch.tensor(rewards_b,     dtype=torch.float32, device=DEVICE).unsqueeze(1)
            next_states_b = torch.tensor(next_states_b, dtype=torch.float32, device=DEVICE)
            dones_b       = torch.tensor(dones_b,       dtype=torch.float32, device=DEVICE).unsqueeze(1)

            current_q = policy_net(states_b).gather(1, actions_b)
            with torch.no_grad():
                best_actions = policy_net(next_states_b).argmax(dim=1, keepdim=True)
                max_next_q   = target_net(next_states_b).gather(1, best_actions)
                target_q     = rewards_b + GAMMA * max_next_q * (1 - dones_b)

            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP)
            optimizer.step()

        state        = next_state
        total_reward += reward
        global_step  += 1

        if global_step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    rewards.append(total_reward)

    if episode % 500 == 0:
        avg = np.mean(rewards[-500:])
        print(f"Episode {episode:>5}/{N_TRAIN} | "
              f"Avg reward (last 500): {avg:>8.4f} | "
              f"Epsilon: {epsilon:.4f}")

# Save model
model_path = "results/dqn_model.pt"
torch.save(policy_net.state_dict(), model_path)
print(f"DQN model saved to {model_path}")

# Save training curve CSV
window = 100
smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
smoothed_full = [None] * (window - 1) + list(smoothed)

curve_path = "results/dqn_training_curve.csv"
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
plt.title("DQN (Double DQN) — Market Maker (optimized)")
plt.legend()
plt.tight_layout()
plt.savefig("results/dqn_training_curve.png", dpi=150)
print("Training curve PNG saved to results/dqn_training_curve.png")
plt.close()

# ── Evaluate DQN and baseline on same seeds ──────────────────────────────────
N_EVAL = 1_000
seeds  = list(range(N_EVAL))
print(f"\n=== Evaluating DQN (greedy, {N_EVAL} episodes) ===")

eval_env = MarketMakingEnv(config=env_cfg)
dqn_rewards, dqn_equities, dqn_inventories = [], [], []

for seed in seeds:
    obs, _ = eval_env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        state_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = int(torch.argmax(policy_net(state_t), dim=1).item())
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated
    dqn_rewards.append(total_reward)
    dqn_equities.append(info.get("equity", 0.0))
    dqn_inventories.append(float(info.get("inventory", obs[0])))

print(f"  mean_reward:              {np.mean(dqn_rewards):.4f}")
print(f"  std_reward:               {np.std(dqn_rewards):.4f}")
print(f"  mean_equity:              {np.mean(dqn_equities):.4f}")
print(f"  mean_abs_final_inventory: {np.mean(np.abs(dqn_inventories)):.4f}")

dqn_csv_path = "results/eval_dqn.csv"
with open(dqn_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "reward", "final_equity", "final_inventory"])
    for i in range(N_EVAL):
        writer.writerow([
            i + 1,
            round(dqn_rewards[i], 4),
            round(dqn_equities[i], 4),
            round(dqn_inventories[i], 4),
        ])
print(f"DQN eval saved to {dqn_csv_path}")

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

bl_csv_path = "results/eval_dqn_baseline.csv"
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

# ── Win rate (DQN reward > baseline reward, same seed) ───────────────────────
wins = sum(1 for q, b in zip(dqn_rewards, bl_rewards) if q > b)
win_rate = wins / N_EVAL

# ── Comparison summary ───────────────────────────────────────────────────────
summary = {
    "dqn": {
        "mean_reward":              round(float(np.mean(dqn_rewards)), 4),
        "std_reward":               round(float(np.std(dqn_rewards)), 4),
        "mean_final_equity":        round(float(np.mean(dqn_equities)), 4),
        "mean_abs_final_inventory": round(float(np.mean(np.abs(dqn_inventories))), 4),
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

summary_path = "results/dqn_comparison_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nComparison summary saved to {summary_path}")
print(f"Win rate (DQN > baseline): {win_rate:.1%}")
