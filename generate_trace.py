"""
Generate step-by-step episode traces for the demo GUI.

Loads the trained Q-table and runs one greedy episode (seed=42),
then runs one random baseline episode with the same seed so the GUI
can show both agents on the same price path.

Output files:
  results/simulation_trace.json  — Q-learning agent trace
  results/baseline_trace.json    — random baseline trace
"""

import json
import os
import pickle
import numpy as np
from market_maker_env import MarketMakingEnv
from q_learning_agent import QLearningAgent

os.makedirs("results", exist_ok=True)

SEED = 42

# ── Load config ───────────────────────────────────────────────────────────────
config_path = "results/best_config.json"
if os.path.exists(config_path):
    with open(config_path) as f:
        best_config = json.load(f)
else:
    print(f"'{config_path}' not found. Using defaults.")
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

env_cfg = {k: best_config[k] for k in best_config
           if k not in ("env_alpha", "epsilon_decay", "learning_rate",
                        "gamma", "epsilon_start", "epsilon_min")}
env_cfg["alpha"] = best_config["env_alpha"]

trace_config = {
    "initial_price": best_config.get("initial_price", 100.0),
    "max_steps":     best_config["max_steps"],
    "tick_size":     best_config.get("tick_size", 0.05),
    "seed":          SEED,
}


def run_episode(env, action_fn):
    """Run one episode, return list of per-step dicts."""
    obs, _ = env.reset(seed=SEED)
    steps = []
    cumulative_reward = 0.0
    done = False

    while not done:
        action = action_fn(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward

        step_data = {
            "step":              env.current_step - 1,
            "mid_price":         round(float(info["mid_price"]), 6),
            "bid_price":         round(float(info["bid_price"]), 6),
            "ask_price":         round(float(info["ask_price"]), 6),
            "bid_offset":        int(info["bid_offset"]),
            "ask_offset":        int(info["ask_offset"]),
            "n_buyers":          int(info["n_buyers"]),
            "n_sellers":         int(info["n_sellers"]),
            "trades_bought":     int(info["filled_sells"]),   # agent bought from sellers
            "trades_sold":       int(info["filled_buys"]),    # agent sold to buyers
            "inventory":         int(info["inventory"]),
            "cash":              round(float(info["cash"]), 4),
            "equity":            round(float(info["equity"]), 4),
            "reward":            round(float(reward), 6),
            "cumulative_reward": round(float(cumulative_reward), 4),
        }
        steps.append(step_data)

        obs = next_obs
        done = terminated or truncated

    return steps


# ── Q-learning trace ──────────────────────────────────────────────────────────
qtable_path = "results/q_table.pkl"
if not os.path.exists(qtable_path):
    raise FileNotFoundError(
        f"'{qtable_path}' not found. Run train_final.py first."
    )

env = MarketMakingEnv(config=env_cfg)
agent = QLearningAgent(env, config={
    "learning_rate":  best_config["learning_rate"],
    "gamma":          best_config["gamma"],
    "epsilon_start":  0.0,   # greedy
    "epsilon_min":    0.0,
    "epsilon_decay":  1.0,
})

with open(qtable_path, "rb") as f:
    data = pickle.load(f)
agent.q_table = data["q_table"]
agent.epsilon = 0.0

def ql_action(obs):
    state = agent.discretize(obs)
    return int(np.argmax(agent.q_table[state]))

print(f"Running Q-learning episode (seed={SEED}) ...")
ql_steps = run_episode(env, ql_action)
print(f"  {len(ql_steps)} steps, final equity={ql_steps[-1]['equity']:.2f}, "
      f"final inventory={ql_steps[-1]['inventory']}")

ql_trace = {"config": trace_config, "agent": "qlearning", "steps": ql_steps}
ql_path = "results/simulation_trace.json"
with open(ql_path, "w") as f:
    json.dump(ql_trace, f)
print(f"Q-learning trace saved to {ql_path}")

# ── Baseline trace ────────────────────────────────────────────────────────────
baseline_env = MarketMakingEnv(config=env_cfg)

def random_action(_obs):
    return int(baseline_env.action_space.sample())

print(f"\nRunning random baseline episode (seed={SEED}) ...")
bl_steps = run_episode(baseline_env, random_action)
print(f"  {len(bl_steps)} steps, final equity={bl_steps[-1]['equity']:.2f}, "
      f"final inventory={bl_steps[-1]['inventory']}")

bl_trace = {"config": trace_config, "agent": "baseline", "steps": bl_steps}
bl_path = "results/baseline_trace.json"
with open(bl_path, "w") as f:
    json.dump(bl_trace, f)
print(f"Baseline trace saved to {bl_path}")
