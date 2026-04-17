import argparse
import csv
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from market_maker_env import MarketMakingEnv
from dqn_agent         import DQN, ReplayBuffer


# Hyperparameters
SEED = 42
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_CAPACITY = 10000
TARGET_UPDATE_FREQ = 200
NUM_EPISODES = 300
MAX_STEPS_PER_EPISODE = 400
LEARNING_STARTS = 2000
GRAD_CLIP_NORM = 10.0

# epsilon greedy exploration schedule
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 300_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    # keep random behavior consistent across runs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epsilon_by_step(step):
    # linearly decay epsilon from start to end so that the agent explores more as it learns
    frac = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_START + frac * (EPS_END - EPS_START)


def select_action(model, state, epsilon, action_dim):
    #initially just tries random actions as epsilon starts at 1.0 and then decays
    if random.random() < epsilon:
        return random.randrange(action_dim)

    #pytorch batch conversion
    state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    #feed the state into the neural network
    with torch.no_grad():
        q_values = model(state_tensor)
        #return action with the highest q-value
    return int(torch.argmax(q_values, dim=1).item())


def train_step(policy_net, target_net, buffer, optimizer, global_step):
    if len(buffer) < BATCH_SIZE or global_step < LEARNING_STARTS:
        return None

    # sample a random minibatch from replay memory
    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
    # 64 states, actions, rewards, etc and convert them to pytorch tensors
    #Convert batch arrays to tensors.
    states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    # Current Q-values
    current_q = policy_net(states).gather(1, actions)

    # Double-DQN target: online net selects action, target net evaluates it
    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
        next_q = target_net(next_states).gather(1, next_actions)
        target_q = rewards + GAMMA * next_q * (1 - dones)

    # Bellman regression loss
    loss = nn.MSELoss()(current_q, target_q)

    #actual learning 
    optimizer.zero_grad()
    #finds which weights caused the prediction to go wrong and updates them 
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP_NORM)
    optimizer.step()

    return loss.item()


def _load_env_config():
    """Load env config from results/best_config.json (shared with the Q-learning
    branch so both agents see the same environment). Falls back to inline
    defaults if the file is missing."""
    path = "results/best_config.json"
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        # Map best_config.json keys → env constructor keys
        env_config = {k: cfg[k] for k in cfg
                      if k not in ("env_alpha", "epsilon_decay", "learning_rate",
                                   "gamma", "epsilon_start", "epsilon_min")}
        env_config["alpha"] = cfg["env_alpha"]
        print(f"Loaded env config from {path}")
        return env_config
    # Fallback defaults (should match main's best_config.json)
    print(f"'{path}' not found; using inline defaults")
    return {
        "max_steps":                  500,
        "tick_size":                  0.05,
        "max_ticks":                  5,
        "max_inventory":              30,
        "stock_volatility":           0.0015,
        "stock_drift":                0.0,
        "adverse_selection_strength": 0.0025,
        "maker_fee":                  0.015,
        "alpha":                      0.005,
        "terminal_inventory_penalty": 1.0,
        "time_risk_start_frac":       0.25,
        "time_risk_k":                3.0,
        "vol_ewma_beta":              0.94,
        "buyer_arrival_rate":         1,
        "seller_arrival_rate":        1,
    }


def main(eval_only: bool = False):
    set_seed(SEED)
    env_config = _load_env_config()
    env = MarketMakingEnv(config=env_config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    

    policy_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net = DQN(state_dim, action_dim).to(DEVICE)
    # Start equal, then sync every TARGET_UPDATE_FREQ steps
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_CAPACITY)

    model_path = "dqn_market_maker.pt"
    if eval_only and os.path.exists(model_path):
        print(f"[--eval-only] Loading existing model from {model_path}, skipping training")
        policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        target_net.load_state_dict(policy_net.state_dict())
        _run_eval(policy_net, env, action_dim)
        return

    global_step = 0
    total_profit = 0.0

    for episode in range(NUM_EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0.0
        inventory_values = []
        equity_values = []
        buyer_fills = 0
        seller_fills = 0

        for step in range(env.max_steps):
            epsilon = epsilon_by_step(global_step)
            action = select_action(policy_net, state, epsilon, action_dim)

            next_state, reward, terminated, truncated, info = env.step(action)
            inventory_values.append(info["inventory"])
            equity_values.append(info["equity"])
            buyer_fills += info["filled_buys"]
            seller_fills += info["filled_sells"]
            done = terminated or truncated

            # Save transition for replay learning
            buffer.push(state, action, reward, next_state, done)

            train_step(policy_net, target_net, buffer, optimizer, global_step)

            state = next_state
            total_reward += reward
            global_step += 1

            if global_step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        final_inventory = inventory_values[-1] if inventory_values else 0
        episode_profit = equity_values[-1] if equity_values else 0.0
        total_profit += episode_profit

        if (episode + 1) % 50 == 0 or (episode + 1) == NUM_EPISODES:
            print(
                f"Episode {episode+1}/{NUM_EPISODES} | "
                f"Episode Profit: {episode_profit:.2f} | "
                f"Total Profit: {total_profit:.2f} | "
                f"Reward: {total_reward:.2f} | "
                f"Final Inv: {final_inventory} | "
                f"Buyer Fills: {buyer_fills} | "
                f"Seller Fills: {seller_fills}"
            )

    torch.save(policy_net.state_dict(), model_path)
    print(f"Training complete. Model saved to {model_path}")

    _run_eval(policy_net, env, action_dim)


def _run_eval(policy_net, env, action_dim, n_episodes: int = 1000):
    """Greedy-policy evaluation on seeds 0..n_episodes-1.

    Writes results/eval_dqn.csv with columns matching eval_qlearning.csv
    and eval_baseline.csv on the main branch: episode, reward, final_equity,
    final_inventory.
    """
    policy_net.eval()

    os.makedirs("results", exist_ok=True)
    csv_path = "results/eval_dqn.csv"

    rewards = []
    equities = []
    inventories = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "final_equity", "final_inventory"])

        for e in range(n_episodes):
            state, _ = env.reset(seed=e)
            done = False
            ep_reward = 0.0
            final_equity = 0.0
            final_inv = 0
            while not done:
                action = select_action(policy_net, state, epsilon=0.0, action_dim=action_dim)
                next_state, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                final_equity = info["equity"]
                final_inv = info["inventory"]
                state = next_state
                done = terminated or truncated

            rewards.append(ep_reward)
            equities.append(final_equity)
            inventories.append(final_inv)
            writer.writerow([e + 1, round(ep_reward, 4), round(final_equity, 4), final_inv])

    print(f"\nEval ({n_episodes} episodes, seeds 0..{n_episodes - 1}, greedy):")
    print(f"  mean_reward:              {float(np.mean(rewards)):.4f}")
    print(f"  std_reward:               {float(np.std(rewards)):.4f}")
    print(f"  mean_final_equity:        {float(np.mean(equities)):.4f}")
    print(f"  mean_abs_final_inventory: {float(np.mean(np.abs(inventories))):.4f}")
    print(f"  CSV saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training and just evaluate the saved model")
    args = parser.parse_args()
    main(eval_only=args.eval_only)