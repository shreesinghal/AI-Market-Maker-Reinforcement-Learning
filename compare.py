"""
compare.py — Train Q-Learning and DQN side-by-side, then plot results.

Usage:
    python compare.py
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from market_maker_env import MarketMakingEnv, DQN, ReplayBuffer
from q_learning_agent import QLearningAgent

# ── Shared config ─────────────────────────────────────────────────────────────
ENV_CONFIG = {
    "max_steps":           500,
    "buyer_arrival_rate":  3,
    "seller_arrival_rate": 3,
    "alpha":               0.01,
}
N_EPISODES  = 3000
EVAL_EPS    = 200
LOG_EVERY   = 200
SMOOTH_WIN  = 100

# ── DQN config ────────────────────────────────────────────────────────────────
GAMMA              = 0.99
LR                 = 1e-3
BATCH_SIZE         = 128
BUFFER_CAPACITY    = 50_000
TARGET_UPDATE_FREQ = 200
EPS_START          = 1.0
EPS_END            = 0.05
EPS_DECAY          = 0.997
GRAD_CLIP          = 1.0
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── DQN helpers ───────────────────────────────────────────────────────────────

def dqn_select_action(model, state, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return int(torch.argmax(model(s), dim=1).item())


def dqn_train_step(policy_net, target_net, buffer, optimizer):
    if len(buffer) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
    states      = torch.tensor(states,      dtype=torch.float32, device=DEVICE)
    actions     = torch.tensor(actions,     dtype=torch.long,    device=DEVICE).unsqueeze(1)
    rewards     = torch.tensor(rewards,     dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
    dones       = torch.tensor(dones,       dtype=torch.float32, device=DEVICE).unsqueeze(1)

    current_q = policy_net(states).gather(1, actions)
    with torch.no_grad():
        best_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
        max_next_q   = target_net(next_states).gather(1, best_actions)
        target_q     = rewards + GAMMA * max_next_q * (1 - dones)

    loss = nn.MSELoss()(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP)
    optimizer.step()


# ── Training routines ─────────────────────────────────────────────────────────

def train_qlearning():
    print("=" * 50)
    print("  Q-Learning Training")
    print("=" * 50)
    env   = MarketMakingEnv(config=ENV_CONFIG)
    agent = QLearningAgent(env, config={
        "learning_rate":  0.1,
        "gamma":          GAMMA,
        "epsilon_start":  EPS_START,
        "epsilon_min":    EPS_END,
        "epsilon_decay":  EPS_DECAY,
    })
    rewards = agent.train(n_episodes=N_EPISODES, log_every=LOG_EVERY)
    agent.save("q_table.pkl")
    stats = agent.evaluate(n_episodes=EVAL_EPS)
    print("\nQ-Learning eval:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    return rewards, stats


def train_dqn():
    print("\n" + "=" * 50)
    print("  DQN (Double DQN) Training")
    print("=" * 50)
    env        = MarketMakingEnv(config=ENV_CONFIG)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer   = optim.Adam(policy_net.parameters(), lr=LR)
    buffer      = ReplayBuffer(BUFFER_CAPACITY)
    epsilon     = EPS_START
    global_step = 0
    all_rewards = []

    for episode in range(1, N_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(ENV_CONFIG["max_steps"]):
            action = dqn_select_action(policy_net, state, epsilon, action_dim)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            dqn_train_step(policy_net, target_net, buffer, optimizer)
            state        = next_state
            total_reward += reward
            global_step  += 1
            if global_step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if done:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        all_rewards.append(total_reward)

        if episode % LOG_EVERY == 0:
            avg = np.mean(all_rewards[-LOG_EVERY:])
            print(f"Episode {episode:>5}/{N_EPISODES} | "
                  f"Avg reward (last {LOG_EVERY}): {avg:>8.4f} | "
                  f"Epsilon: {epsilon:.4f}")

    torch.save(policy_net.state_dict(), "dqn_market_maker.pt")

    # Greedy evaluation
    rewards_eval, equities = [], []
    for _ in range(EVAL_EPS):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = dqn_select_action(policy_net, state, 0.0, action_dim)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards_eval.append(total_reward)
        equities.append(info.get("equity", 0.0))

    stats = {
        "mean_reward": np.mean(rewards_eval),
        "std_reward":  np.std(rewards_eval),
        "mean_equity": np.mean(equities),
        "std_equity":  np.std(equities),
    }
    print("\nDQN eval:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    return all_rewards, stats


# ── Comparison plot ───────────────────────────────────────────────────────────

def plot_comparison(ql_rewards, dqn_rewards):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, rewards, label, color in zip(
        axes,
        [ql_rewards, dqn_rewards],
        ["Q-Learning", "DQN (Double)"],
        ["steelblue", "darkorange"],
    ):
        smoothed = np.convolve(rewards, np.ones(SMOOTH_WIN) / SMOOTH_WIN, mode="valid")
        ax.plot(rewards, alpha=0.2, color=color)
        ax.plot(range(SMOOTH_WIN - 1, len(rewards)), smoothed,
                color=color, label=f"{SMOOTH_WIN}-ep avg")
        ax.set_title(label)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.legend()

    plt.suptitle("Market Maker — Q-Learning vs DQN", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    print("\nComparison plot saved to comparison.png")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ql_rewards,  ql_stats  = train_qlearning()
    dqn_rewards, dqn_stats = train_dqn()

    print("\n" + "=" * 50)
    print("  Final Comparison (greedy, 200 episodes)")
    print("=" * 50)
    print(f"{'Metric':<20} {'Q-Learning':>12} {'DQN':>12}")
    print("-" * 46)
    for k in ql_stats:
        print(f"{k:<20} {ql_stats[k]:>12.4f} {dqn_stats[k]:>12.4f}")

    try:
        plot_comparison(ql_rewards, dqn_rewards)
    except ImportError:
        print("Install matplotlib to see the comparison plot.")
