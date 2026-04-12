import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from market_maker_env import MarketMakingEnv, DQN, ReplayBuffer

# -------------------------
# Hyperparameters
# -------------------------
GAMMA             = 0.99
LR                = 1e-3
BATCH_SIZE        = 128
BUFFER_CAPACITY   = 50_000
TARGET_UPDATE_FREQ = 200          # steps between target-net syncs
NUM_EPISODES      = 3000
MAX_STEPS         = 500

EPS_START = 1.0
EPS_END   = 0.05
EPS_DECAY = 0.997                 # per-episode decay (matches Q-learning schedule)

GRAD_CLIP = 1.0                   # gradient clipping norm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(model, state, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        return int(torch.argmax(model(state_t), dim=1).item())


def train_step(policy_net, target_net, buffer, optimizer):
    if len(buffer) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

    states      = torch.tensor(states,      dtype=torch.float32, device=DEVICE)
    actions     = torch.tensor(actions,     dtype=torch.long,    device=DEVICE).unsqueeze(1)
    rewards     = torch.tensor(rewards,     dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
    dones       = torch.tensor(dones,       dtype=torch.float32, device=DEVICE).unsqueeze(1)

    current_q = policy_net(states).gather(1, actions)

    # Double DQN: policy_net picks the action, target_net evaluates it
    with torch.no_grad():
        best_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
        max_next_q   = target_net(next_states).gather(1, best_actions)
        target_q     = rewards + GAMMA * max_next_q * (1 - dones)

    loss = nn.MSELoss()(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP)
    optimizer.step()

    return loss.item()


def main():
    env = MarketMakingEnv(config={
        "max_steps":           MAX_STEPS,
        "buyer_arrival_rate":  3,
        "seller_arrival_rate": 3,
        "alpha":               0.01,
    })

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
    all_rewards = []

    print("=== DQN Training ===")
    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            action = select_action(policy_net, state, epsilon, action_dim)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            train_step(policy_net, target_net, buffer, optimizer)

            state        = next_state
            total_reward += reward
            global_step  += 1

            if global_step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        all_rewards.append(total_reward)

        if episode % 200 == 0:
            avg = np.mean(all_rewards[-200:])
            print(f"Episode {episode:>5}/{NUM_EPISODES} | "
                  f"Avg reward (last 200): {avg:>8.4f} | "
                  f"Epsilon: {epsilon:.4f}")

    torch.save(policy_net.state_dict(), "dqn_market_maker.pt")
    print("Model saved to dqn_market_maker.pt")

    # Greedy evaluation
    print("\n=== Evaluation (greedy, 200 episodes) ===")
    rewards, equities = [], []
    for _ in range(200):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = select_action(policy_net, state, 0.0, action_dim)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        equities.append(info.get("equity", 0.0))

    print(f"  mean_reward: {np.mean(rewards):.4f}")
    print(f"  std_reward:  {np.std(rewards):.4f}")
    print(f"  mean_equity: {np.mean(equities):.4f}")
    print(f"  std_equity:  {np.std(equities):.4f}")

    try:
        import matplotlib.pyplot as plt

        window   = 100
        smoothed = np.convolve(all_rewards, np.ones(window) / window, mode="valid")

        plt.figure(figsize=(10, 4))
        plt.plot(all_rewards, alpha=0.3, label="Episode reward")
        plt.plot(range(window - 1, len(all_rewards)), smoothed,
                 label=f"{window}-ep moving avg")
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.title("DQN (Double DQN) — Market Maker")
        plt.legend()
        plt.tight_layout()
        plt.savefig("dqn_curve.png", dpi=150)
        print("\nLearning curve saved to dqn_curve.png")
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
