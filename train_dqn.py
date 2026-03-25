import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from market_maker_env import MarketMakingEnv, DQN, ReplayBuffer


# -------------------------
# Hyperparameters
# -------------------------
GAMMA = .99
LR = 1e-3
#64 past transitions
BATCH_SIZE = 64
BUFFER_CAPACITY = 10000
TARGET_UPDATE_FREQ = 100
NUM_EPISODES = 300
MAX_STEPS_PER_EPISODE = 1000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def train_step(policy_net, target_net, buffer, optimizer):
    if len(buffer) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
    # 64 states, actions, rewards, etc and convert them to pytorch tensors
    states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
    actions = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    # Current Q-values
    current_q = policy_net(states).gather(1, actions)

    # Target Q-values
    with torch.no_grad():
        max_next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
        #bellamn 
        target_q = rewards + GAMMA * max_next_q * (1 - dones)

    #compare the current policy predicts with what the bellman equation says it should predict
    loss = nn.MSELoss()(current_q, target_q)

    #actual learning 
    optimizer.zero_grad()
    #finds which weights caused the prediction to go wrong and updates them 
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    env = MarketMakingEnv(config={"max_steps": MAX_STEPS_PER_EPISODE})

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net = DQN(state_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_CAPACITY)

    epsilon = EPS_START
    global_step = 0

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0.0
        losses = []

        for step in range(MAX_STEPS_PER_EPISODE):
            action = select_action(policy_net, state, epsilon, action_dim)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)

            loss = train_step(policy_net, target_net, buffer, optimizer)
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward
            global_step += 1

            if global_step % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        avg_loss = np.mean(losses) if losses else 0.0
        print(
            f"Episode {episode+1}/{NUM_EPISODES} | "
            f"Reward: {total_reward:.2f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Avg Loss: {avg_loss:.4f}"
        )

    torch.save(policy_net.state_dict(), "dqn_market_maker.pt")
    print("Training complete. Model saved to dqn_market_maker.pt")


if __name__ == "__main__":
    main()