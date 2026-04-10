import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from market_maker_env import MarketMakingEnv, DQN, ReplayBuffer


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


def main():
    set_seed(SEED)
    env_config = {
        "max_steps": MAX_STEPS_PER_EPISODE,
        "tick_size": 0.02,
        "max_ticks": 6,
        "max_inventory": 30,
        "stock_volatility": 0.0015,
        "stock_drift": 0.0,
        "adverse_selection_strength": 0.0025,
        "maker_fee": 0.015,
        "alpha": 0.02,
        "terminal_inventory_penalty": 4.0,
        "time_risk_start_frac": 0.25,
        "time_risk_k": 3.5,
        "vol_ewma_beta": 0.94,
        "buyer_arrival_rate": 1.0,
        "seller_arrival_rate": 1.0,
    }
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

    global_step = 0
    total_profit = 0.0

    for episode in range(NUM_EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0.0
        inventory_values = []
        equity_values = []
        buyer_fills = 0
        seller_fills = 0

        for step in range(MAX_STEPS_PER_EPISODE):
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

    torch.save(policy_net.state_dict(), "dqn_market_maker.pt")
    print("Training complete. Model saved to dqn_market_maker.pt")

    # Greedy-policy evaluation
    eval_episodes = 50
    eval_final_profits = []
    eval_final_invs = []
    eval_rewards = []

    policy_net.eval()
    for e in range(eval_episodes):
        state, _ = env.reset(seed=10_000 + e)
        done = False
        ep_reward = 0.0
        final_profit = 0.0
        final_inv = 0
        while not done:
            action = select_action(policy_net, state, epsilon=0.0, action_dim=action_dim)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            final_profit = info["equity"]
            final_inv = info["inventory"]
            state = next_state
            done = terminated or truncated
        eval_rewards.append(ep_reward)
        eval_final_profits.append(final_profit)
        eval_final_invs.append(final_inv)

    avg_eval_profit = float(np.mean(eval_final_profits))
    avg_eval_abs_inv = float(np.mean(np.abs(eval_final_invs)))
    avg_eval_reward = float(np.mean(eval_rewards))
    print(
        f"Eval (epsilon=0): "
        f"Avg Reward: {avg_eval_reward:.2f} | "
        f"Avg Final Profit: {avg_eval_profit:.2f} | "
        f"Avg |Final Inv|: {avg_eval_abs_inv:.2f}"
    )


if __name__ == "__main__":
    main()