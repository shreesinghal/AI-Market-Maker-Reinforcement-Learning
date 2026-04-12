import numpy as np
import matplotlib.pyplot as plt
from market_maker_env import MarketMakingEnv

env = MarketMakingEnv(config={
    "max_steps":                   500,
    "buyer_arrival_rate":          1,
    "seller_arrival_rate":         1,
    "alpha":                       0.01,
    "tick_size":                   0.05,
    "max_ticks":                   5,
    "stock_volatility":            0.0015,
    "adverse_selection_strength":  0.0025,
    "maker_fee":                   0.015,
    "terminal_inventory_penalty":  1.0,
})

n_episodes = 3000
episode_rewards = []

for ep in range(n_episodes):
    obs, _ = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    episode_rewards.append(total_reward)

    if (ep + 1) % 200 == 0:
        recent_avg = np.mean(episode_rewards[-200:])
        print(f"Episode {ep + 1}/{n_episodes} | Avg reward (last 200): {recent_avg:.4f}")

window = 100
smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")

plt.figure(figsize=(10, 4))
plt.plot(episode_rewards, alpha=0.3, label="Episode reward")
plt.plot(range(window - 1, len(episode_rewards)), smoothed, label=f"{window}-ep moving avg")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.title("Random Baseline — Market Maker")
plt.legend()
plt.tight_layout()
plt.savefig("baseline_curve.png", dpi=150)
print("\nBaseline curve saved to baseline_curve.png")
plt.show()
