import numpy as np
from market_maker_env import MarketMakingEnv
from q_learning_agent import QLearningAgent

env = MarketMakingEnv(config={
    "max_steps":            500,
    "buyer_arrival_rate":   3,
    "seller_arrival_rate":  3,
    "alpha":                0.001,
})

agent = QLearningAgent(env, config={
    "learning_rate":  0.1,
    "gamma":          0.99,
    "epsilon_start":  1.0,
    "epsilon_min":    0.05,
    "epsilon_decay":  0.997,
})

print("=== Training ===")
rewards = agent.train(n_episodes=3000, log_every=200)
agent.save("q_table.pkl")

print("\n=== Evaluation (greedy, 200 episodes) ===")
stats = agent.evaluate(n_episodes=200)
for k, v in stats.items():
    print(f"  {k}: {v:.4f}")

try:
    import matplotlib.pyplot as plt

    window = 100
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 4))
    plt.plot(rewards, alpha=0.3, label="Episode reward")
    plt.plot(range(window - 1, len(rewards)), smoothed, label=f"{window}-ep moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Q-Learning — Market Maker")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    print("\nLearning curve saved to training_curve.png")
    plt.show()
except ImportError:
    pass
