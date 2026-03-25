import numpy as np
import pickle
from market_maker_env import MarketMakingEnv


class QLearningAgent:
    """
    Tabular Q-learning agent for the MarketMakingEnv.

    The continuous observation space is discretized into bins so we can
    maintain a finite Q-table.

    State dimensions
    ----------------
    inventory     : -max_inventory … +max_inventory  → inv_bins
    price_ratio   : mid_price / initial_price         → price_bins
    volatility    : stock volatility                  → vol_bins
    time_remaining: 0 … 1                             → time_bins

    Action space
    ------------
    MultiDiscrete([max_ticks, max_ticks]) is flattened to a single integer
    index so the Q-table stays 2-D (state × action).
    """

    def __init__(self, env: MarketMakingEnv, config: dict = None):
        cfg = config or {}
        self.env = env

        # ── Hyperparameters ──────────────────────────────────────────────
        self.alpha         = cfg.get("learning_rate",  0.1)
        self.gamma         = cfg.get("gamma",          0.99)
        self.epsilon       = cfg.get("epsilon_start",  1.0)
        self.epsilon_min   = cfg.get("epsilon_min",    0.05)
        self.epsilon_decay = cfg.get("epsilon_decay",  0.995)

        # ── Discretisation edges ─────────────────────────────────────────
        self.inv_bins   = np.linspace(-env.max_inventory, env.max_inventory, 20)
        self.price_bins = np.array([0.7, 0.85, 0.93, 0.97, 1.0,
                                    1.03, 1.07, 1.15, 1.30])   # ratios vs initial
        self.vol_bins   = np.array([0.005, 0.01, 0.02, 0.04, 0.08])
        self.time_bins  = np.linspace(0.0, 1.0, 10)

        # Number of buckets in each dimension (digitize returns 1-based indices)
        n_inv   = len(self.inv_bins)   + 1
        n_price = len(self.price_bins) + 1
        n_vol   = len(self.vol_bins)   + 1
        n_time  = len(self.time_bins)  + 1

        # ── Action space ─────────────────────────────────────────────────
        self.n1 = int(env.action_space.nvec[0])   # bid tick choices
        self.n2 = int(env.action_space.nvec[1])   # ask tick choices
        self.n_actions = self.n1 * self.n2

        # ── Q-table  shape: (inv, price, vol, time, action) ──────────────
        self.q_table = np.zeros((n_inv, n_price, n_vol, n_time, self.n_actions))

    # ── State helpers ────────────────────────────────────────────────────

    def discretize(self, obs: np.ndarray) -> tuple:
        inventory, mid_price, volatility, time_remaining = obs
        price_ratio = mid_price / self.env.initial_price

        inv_idx   = int(np.digitize(inventory,      self.inv_bins))
        price_idx = int(np.digitize(price_ratio,    self.price_bins))
        vol_idx   = int(np.digitize(volatility,     self.vol_bins))
        time_idx  = int(np.digitize(time_remaining, self.time_bins))

        return (inv_idx, price_idx, vol_idx, time_idx)

    # ── Action helpers ───────────────────────────────────────────────────

    def flatten_action(self, action: np.ndarray) -> int:
        return int(action[0]) * self.n2 + int(action[1])

    def unflatten_action(self, idx: int) -> np.ndarray:
        return np.array([idx // self.n2, idx % self.n2])

    # ── Policy ───────────────────────────────────────────────────────────

    def select_action(self, state: tuple) -> np.ndarray:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        best_idx = int(np.argmax(self.q_table[state]))
        return self.unflatten_action(best_idx)

    # ── Learning update ──────────────────────────────────────────────────

    def update(self, state, action_idx, reward, next_state, done):
        current_q = self.q_table[state][action_idx]
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action_idx] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Training loop ────────────────────────────────────────────────────

    def train(self, n_episodes: int = 2000, log_every: int = 100) -> list:
        """
        Run Q-learning for n_episodes.

        Returns
        -------
        episode_rewards : list of float
            Total reward per episode.
        """
        episode_rewards = []

        for episode in range(1, n_episodes + 1):
            obs, _ = self.env.reset()
            state = self.discretize(obs)
            total_reward = 0.0
            done = False

            while not done:
                action     = self.select_action(state)
                action_idx = self.flatten_action(action)

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = self.discretize(next_obs)
                self.update(state, action_idx, reward, next_state, done)

                state        = next_state
                total_reward += reward

            self.decay_epsilon()
            episode_rewards.append(total_reward)

            if episode % log_every == 0:
                avg = np.mean(episode_rewards[-log_every:])
                print(f"Episode {episode:>5}/{n_episodes} | "
                      f"Avg reward (last {log_every}): {avg:>8.2f} | "
                      f"Epsilon: {self.epsilon:.4f}")

        return episode_rewards

    # ── Evaluation ───────────────────────────────────────────────────────

    def evaluate(self, n_episodes: int = 100) -> dict:
        """Run greedy policy (epsilon=0) and return summary stats."""
        saved_eps = self.epsilon
        self.epsilon = 0.0

        rewards, equities = [], []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            state = self.discretize(obs)
            total_reward = 0.0
            done = False
            while not done:
                action = self.select_action(state)
                obs, reward, terminated, truncated, info = self.env.step(action)
                state = self.discretize(obs)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
            equities.append(info.get("equity", 0.0))

        self.epsilon = saved_eps
        return {
            "mean_reward":  np.mean(rewards),
            "std_reward":   np.std(rewards),
            "mean_equity":  np.mean(equities),
            "std_equity":   np.std(equities),
        }

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str = "q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)
        print(f"Q-table saved to {path}")

    def load(self, path: str = "q_table.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]
        print(f"Q-table loaded from {path}")
