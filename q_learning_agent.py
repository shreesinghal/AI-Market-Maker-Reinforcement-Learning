import numpy as np
import pickle
from environment import MarketMakingEnv


class QLearningAgent:
    """
    Tabular Q-learning agent for the MarketMakingEnv.

    The continuous observation space is discretized into bins so we can
    maintain a finite Q-table.

    State dimensions (all observations are normalized by the env)
    ----------------
    inv_norm      : inventory / max_inventory  ∈ [-1, 1]    → inv_bins
    price_ratio   : mid_price / initial_price  ∈ [0.7, 1.3] → price_bins
    volatility    : EWMA of |returns|          ∈ [0, 0.01]  → vol_bins
    time_remaining: 0 … 1                                   → time_bins

    Action space
    ------------
    Discrete(max_ticks²) — a single integer encoding (bid_ticks * max_ticks + ask_ticks).
    Used directly as the Q-table action index.
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
        # Inventory: normalized to [-1, 1]
        self.inv_bins = np.linspace(-1.0, 1.0, 12)      # 13 buckets

        # Price ratio: finer resolution near 1.0 where most time is spent
        self.price_bins = np.array([0.85, 0.93, 0.97, 0.99, 1.0,
                                    1.01, 1.03, 1.07, 1.15])  # 10 buckets

        # Volatility: EWMA of |returns|, typically 0.0005–0.005
        self.vol_bins = np.array([0.0005, 0.001, 0.0015, 0.002, 0.003])  # 6 buckets

        # Time remaining: [0, 1]
        self.time_bins = np.linspace(0.0, 1.0, 10)      # 11 buckets

        # Number of buckets (digitize returns 1-based, so bins+1 possible values)
        n_inv   = len(self.inv_bins)   + 1   # 13
        n_price = len(self.price_bins) + 1   # 10
        n_vol   = len(self.vol_bins)   + 1   # 6
        n_time  = len(self.time_bins)  + 1   # 11

        # ── Action space ─────────────────────────────────────────────────
        self.n_actions = env.action_space.n   # 25 (5×5)

        # ── Q-table  shape: (inv, price, vol, time, action) ──────────────
        # = (13, 10, 6, 11, 25) = 214,500 entries
        self.q_table = np.zeros((n_inv, n_price, n_vol, n_time, self.n_actions))

    # ── State helpers ────────────────────────────────────────────────────

    def discretize(self, obs: np.ndarray) -> tuple:
        inv_norm, price_ratio, volatility, time_remaining = obs

        inv_idx   = int(np.digitize(inv_norm,       self.inv_bins))
        price_idx = int(np.digitize(price_ratio,    self.price_bins))
        vol_idx   = int(np.digitize(volatility,     self.vol_bins))
        time_idx  = int(np.digitize(time_remaining, self.time_bins))

        return (inv_idx, price_idx, vol_idx, time_idx)

    # ── Policy ───────────────────────────────────────────────────────────

    def select_action(self, state: tuple) -> int:
        """Epsilon-greedy action selection. Returns an int in [0, n_actions)."""
        if np.random.random() < self.epsilon:
            return int(self.env.action_space.sample())
        return int(np.argmax(self.q_table[state]))

    # ── Learning update ──────────────────────────────────────────────────

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Training loop ────────────────────────────────────────────────────

    def train(self, n_episodes: int = 2000, log_every: int = 100) -> list:
        """
        Run Q-learning for n_episodes.

        Returns
        -------
        (episode_rewards, episode_equities) : tuple of (list[float], list[float])
            Total reward and final equity per episode.
        """
        episode_rewards = []
        episode_equities = []

        for episode in range(1, n_episodes + 1):
            obs, _ = self.env.reset()
            state = self.discretize(obs)
            total_reward = 0.0
            done = False
            info = {}

            while not done:
                action = self.select_action(state)

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                next_state = self.discretize(next_obs)
                self.update(state, action, reward, next_state, done)

                state        = next_state
                total_reward += reward

            self.decay_epsilon()
            episode_rewards.append(total_reward)
            episode_equities.append(info.get("equity", 0.0))

            if episode % log_every == 0:
                avg = np.mean(episode_rewards[-log_every:])
                print(f"Episode {episode:>5}/{n_episodes} | "
                      f"Avg reward (last {log_every}): {avg:>8.2f} | "
                      f"Epsilon: {self.epsilon:.4f}")

        return episode_rewards, episode_equities

    # ── Evaluation ───────────────────────────────────────────────────────

    def evaluate(self, n_episodes: int = 100, seeds: list = None) -> dict:
        """
        Run greedy policy (epsilon=0) and return summary stats + per-episode lists.

        Parameters
        ----------
        seeds : list of int, optional
            If provided, reset each episode with seeds[i] for reproducibility.
        """
        saved_eps = self.epsilon
        self.epsilon = 0.0

        rewards, equities, inventories = [], [], []
        for i in range(n_episodes):
            seed = seeds[i] if seeds is not None else None
            obs, _ = self.env.reset(seed=seed)
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
            inventories.append(info.get("inventory", float(obs[0]) * self.env.max_inventory))

        self.epsilon = saved_eps
        return {
            "mean_reward":              np.mean(rewards),
            "std_reward":               np.std(rewards),
            "mean_equity":              np.mean(equities),
            "std_equity":               np.std(equities),
            "mean_abs_final_inventory": np.mean(np.abs(inventories)),
            "rewards":                  rewards,
            "equities":                 equities,
            "inventories":              inventories,
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
