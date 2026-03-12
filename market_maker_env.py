import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MarketMakingEnv(gym.Env):
    """
    A market-making environment where an RL agent sets bid/ask prices
    to maximize profit while managing inventory risk.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()

        # Default config
        cfg = config or {}
        self.tick_size = cfg.get("tick_size", 0.01)       # minimum price increment
        self.max_ticks = cfg.get("max_ticks", 5)           # max ticks away from mid
        self.max_inventory = cfg.get("max_inventory", 100)  # inventory cap (long or short)
        self.initial_price = cfg.get("initial_price", 100.0)
        self.max_steps = cfg.get("max_steps", 1000)         # time steps per episode

        # Action space
        # Each action is a pair of integers: (bid_offset, ask_offset)
        # MultiDiscrete([5, 5]) means each value ranges from 0 to 4
        # We add 1 when using them so the actual range is 1 to 5 ticks
        self.action_space = spaces.MultiDiscrete([self.max_ticks, self.max_ticks])

        # Observation space
        # [inventory, mid_price, volatility, time_remaining]
        self.observation_space = spaces.Box(
            low=np.array([
                -self.max_inventory,  # inventory: can be short (negative)
                0.0,                  # mid_price: won't go below 0
                0.0,                  # volatility: always positive
                0.0,                  # time_remaining: 0 = episode over
            ], dtype=np.float32),
            high=np.array([
                self.max_inventory,   # inventory: max long position
                np.inf,               # mid_price: no upper bound
                np.inf,               # volatility: no upper bound
                1.0,                  # time_remaining: 1 = just started
            ], dtype=np.float32),
        )

        # Internal state (for reset())
        self.inventory = 0
        self.mid_price = self.initial_price
        self.volatility = 0.0
        self.time_remaining = 1.0
        self.current_step = 0
        self.cash = 0.0  # tracks money from trades

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TODO: initialize mid_price, inventory, volatility, etc.
        # TODO: return (observation, info)
        pass

    def step(self, action):
        """
        At each time step:
        1. Decode action into bid/ask prices
        2. Simulate price movement
        3. Check if any orders get filled
        4. Calculate reward
        5. Return (observation, reward, terminated, truncated, info)
        """
        # TODO: implement the core logic
        pass

    def _get_observation(self):
        """Gives current state as observation array."""
        return np.array([
            self.inventory,
            self.mid_price,
            self.volatility,
            self.time_remaining,
        ], dtype=np.float32)

    def render(self):
        """Print current state for debugging."""
        print(
            f"Step: {self.current_step} | "
            f"Mid: ${self.mid_price:.2f} | "
            f"Inventory: {self.inventory} | "
            f"Cash: ${self.cash:.2f} | "
            f"Time left: {self.time_remaining:.2%}"
        )