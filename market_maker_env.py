import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Buyer import Buyer
from Seller import Seller
from Stock import Stock


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
        self.prev_equity = 0.0
        self.alpha = cfg.get("alpha", 0.001)
        self.buyer_arrival_rate = cfg.get("buyer_arrival_rate", 3)
        self.seller_arrival_rate = cfg.get("seller_arrival_rate", 3)

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

        self.inventory = 0
        self.mid_price = self.initial_price
        self.volatility = 0.02
        self.time_remaining = 1.0
        self.current_step = 0
        self.cash = 0.0

        observation = self._get_observation()
        info = {}

        return observation, info

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

        bid_ticks, ask_ticks = action

        # convert ticks to prices
        bid_offset = (bid_ticks + 1) * self.tick_size
        ask_offset = (ask_ticks + 1) * self.tick_size

        mid_price = self.stock.get_price()

        bid_price = mid_price - bid_offset
        ask_price = mid_price + ask_offset

        # simulate buyers sellers TODO
        # closer quotes = higher fill probability
        bid_distance = abs(mid_price - bid_price)
        ask_distance = abs(ask_price - mid_price)

        # Simulate buyer arrivals
        n_buyers = self.np_random.poisson(lam=self.buyer_arrival_rate)
        for i in range(n_buyers):
            buyer = Buyer()
            if buyer.wants_to_trade(ask_price, mid_price, self.tick_size, rng=self.np_random):
                self.inventory -= 1
                self.cash += ask_price

        # Simulate seller arrivals
        n_sellers = self.np_random.poisson(lam=self.seller_arrival_rate)
        for i in range(n_sellers):
            seller = Seller()
            if seller.wants_to_trade(bid_price, mid_price, self.tick_size, rng=self.np_random):
                self.inventory += 1
                self.cash -= bid_price

        # save previous mid_price
        previous_price = self.mid_price

        #price change
        self.stock.step()
        self.mid_price = self.stock.get_price()

        # reward
        equity = self.cash + self.inventory * previous_price
        reward = equity - self.prev_equity
        inventory_penalty = self.alpha * (self.inventory ** 2)
        reward -= inventory_penalty
        self.prev_equity = equity

        # update time and step
        self.current_step += 1
        self.time_remaining = 1 - (self.current_step / self.max_steps)

        terminated = False
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        info = {"equity": equity}

        return observation, reward, terminated, truncated, info
        



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