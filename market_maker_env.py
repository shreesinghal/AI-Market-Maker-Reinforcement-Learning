import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Buyer import Buyer
from Seller import Seller
from Stock import Stock
import torch
import torch.nn as nn
import random
from collections import deque
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


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
        self.tick_size = cfg.get("tick_size", 0.01)         # minimum price increment
        self.max_ticks = cfg.get("max_ticks", 5)            # max ticks away from mid
        self.max_inventory = cfg.get("max_inventory", 100)  # inventory cap (long or short)
        self.initial_price = cfg.get("initial_price", 100.0)
        self.max_steps = cfg.get("max_steps", 1000)         # time steps per episode
        self.alpha = cfg.get("alpha", 0.01)
        self.buyer_arrival_rate = cfg.get("buyer_arrival_rate", 3)
        self.seller_arrival_rate = cfg.get("seller_arrival_rate", 3)

        # Initialize stock object
        self.stock = Stock("APPL", self.initial_price, volatility=0.02)

        # Action space
        # Each action is a pair of integers: (bid_offset, ask_offset)
        # MultiDiscrete([5, 5]) means each value ranges from 0 to 4
        # We add 1 when using them so the actual range is 1 to 5 ticks
        self.action_space = spaces.Discrete(self.max_ticks * self.max_ticks)

        # Observation space
        # [inventory, mid_price, volatility, time_remaining]
        self.observation_space = spaces.Box(
            low=np.array([
                -1.0,  # normalized inventory
                0.0,   # normalized price
                0.0,   # volatility
                0.0,   # time_remaining
            ], dtype=np.float32),
            high=np.array([
                1.0,       # normalized inventory
                np.inf,    # normalized price
                np.inf,    # volatility
                1.0,       # time_remaining
            ], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state (for reset())
        self.inventory = 0
        self.mid_price = self.initial_price
        self.volatility = self.stock.volatility
        self.time_remaining = 1.0
        self.current_step = 0
        self.cash = 0.0  # tracks money from trades

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.inventory = 0
        self.mid_price = self.initial_price
        self.stock.price = self.initial_price
        self.volatility = self.stock.volatility
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

        bid_ticks = action // self.max_ticks
        ask_ticks = action % self.max_ticks

        # convert ticks to prices
        bid_offset = (bid_ticks + 1) * self.tick_size
        ask_offset = (ask_ticks + 1) * self.tick_size

        mid_price = self.stock.get_price()

        bid_price = mid_price - bid_offset
        ask_price = mid_price + ask_offset

        # Simulate buyer arrivals
        n_buyers = self.np_random.poisson(lam=self.buyer_arrival_rate)
        buyer_fills = 0
        for _ in range(n_buyers):
            buyer = Buyer()
            if buyer.wants_to_trade(ask_price, mid_price, self.tick_size, rng=self.np_random):
                if self.inventory > -self.max_inventory:
                    self.inventory -= 1
                    self.cash += ask_price
                    buyer_fills += 1

        # Simulate seller arrivals
        n_sellers = self.np_random.poisson(lam=self.seller_arrival_rate)
        seller_fills = 0
        for _ in range(n_sellers):
            seller = Seller()
            if seller.wants_to_trade(bid_price, mid_price, self.tick_size, rng=self.np_random):
                if self.inventory < self.max_inventory:
                    self.inventory += 1
                    self.cash -= bid_price
                    seller_fills += 1

        # Price change
        self.stock.step()
        self.mid_price = self.stock.get_price()

        # Reward: spread income actually captured, minus inventory risk penalty.
        # Using Δequity was wrong — price noise (±$2/step on 1 share) swamps
        # the spread signal ($0.01–0.05/fill), making it impossible to learn.
        spread_income = buyer_fills * ask_offset + seller_fills * bid_offset
        inventory_penalty = self.alpha * (self.inventory ** 2)
        reward = spread_income - inventory_penalty

        # Track equity separately for logging
        equity = self.cash + self.inventory * self.mid_price

        # Update time and step
        self.current_step += 1
        self.time_remaining = 1 - (self.current_step / self.max_steps)

        terminated = False
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        info = {
            "equity":       equity,
            "bid_price":    bid_price,
            "ask_price":    ask_price,
            "mid_price":    mid_price,
            "n_buyers":     n_buyers,
            "n_sellers":    n_sellers,
            "buyer_fills":  buyer_fills,
            "seller_fills": seller_fills,
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Gives current state as normalized observation array."""
        return np.array([
            self.inventory / self.max_inventory,
            self.mid_price / self.initial_price,
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

