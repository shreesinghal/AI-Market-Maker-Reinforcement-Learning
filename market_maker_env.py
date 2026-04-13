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
        self.time_risk_start_frac = cfg.get("time_risk_start_frac", 0.5)  # ramp starts when this fraction of time remains
        self.time_risk_k = cfg.get("time_risk_k", 3.0)                   # max extra multiplier on inventory penalty
        self.buyer_arrival_rate = cfg.get("buyer_arrival_rate", 3)
        self.seller_arrival_rate = cfg.get("seller_arrival_rate", 3)
        self.stock_volatility = cfg.get("stock_volatility", 0.02)
        self.adverse_selection_strength = cfg.get("adverse_selection_strength", 0.0)
        self.maker_fee = cfg.get("maker_fee", 0.0)
        self.terminal_inventory_penalty = cfg.get("terminal_inventory_penalty", 0.0)
        self.vol_ewma_beta = cfg.get("vol_ewma_beta", 0.94)

        # Initialize stock object
        self.stock = Stock("APPL", self.initial_price, volatility=self.stock_volatility)

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
        self.prev_equity = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.inventory = 0
        self.mid_price = self.initial_price
        self.stock.price = self.initial_price
        self.volatility = self.stock.volatility
        self.time_remaining = 1.0
        self.current_step = 0
        self.cash = 0.0
        self.prev_equity = 0.0

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

        # Adverse selection: informed flow pushes price against market maker
        net_order_flow = buyer_fills - seller_fills
        if net_order_flow != 0 and self.adverse_selection_strength > 0:
            self.stock.price *= (1 + self.adverse_selection_strength * net_order_flow)

        # Maker fee (rebate per fill)
        total_fills = buyer_fills + seller_fills
        if total_fills > 0 and self.maker_fee > 0:
            self.cash += self.maker_fee * total_fills * mid_price

        # Price change
        self.stock.step()
        self.mid_price = self.stock.get_price()

        # Update volatility with EWMA of absolute returns
        if mid_price > 0:
            ret = abs((self.mid_price - mid_price) / mid_price)
            self.volatility = self.vol_ewma_beta * self.volatility + (1 - self.vol_ewma_beta) * ret

        # Reward = normalized equity change - ramped inventory penalty
        equity = self.cash + self.inventory * self.mid_price
        profit_change = (equity - self.prev_equity) / self.initial_price

        inv_frac_abs = abs(self.inventory / self.max_inventory)
        if self.time_risk_start_frac <= 0:
            time_progress = 1.0
        else:
            time_progress = np.clip(
                (self.time_risk_start_frac - self.time_remaining) / self.time_risk_start_frac,
                0.0,
                1.0,
            )
        risk_multiplier = 1.0 + self.time_risk_k * time_progress
        inventory_penalty = self.alpha * (inv_frac_abs ** 2) * risk_multiplier

        reward = profit_change - inventory_penalty

        # Update time and step
        self.current_step += 1
        self.time_remaining = 1 - (self.current_step / self.max_steps)

        terminated = False
        truncated = self.current_step >= self.max_steps

        # Terminal inventory penalty at episode end
        if truncated and self.terminal_inventory_penalty > 0:
            reward -= self.terminal_inventory_penalty * inv_frac_abs

        reward = np.clip(reward, -5.0, 5.0)
        self.prev_equity = equity

        observation = self._get_observation()
        info = {
            "equity":       equity,
            "inventory":    self.inventory,
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

