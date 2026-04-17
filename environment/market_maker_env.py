import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Buyer import Buyer
from Seller import Seller
from Stock import Stock


class MarketMakingEnv(gym.Env):
    """
    A market-making environment where an RL agent sets bid/ask prices
    to maximize profit while managing inventory risk
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()

        # Default config
        cfg = config or {}
        self.tick_size = cfg.get("tick_size", 0.05)         # Minimum price increment.
        self.max_ticks = cfg.get("max_ticks", 5)            # Max quote offset in ticks.
        self.max_inventory = cfg.get("max_inventory", 100)  # Inventory limit (long/short).
        self.initial_price = cfg.get("initial_price", 100.0)
        self.max_steps = cfg.get("max_steps", 1000)         # Episode length.
        self.stock_volatility = cfg.get("stock_volatility", 0.0015)
        self.stock_drift = cfg.get("stock_drift", 0.0)
        self.adverse_selection_strength = cfg.get("adverse_selection_strength", 0.0025)
        self.maker_fee = cfg.get("maker_fee", 0.015)
        self.prev_equity = 0.0
        self.alpha = cfg.get("alpha", 0.01)
        self.terminal_inventory_penalty = cfg.get("terminal_inventory_penalty", 1.0)
        # Increase inventory penalty into the final stretch.
        self.time_risk_start_frac = cfg.get("time_risk_start_frac", 0.25)  # Ramp starts in last x% of episode.
        self.time_risk_k = cfg.get("time_risk_k", 3.0)  # Max multiplier is (1 + k).
        self.vol_ewma_beta = cfg.get("vol_ewma_beta", 0.94)
        self.buyer_arrival_rate = cfg.get("buyer_arrival_rate", 1)
        self.seller_arrival_rate = cfg.get("seller_arrival_rate", 1)

        # Initialize stock object
        self.stock = Stock(
            "AAPL",
            self.initial_price,
            volatility=self.stock_volatility,
            drift=self.stock_drift,
        )

        # Action space: Discrete(max_ticks * max_ticks)
        # action = bid_ticks * max_ticks + ask_ticks  (both 0-indexed, offset = ticks+1)
        self.action_space = spaces.Discrete(self.max_ticks * self.max_ticks)

        # Observation = [inv/max_inv, mid_price/initial_price, volatility, time_remaining]
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
        self.cash = 0.0
        self.last_mid_price = self.initial_price

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
        self.last_mid_price = self.initial_price

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """
        At each time step:
        1. Decode action into bid/ask prices
        2. Simulate buyer/seller arrivals and fills
        3. Step price with adverse selection + EWMA volatility update
        4. Calculate reward (normalized profit - ramped inventory penalty)
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

        spread_capture = 0.0
        filled_buys = 0
        filled_sells = 0

        # Buyer arrivals hit the ask (agent sells, inventory decreases)
        n_buyers = self.np_random.poisson(lam=self.buyer_arrival_rate)
        for _ in range(n_buyers):
            buyer = Buyer()
            if buyer.wants_to_trade(ask_price, mid_price, self.tick_size, rng=self.np_random):
                if self.inventory > -self.max_inventory:
                    self.inventory -= 1
                    self.cash += ask_price - self.maker_fee
                    spread_capture += ask_offset
                    filled_buys += 1

        # Seller arrivals hit the bid (agent buys, inventory increases)
        n_sellers = self.np_random.poisson(lam=self.seller_arrival_rate)
        for _ in range(n_sellers):
            seller = Seller()
            if seller.wants_to_trade(bid_price, mid_price, self.tick_size, rng=self.np_random):
                if self.inventory < self.max_inventory:
                    self.inventory += 1
                    self.cash -= bid_price + self.maker_fee
                    spread_capture += bid_offset
                    filled_sells += 1

        hit_inventory_cap = abs(self.inventory) >= self.max_inventory

        # Price change with adverse selection
        self.stock.step(rng=self.np_random)
        self.mid_price = self.stock.get_price()
        total_fills = filled_buys + filled_sells
        if total_fills > 0:
            # Buy pressure moves price up, sell pressure moves it down
            flow_imbalance = (filled_buys - filled_sells) / total_fills
            adverse_move = self.adverse_selection_strength * flow_imbalance
            self.mid_price *= (1.0 + adverse_move)

        price_return = (self.mid_price - self.last_mid_price) / max(self.last_mid_price, 1e-8)
        self.volatility = (
            self.vol_ewma_beta * self.volatility
            + (1.0 - self.vol_ewma_beta) * abs(price_return)
        )
        self.last_mid_price = self.mid_price

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
        reward = np.clip(reward, -5.0, 5.0)

        self.prev_equity = equity

        # Advance time
        self.current_step += 1
        self.time_remaining = 1 - (self.current_step / self.max_steps)

        terminated = hit_inventory_cap
        truncated = self.current_step >= self.max_steps
        if terminated or truncated:
            terminal_penalty = self.terminal_inventory_penalty * (inv_frac_abs ** 2)
            reward -= terminal_penalty

        observation = self._get_observation()
        info = {
            "equity":        equity,
            "inventory":     self.inventory,
            "cash":          self.cash,
            "mid_price":     self.mid_price,
            "bid_price":     bid_price,
            "ask_price":     ask_price,
            "bid_offset":    int(bid_ticks + 1),
            "ask_offset":    int(ask_ticks + 1),
            "spread_capture": spread_capture,
            "filled_buys":   filled_buys,
            "filled_sells":  filled_sells,
            "n_buyers":      int(n_buyers),
            "n_sellers":     int(n_sellers),
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Return normalized observation vector."""
        return np.array([
            self.inventory / self.max_inventory,
            self.mid_price / self.initial_price,
            self.volatility,
            self.time_remaining,
        ], dtype=np.float32)

    def render(self):
        """Print current state for quick debugging."""
        print(
            f"Step: {self.current_step} | "
            f"Mid: ${self.mid_price:.2f} | "
            f"Inventory: {self.inventory} | "
            f"Cash: ${self.cash:.2f} | "
            f"Time left: {self.time_remaining:.2%}"
        )
