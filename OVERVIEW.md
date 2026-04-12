# AI Market Maker — Reinforcement Learning

A tabular Q-learning agent that learns to quote bid/ask prices as a market maker, trained in a custom Gymnasium environment.

---

## Architecture

```
Stock.py              — price simulation (Gaussian random walk)
Buyer.py              — stochastic buyer model
Seller.py             — stochastic seller model
market_maker_env.py   — Gymnasium environment
q_learning_agent.py   — tabular Q-learning agent
train.py              — training + evaluation script
baseline.py           — random-action baseline
```

---

## Components

### `Stock`
Simulates a mid-price using a multiplicative Gaussian random walk:

```
price *= (1 + shock),  shock ~ N(drift, volatility)
```

Default: `volatility=0.01`, `drift=0.0` (pure random walk). Price is clipped at `$0.01` to prevent negatives.

---

### `Buyer` / `Seller`
Stochastic counterparties that decide whether to trade based on quote distance from mid-price.

Fill probability model:
```
fill_prob = max(0, 0.95 - 0.15 * ticks_away)
```

- At 1 tick away: ~80% fill probability
- Probability decays 15% per additional tick
- Buyers trade against the ask; sellers trade against the bid

---

### `MarketMakingEnv` (Gymnasium)

**Action space:** `MultiDiscrete([5, 5])` — independently choose bid and ask offsets from 1–5 ticks away from mid-price.

**Observation space:** `Box` with 4 features:

| Feature | Range | Description |
|---|---|---|
| `inventory` | `[-100, 100]` | Current share position |
| `mid_price` | `[0, ∞)` | Current stock mid-price |
| `volatility` | `[0, ∞)` | Stock volatility |
| `time_remaining` | `[0, 1]` | Fraction of episode left |

**Reward:**
```
reward = Δequity − α × inventory²
```
where `equity = cash + inventory × mid_price` and `α` penalizes inventory risk.

**Episode flow per step:**
1. Decode action → bid/ask prices
2. Simulate Poisson-distributed buyer/seller arrivals; fill trades probabilistically
3. Step the stock price
4. Compute reward; advance time

Default config: `max_steps=1000`, `buyer_arrival_rate=3`, `seller_arrival_rate=3`, `alpha=0.0001`.

---

### `QLearningAgent`

Tabular Q-learning with a discretized state space.

**Discretization:**

| Dimension | Bins |
|---|---|
| Inventory | 20 linearly spaced bins over `[-max_inv, max_inv]` |
| Price ratio (`mid/initial`) | 9 manually chosen breakpoints |
| Volatility | 5 bins |
| Time remaining | 10 linearly spaced bins |

**Q-table shape:** `(21 × 10 × 6 × 11 × 25)` — ~346,500 entries.

**Hyperparameters (defaults):**

| Parameter | Value |
|---|---|
| Learning rate (α) | 0.1 |
| Discount (γ) | 0.99 |
| ε start | 1.0 |
| ε min | 0.05 |
| ε decay | 0.997 |

**Update rule:** Standard Q-learning (off-policy TD):
```
Q(s,a) += α × (r + γ × max_a' Q(s',a') − Q(s,a))
```

**Persistence:** Q-table serialized to `q_table.pkl` via pickle.

---

### `train.py`

Runs 3000 training episodes, evaluates the greedy policy over 200 episodes, and saves a learning curve to `training_curve.png`.

Training config: `max_steps=500`, `alpha=0.001`.

### `baseline.py`

Runs 3000 episodes with a purely random policy (uniform random action each step) using the same environment config as `train.py`. Saves a reward curve to `baseline_curve.png` for comparison against the trained agent.
