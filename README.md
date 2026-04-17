# AI Market Maker: Q-Learning vs Deep Q-Network

This repository contains two reinforcement learning market maker agents: **tabular Q-learning** and **Deep Q-Network (DQN)**. Both agents operate on a custom market-making environment built with Gymnasium. An addtional random baseline agent is included for comparison purposes.

## Motivation

Both Q-learning and Deep Q-Network models can be applied for the task of creating a market maker agent, but it is unclear to us students how these two models would compare in efficiency, memory, time, and ease of implementation. Through this project, we aim to create both a Q-learning and Deep Q-Network model and apply both to the same environment with the same goal to compare the two.

## How It Works

The environment simulates a market maker that must continuously quote bid and ask prices around a fluctuating stock mid-price. At each step the agent chooses how many ticks away from mid to place its bid and ask (1-5 ticks each, giving 25 possible actions). Stochastic buyers and sellers arrive each step and may fill the agent's quotes. The agent earns the bid-ask spread on completed trades but is penalized for holding inventory.

**Observation space** (4 features): normalized inventory, price ratio, volatility estimate, time remaining

**Reward**: change in equity minus an inventory-risk penalty that ramps up toward the end of each episode

Both agents are trained and evaluated on the same environment configuration so the comparison is fair.

## Project Structure

```
├── environment/
│   ├── market_maker_env.py   # Gymnasium environment
│   ├── Stock.py              # Mid-price simulator (Gaussian random walk)
│   ├── Buyer.py              # Stochastic buyer model
│   └── Seller.py             # Stochastic seller model
├── q_learning_agent.py       # Q-learning agent
├── dqn_agent.py              # DQN neural network + replay buffer
├── train_qlearning.py            # Train & evaluate Q-learning
├── train_dqn.py              # Train & evaluate DQN
├── baseline.py               # Random baseline evaluation
├── compare_agents.py         # Side-by-side comparison
├── tune.py                   # Hyperparameter tuning (Q-learning)
├── results/                  # Saved models, evaluation CSVs, plots
└── requirements.txt
```
## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

The core dependencies are:

| Package | Purpose |
|---|---|
| `numpy` | Numerical computation |
| `torch` | Deep learning (DQN) |
| `gymnasium` | RL environment API |
| `matplotlib` | Training curve plots |
| `pandas` | Agent comparison analysis |

## Running the Models

### Train the Q-Learning Agent

```bash
python train_qlearning.py
```

**Outputs:**

| File | Description |
|---|---|
| `results/q_table.pkl` | Trained Q-table |
| `results/training_curve.csv` | Per-episode training rewards |
| `results/training_curve.png` | Learning curve plot |
| `results/eval_qlearning.csv` | Evaluation results (1,000 episodes) |
| `results/eval_baseline.csv` | Random baseline results |
| `results/comparison_summary.json` | Summary statistics |

### Train the DQN Agent

```bash
python train_dqn.py
```

**Outputs:**

| File | Description |
|---|---|
| `dqn_market_maker.pt` | Trained DQN model weights |
| `results/eval_dqn.csv` | Evaluation results (1,000 episodes) |

To skip training and evaluate an existing model:

```bash
python train_dqn.py --eval-only
```

### Compare All Agents

After both models have been trained and evaluated:

```bash
python compare_agents.py
```

This reads the evaluation CSVs and produces a head-to-head comparison in `results/comparison.md`.

### Run the Random Baseline 

```bash
python baseline.py
```

Runs 3,000 episodes with a uniform random policy and saves a reward curve to `baseline_curve.png`.

## Model Details

### Q-Learning

- **Approach**: Tabular Q-learning with discretized states
- **Q-table size**: 13 x 10 x 6 x 11 x 25 = 214,500 entries
- **Training**: 10,000 episodes, epsilon-greedy (decaying from 1.0 to 0.05)
- **Key hyperparameters**: learning rate = 0.01, discount = 0.99, epsilon decay = 0.9985

### DQN

- **Architecture**: 2 hidden layers (64 units each) with ReLU activations
- **Techniques**: Experience replay (buffer size 10,000), target network (updated every 200 steps), double DQN
- **Training**: 300 episodes, linear epsilon decay over 300,000 steps
- **Key hyperparameters**: learning rate = 0.001, batch size = 64, gradient clipping = 10.0
