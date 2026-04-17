# Agent Comparison

Three agents evaluated on the same 1000 seeds using the shared env config
from `results/best_config.json`.

## Aggregate metrics

| Agent | Episodes | Mean Reward | Std Reward | Mean Final Equity | Mean \|Inventory\| | Max \|Inventory\| |
|---|---:|---:|---:|---:|---:|---:|
| **Q-learning** | 1000 | 0.534 | 0.128 | 57.82 | 2.26 | 30 |
| **DQN** | 1000 | 0.457 | 0.174 | 56.71 | 3.22 | 12 |
| **Random** | 1000 | -0.458 | 0.730 | 43.59 | 17.41 | 30 |

## Head-to-head win rate (per-seed reward)

| Matchup | Wins | Win Rate |
|---|---:|---:|
| Q-learning vs Random | 945 / 1000 | 94.5% |
| DQN vs Random | 898 / 1000 | 89.8% |
| Q-learning vs DQN | 672 / 1000 | 67.2% |

## Takeaways

- Q-learning mean reward **+0.534** vs Random **-0.458** — +0.992 absolute improvement.
- DQN mean reward **+0.457** vs Random **-0.458** — +0.915 absolute improvement.
- Between the two learned agents: Q-learning +0.534 vs DQN +0.457.
- Inventory management: Q-learning holds **2.3** shares vs Random's **17.4** (~7.7× tighter).
