"""
Side-by-side comparison of the three agents evaluated on the same 1000 seeds.

Reads:
  results/eval_qlearning.csv
  results/eval_baseline.csv
  results/eval_dqn.csv

Writes:
  results/comparison.md  — markdown table for the report
And prints the same table to the terminal.
"""

import os
import pandas as pd

FILES = [
    ("Q-learning", "results/eval_qlearning.csv"),
    ("DQN",        "results/eval_dqn.csv"),
    ("Random",     "results/eval_baseline.csv"),
]

# ── Load ──────────────────────────────────────────────────────────────────────
rows = []
for name, path in FILES:
    if not os.path.exists(path):
        print(f"Missing: {path} — skipping {name}")
        continue
    df = pd.read_csv(path)
    rows.append({
        "agent":         name,
        "episodes":      len(df),
        "mean_reward":   df.reward.mean(),
        "std_reward":    df.reward.std(),
        "mean_equity":   df.final_equity.mean(),
        "std_equity":    df.final_equity.std(),
        "mean_abs_inv":  df.final_inventory.abs().mean(),
        "max_abs_inv":   df.final_inventory.abs().max(),
    })

if not rows:
    raise SystemExit("No eval CSVs found. Run train_final.py + DQN eval first.")

# ── Terminal printout ─────────────────────────────────────────────────────────
print(f"\n{'Agent':11} {'episodes':>9} {'mean_reward':>12} {'std_reward':>11} "
      f"{'mean_equity':>12} {'mean_|inv|':>11} {'max_|inv|':>10}")
print("-" * 82)
for r in rows:
    print(f"{r['agent']:11} {r['episodes']:>9d} "
          f"{r['mean_reward']:>12.3f} {r['std_reward']:>11.3f} "
          f"{r['mean_equity']:>12.2f} {r['mean_abs_inv']:>11.2f} "
          f"{r['max_abs_inv']:>10.0f}")

# ── Head-to-head win rates (per-seed) ─────────────────────────────────────────
def per_seed_rewards(path):
    df = pd.read_csv(path).sort_values("episode")
    return df.reward.reset_index(drop=True)

have = {name: path for name, path in FILES if os.path.exists(path)}
print("\nHead-to-head win rates (per-seed reward comparison, 1000 seeds):")
pairs = [("Q-learning", "Random"), ("DQN", "Random"), ("Q-learning", "DQN")]
win_rate_rows = []
for a, b in pairs:
    if a in have and b in have:
        ra = per_seed_rewards(have[a])
        rb = per_seed_rewards(have[b])
        wins = int((ra > rb).sum())
        n = min(len(ra), len(rb))
        pct = wins / n * 100
        print(f"  {a:11} > {b:11}:  {wins:4d}/{n} = {pct:5.1f}%")
        win_rate_rows.append((a, b, wins, n, pct))

# ── Markdown output ───────────────────────────────────────────────────────────
md_path = "results/comparison.md"
lines = [
    "# Agent Comparison",
    "",
    "Three agents evaluated on the same 1000 seeds using the shared env config",
    "from `results/best_config.json`.",
    "",
    "## Aggregate metrics",
    "",
    "| Agent | Episodes | Mean Reward | Std Reward | Mean Final Equity | Mean \\|Inventory\\| | Max \\|Inventory\\| |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| **{r['agent']}** | {r['episodes']} | "
        f"{r['mean_reward']:.3f} | {r['std_reward']:.3f} | "
        f"{r['mean_equity']:.2f} | {r['mean_abs_inv']:.2f} | "
        f"{r['max_abs_inv']:.0f} |"
    )

if win_rate_rows:
    lines += [
        "",
        "## Head-to-head win rate (per-seed reward)",
        "",
        "| Matchup | Wins | Win Rate |",
        "|---|---:|---:|",
    ]
    for a, b, wins, n, pct in win_rate_rows:
        lines.append(f"| {a} vs {b} | {wins} / {n} | {pct:.1f}% |")

# Takeaways based on the numbers
q = next((r for r in rows if r["agent"] == "Q-learning"), None)
d = next((r for r in rows if r["agent"] == "DQN"), None)
b = next((r for r in rows if r["agent"] == "Random"), None)
lines += ["", "## Takeaways", ""]
if q and b:
    lines.append(f"- Q-learning mean reward **{q['mean_reward']:+.3f}** vs Random **{b['mean_reward']:+.3f}** "
                 f"— {q['mean_reward']-b['mean_reward']:+.3f} absolute improvement.")
if d and b:
    lines.append(f"- DQN mean reward **{d['mean_reward']:+.3f}** vs Random **{b['mean_reward']:+.3f}** "
                 f"— {d['mean_reward']-b['mean_reward']:+.3f} absolute improvement.")
if q and d:
    lines.append(f"- Between the two learned agents: Q-learning {q['mean_reward']:+.3f} vs DQN {d['mean_reward']:+.3f}.")
if q and b and q["mean_abs_inv"] < b["mean_abs_inv"]:
    ratio = b["mean_abs_inv"] / max(q["mean_abs_inv"], 1e-6)
    lines.append(f"- Inventory management: Q-learning holds **{q['mean_abs_inv']:.1f}** shares vs Random's "
                 f"**{b['mean_abs_inv']:.1f}** (~{ratio:.1f}× tighter).")

with open(md_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nMarkdown comparison saved to {md_path}")
