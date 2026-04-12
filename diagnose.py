"""
Diagnostic script — run from your project root.
Helps figure out why the Q-learning agent isn't beating the baseline.
"""

import numpy as np
import pickle
import os

# --- 1. Q-table coverage ---
print("=" * 60)
print("1. Q-TABLE COVERAGE")
print("=" * 60)

q_path = "results/q_table.pkl"
if os.path.exists(q_path):
    with open(q_path, "rb") as f:
        q_table = pickle.load(f)

    total_entries = q_table.size
    # Entries that are still 0 (never updated)
    zero_entries = np.count_nonzero(q_table == 0.0)
    visited_entries = total_entries - zero_entries
    coverage = visited_entries / total_entries * 100

    print(f"Q-table shape: {q_table.shape}")
    print(f"Total entries: {total_entries:,}")
    print(f"Non-zero entries: {visited_entries:,} ({coverage:.1f}%)")
    print(f"Zero entries: {zero_entries:,} ({100 - coverage:.1f}%)")

    # Check which states have ANY non-zero action
    # Sum absolute Q-values across actions (last dimension)
    state_visited = np.any(q_table != 0.0, axis=-1)
    states_visited = np.count_nonzero(state_visited)
    total_states = state_visited.size
    print(f"\nStates with at least one visited action: {states_visited}/{total_states} ({states_visited/total_states*100:.1f}%)")

    # Q-value statistics for non-zero entries
    nonzero_vals = q_table[q_table != 0.0]
    if len(nonzero_vals) > 0:
        print(f"\nQ-value stats (non-zero only):")
        print(f"  Mean: {nonzero_vals.mean():.2f}")
        print(f"  Std:  {nonzero_vals.std():.2f}")
        print(f"  Min:  {nonzero_vals.min():.2f}")
        print(f"  Max:  {nonzero_vals.max():.2f}")

    # --- 2. Action distribution ---
    print("\n" + "=" * 60)
    print("2. LEARNED ACTION PREFERENCES")
    print("=" * 60)

    # For visited states, what actions does the agent prefer?
    # Get the greedy action for each state
    best_actions = np.argmax(q_table, axis=-1)

    # Only count states that were actually visited
    visited_mask = state_visited
    best_actions_visited = best_actions[visited_mask]

    # Decode actions: bid_offset = action // 5, ask_offset = action % 5
    bid_offsets = best_actions_visited // 5
    ask_offsets = best_actions_visited % 5

    print(f"\nBid offset distribution (0=1tick, 4=5ticks):")
    for i in range(5):
        count = np.sum(bid_offsets == i)
        pct = count / len(bid_offsets) * 100
        bar = "#" * int(pct / 2)
        print(f"  Offset {i+1} tick: {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\nAsk offset distribution (0=1tick, 4=5ticks):")
    for i in range(5):
        count = np.sum(ask_offsets == i)
        pct = count / len(ask_offsets) * 100
        bar = "#" * int(pct / 2)
        print(f"  Offset {i+1} tick: {count:4d} ({pct:5.1f}%) {bar}")

    # --- 3. Does the agent adjust by inventory? ---
    print("\n" + "=" * 60)
    print("3. DOES THE AGENT ADJUST BY INVENTORY?")
    print("=" * 60)
    print("(This is the key test — a smart market maker changes")
    print(" spreads based on inventory position)\n")

    # q_table shape is (n_inv, n_price, n_time, n_actions)
    n_inv = q_table.shape[0]

    # Average best action at different inventory levels
    # Collapse across price and time dimensions
    for inv_idx in [0, n_inv // 4, n_inv // 2, 3 * n_inv // 4, n_inv - 1]:
        # Get Q-values for this inventory level, all prices, all times
        q_slice = q_table[inv_idx]  # shape: (n_price, n_time, n_actions)

        # Check if any states at this inventory level were visited
        slice_visited = np.any(q_slice != 0.0, axis=-1)
        n_visited = np.count_nonzero(slice_visited)

        if n_visited > 0:
            # Get best actions for visited states
            best = np.argmax(q_slice, axis=-1)
            best_visited = best[slice_visited]
            avg_bid = np.mean(best_visited // 5) + 1  # +1 because 0-indexed
            avg_ask = np.mean(best_visited % 5) + 1

            inv_label = f"inv_bin={inv_idx}"
            if inv_idx == 0:
                inv_label += " (very short)"
            elif inv_idx == n_inv // 2:
                inv_label += " (neutral)"
            elif inv_idx == n_inv - 1:
                inv_label += " (very long)"

            print(f"  {inv_label}: avg bid={avg_bid:.1f} ticks, avg ask={avg_ask:.1f} ticks ({n_visited} states visited)")
        else:
            print(f"  inv_bin={inv_idx}: NO STATES VISITED")

else:
    print(f"No Q-table found at {q_path}")

# --- 4. Reward component analysis ---
print("\n" + "=" * 60)
print("4. REWARD NOISE ANALYSIS")
print("=" * 60)
print("Running 100 episodes to measure reward components...\n")

try:
    from market_maker_env import MarketMakingEnv

    env = MarketMakingEnv(config={
        "max_steps": 500,
        "buyer_arrival_rate": 3,
        "seller_arrival_rate": 3,
        "alpha": 0.0001,
        "spread_bonus": 0.5,
        "terminal_penalty": 0.1,
    })

    equity_changes = []
    penalties = []
    spreads_captured = []

    for ep in range(100):
        obs, _ = env.reset(seed=ep)
        prev_equity = 0.0
        ep_equity_changes = []
        ep_penalties = []

        for step in range(500):
            action = env.action_space.sample()  # random agent
            obs, reward, terminated, truncated, info = env.step(action)

            # Reconstruct components
            curr_equity = info.get("equity", env.cash + env.inventory * env.mid_price)
            delta_eq = curr_equity - prev_equity
            penalty = 0.0001 * (env.inventory ** 2)

            ep_equity_changes.append(abs(delta_eq))
            ep_penalties.append(penalty)
            prev_equity = curr_equity

            if terminated or truncated:
                break

        equity_changes.extend(ep_equity_changes)
        penalties.extend(ep_penalties)

    equity_changes = np.array(equity_changes)
    penalties = np.array(penalties)

    print(f"  Avg |equity change| per step:  {equity_changes.mean():.2f}")
    print(f"  Avg inventory penalty per step: {penalties.mean():.4f}")
    print(f"  Ratio (equity / penalty):       {equity_changes.mean() / max(penalties.mean(), 1e-8):.0f}x")
    print()
    print(f"  The equity swing is {equity_changes.mean() / max(penalties.mean(), 1e-8):.0f}x larger")
    print(f"  than the inventory penalty.")
    print(f"  If this ratio is >100x, the penalty is basically invisible")
    print(f"  to the agent and it has no incentive to manage inventory.")

except Exception as e:
    print(f"  Could not run environment: {e}")
    print(f"  (Run this script from your project root directory)")