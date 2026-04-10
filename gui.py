"""
gui.py  —  Market Maker DQN  Live Training Visualizer
Run with:  python gui.py
"""

import tkinter as tk
from tkinter import ttk
import threading
import queue
import sys
import os
import random

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from market_maker_env import MarketMakingEnv, DQN, ReplayBuffer

# ── Hyperparameters ───────────────────────────────────────────────────────────
GAMMA             = 0.99
LR                = 1e-3
BATCH_SIZE        = 64
BUFFER_CAPACITY   = 10_000
TARGET_UPDATE_FREQ= 100
NUM_EPISODES      = 300
MAX_STEPS         = 1000
EPS_START         = 1.0
EPS_END           = 0.05
EPS_DECAY         = 0.995
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# how many recent step-points to keep in rolling charts
WINDOW = 400

# ── Catppuccin-inspired dark palette ─────────────────────────────────────────
BG       = "#1e1e2e"
PANEL    = "#2a2a3e"
FG       = "#cdd6f4"
GREEN    = "#a6e3a1"
RED      = "#f38ba8"
BLUE     = "#89b4fa"
YELLOW   = "#f9e2af"
MAUVE    = "#cba6f7"
TEAL     = "#94e2d5"
GREY     = "#44475a"


# ── DQN helpers ───────────────────────────────────────────────────────────────
def _select_action(model, state, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        return int(torch.argmax(model(t), dim=1).item())


def _train_step(policy, target, buf, opt):
    if len(buf) < BATCH_SIZE:
        return None
    states, actions, rewards, next_states, dones = buf.sample(BATCH_SIZE)
    states      = torch.tensor(states,      dtype=torch.float32, device=DEVICE)
    actions     = torch.tensor(actions,     dtype=torch.long,    device=DEVICE).unsqueeze(1)
    rewards_t   = torch.tensor(rewards,     dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
    dones_t     = torch.tensor(dones,       dtype=torch.float32, device=DEVICE).unsqueeze(1)
    current_q   = policy(states).gather(1, actions)
    with torch.no_grad():
        max_next_q = target(next_states).max(dim=1, keepdim=True)[0]
        target_q   = rewards_t + GAMMA * max_next_q * (1 - dones_t)
    loss = nn.MSELoss()(current_q, target_q)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


# ── GUI ───────────────────────────────────────────────────────────────────────
class TrainingGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Market Maker DQN — Live Training Visualizer")
        self.root.configure(bg=BG)
        self.root.geometry("1680x960")
        self.root.minsize(1200, 700)

        self._q        = queue.Queue()
        self._running  = False
        self._paused   = False
        self._pause_ev = threading.Event()
        self._pause_ev.set()

        self._build_top_bar()
        self._build_body()
        self._reset_data()

        self.root.after(60, self._poll)

    # =========================================================================
    # Layout
    # =========================================================================
    def _build_top_bar(self):
        bar = tk.Frame(self.root, bg=BG, pady=6)
        bar.pack(fill="x", padx=10)

        btn = dict(bg=PANEL, fg=FG, relief="flat", padx=14, pady=5,
                   font=("Consolas", 10, "bold"), cursor="hand2",
                   activebackground=GREY, activeforeground=FG)

        self.btn_start = tk.Button(bar, text="▶  Start",  command=self._start,  **btn)
        self.btn_pause = tk.Button(bar, text="⏸  Pause",  command=self._pause,
                                   state="disabled", **btn)
        self.btn_stop  = tk.Button(bar, text="⏹  Stop",   command=self._stop,
                                   state="disabled", **btn)
        for b in (self.btn_start, self.btn_pause, self.btn_stop):
            b.pack(side="left", padx=4)

        sep = tk.Frame(bar, bg=GREY, width=2, height=28)
        sep.pack(side="left", padx=10)

        tk.Label(bar, text="Speed:", bg=BG, fg=FG,
                 font=("Consolas", 9)).pack(side="left", padx=(0, 4))
        self.speed_var = tk.IntVar(value=1)
        ttk.Scale(bar, from_=1, to=30, orient="horizontal",
                  variable=self.speed_var, length=120).pack(side="left")

        sep2 = tk.Frame(bar, bg=GREY, width=2, height=28)
        sep2.pack(side="left", padx=10)

        stats = [
            ("lbl_ep",   "Episode:  —"),
            ("lbl_step", "Step:  —"),
            ("lbl_eps",  "ε:  —"),
            ("lbl_rew",  "Reward:  —"),
            ("lbl_inv",  "Inventory:  —"),
            ("lbl_eq",   "Equity:  —"),
        ]
        for attr, text in stats:
            lbl = tk.Label(bar, text=text, bg=PANEL, fg=FG,
                           font=("Consolas", 9), padx=10, pady=4)
            lbl.pack(side="left", padx=5)
            setattr(self, attr, lbl)

    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # ── Left/centre charts ────────────────────────────────────────────
        charts = tk.Frame(body, bg=BG)
        charts.pack(side="left", fill="both", expand=True)

        # Row 1
        row1 = tk.Frame(charts, bg=BG)
        row1.pack(fill="both", expand=True)

        self.fig_price, self.ax_price = self._make_panel(row1, side="left",
            expand=True, w=7.2, h=3.1, title="Stock Price  •  mid / bid / ask")
        self.fig_inv,   self.ax_inv   = self._make_panel(row1, side="left",
            expand=True, w=3.4, h=3.1, title="Inventory")
        self.fig_rew,   self.ax_rew   = self._make_panel(row1, side="left",
            expand=True, w=3.4, h=3.1, title="Step Reward")

        # Row 2
        row2 = tk.Frame(charts, bg=BG)
        row2.pack(fill="both", expand=True)

        self.fig_eq,   self.ax_eq   = self._make_panel(row2, side="left",
            expand=True, w=5.0, h=3.1, title="Equity & Cash")
        self.fig_heat, self.ax_heat = self._make_panel(row2, side="left",
            expand=False, w=3.6, h=3.1, title="Action Heatmap  (bid ticks × ask ticks)")
        self.fig_ep,   self.ax_ep   = self._make_panel(row2, side="left",
            expand=True, w=5.2, h=3.1, title="Episode Total Reward")

        # ── Right sidebar: trade log ──────────────────────────────────────
        sidebar = tk.Frame(body, bg=PANEL, width=230)
        sidebar.pack(side="right", fill="y", padx=(6, 0))
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text=" Trade & Fill Log", bg=PANEL, fg=MAUVE,
                 font=("Consolas", 9, "bold"), anchor="w"
                 ).pack(fill="x", padx=6, pady=(6, 2))

        self.log = tk.Text(sidebar, bg=PANEL, fg=FG, font=("Consolas", 8),
                           relief="flat", state="disabled", wrap="word",
                           selectbackground=GREY)
        self.log.pack(fill="both", expand=True, padx=4, pady=4)
        self.log.tag_config("buy",  foreground=GREEN)
        self.log.tag_config("sell", foreground=RED)
        self.log.tag_config("hdr",  foreground=YELLOW)
        self.log.tag_config("ep",   foreground=MAUVE)
        self.log.tag_config("done", foreground=TEAL)

    def _make_panel(self, parent, *, side, expand, w, h, title):
        fig, ax = plt.subplots(figsize=(w, h))
        fig.patch.set_facecolor(PANEL)
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=FG, fontsize=8, pad=4)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GREY)
        fig.tight_layout(pad=1.4)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        widget = canvas.get_tk_widget()
        widget.configure(bg=PANEL, highlightthickness=0)
        widget.pack(side=side, fill="both", expand=expand, padx=3, pady=3)

        fig._cv = canvas     # store reference so we can call draw_idle
        return fig, ax

    # =========================================================================
    # Data stores
    # =========================================================================
    def _reset_data(self):
        # rolling step-level series (last WINDOW points)
        self.xs        = []   # global step index
        self.mid_y     = []
        self.bid_y     = []
        self.ask_y     = []
        self.inv_y     = []
        self.rew_y     = []
        self.cash_y    = []
        self.eq_y      = []
        # episode-level series
        self.ep_rewards = []
        # action frequency grid
        self.act_counts = np.zeros((5, 5), dtype=float)
        self._global_step = 0

    def _push(self, lst, val):
        lst.append(val)
        if len(lst) > WINDOW:
            lst.pop(0)

    # =========================================================================
    # Training thread
    # =========================================================================
    def _train_loop(self):
        env    = MarketMakingEnv(config={"max_steps": MAX_STEPS})
        s_dim  = env.observation_space.shape[0]
        a_dim  = env.action_space.n

        policy = DQN(s_dim, a_dim).to(DEVICE)
        target = DQN(s_dim, a_dim).to(DEVICE)
        target.load_state_dict(policy.state_dict())
        target.eval()

        opt    = optim.Adam(policy.parameters(), lr=LR)
        buf    = ReplayBuffer(BUFFER_CAPACITY)
        eps    = EPS_START
        gstep  = 0

        for ep in range(1, NUM_EPISODES + 1):
            if not self._running:
                break
            state, _ = env.reset()
            ep_reward = 0.0
            losses    = []

            for step in range(MAX_STEPS):
                self._pause_ev.wait()
                if not self._running:
                    break

                action      = _select_action(policy, state, eps, a_dim)
                next_state, reward, terminated, truncated, info = env.step(action)
                done        = terminated or truncated

                buf.push(state, action, reward, next_state, done)
                loss = _train_step(policy, target, buf, opt)
                if loss is not None:
                    losses.append(loss)

                state      = next_state
                ep_reward += reward
                gstep     += 1

                if gstep % TARGET_UPDATE_FREQ == 0:
                    target.load_state_dict(policy.state_dict())

                # throttle: only enqueue every N steps based on speed slider
                skip = max(1, self.speed_var.get())
                if step % skip == 0:
                    self._q.put({
                        "t":            "step",
                        "episode":      ep,
                        "step":         step,
                        "gstep":        gstep,
                        "mid":          info["mid_price"],
                        "bid":          info["bid_price"],
                        "ask":          info["ask_price"],
                        "inventory":    env.inventory,
                        "cash":         env.cash,
                        "equity":       info["equity"],
                        "reward":       reward,
                        "epsilon":      eps,
                        "action":       action,
                        "loss":         loss,
                        "n_buyers":     info["n_buyers"],
                        "n_sellers":    info["n_sellers"],
                        "buyer_fills":  info["buyer_fills"],
                        "seller_fills": info["seller_fills"],
                    })

                if done:
                    break

            eps = max(EPS_END, eps * EPS_DECAY)
            avg_loss = float(np.mean(losses)) if losses else 0.0
            self._q.put({
                "t":         "episode",
                "episode":   ep,
                "ep_reward": ep_reward,
                "epsilon":   eps,
                "avg_loss":  avg_loss,
            })

        if self._running:
            torch.save(policy.state_dict(), "dqn_market_maker.pt")
        self._q.put({"t": "done"})

    # =========================================================================
    # Queue polling  (runs on the GUI / main thread)
    # =========================================================================
    def _poll(self):
        need_draw = False
        try:
            while True:
                msg = self._q.get_nowait()
                self._ingest(msg)
                need_draw = True
        except queue.Empty:
            pass
        if need_draw:
            self._redraw()
        self.root.after(60, self._poll)

    def _ingest(self, msg):
        t = msg["t"]
        if t == "step":
            self._ingest_step(msg)
        elif t == "episode":
            self._ingest_episode(msg)
        elif t == "done":
            self._on_done()

    def _ingest_step(self, d):
        self._push(self.xs,     d["gstep"])
        self._push(self.mid_y,  d["mid"])
        self._push(self.bid_y,  d["bid"])
        self._push(self.ask_y,  d["ask"])
        self._push(self.inv_y,  d["inventory"])
        self._push(self.rew_y,  d["reward"])
        self._push(self.cash_y, d["cash"])
        self._push(self.eq_y,   d["equity"])

        r, c = d["action"] // 5, d["action"] % 5
        self.act_counts[r, c] += 1

        # status bar
        inv  = d["inventory"]
        inv_color = RED if inv > 50 else (GREEN if inv < -50 else FG)
        self.lbl_ep.config(  text=f"Episode:  {d['episode']}/{NUM_EPISODES}")
        self.lbl_step.config(text=f"Step:  {d['step']}")
        self.lbl_eps.config( text=f"ε:  {d['epsilon']:.3f}")
        self.lbl_rew.config( text=f"Reward:  {d['reward']:+.3f}")
        self.lbl_inv.config( text=f"Inventory:  {inv:+d}", fg=inv_color)
        self.lbl_eq.config(  text=f"Equity:  ${d['equity']:.2f}")

        # log fills only (not every step)
        bf = d["buyer_fills"]
        sf = d["seller_fills"]
        if bf or sf:
            ep, step = d["episode"], d["step"]
            mid = d["mid"]
            if bf:
                self._log(
                    f"S{step:04d} | {bf} buyer{'s' if bf>1 else ''} filled "
                    f"@ ask={d['ask']:.3f}  (mid={mid:.2f})\n", "buy")
            if sf:
                self._log(
                    f"S{step:04d} | {sf} seller{'s' if sf>1 else ''} filled "
                    f"@ bid={d['bid']:.3f}  (mid={mid:.2f})\n", "sell")

    def _ingest_episode(self, d):
        self.ep_rewards.append(d["ep_reward"])
        self._log(
            f"── Ep {d['episode']:3d}  reward={d['ep_reward']:+8.2f}  "
            f"ε={d['epsilon']:.3f}  loss={d['avg_loss']:.4f} ──\n", "ep")

    def _on_done(self):
        self._running = False
        self._log("=== Training complete. Model saved to dqn_market_maker.pt ===\n", "done")
        self.btn_start.config(state="normal",   text="▶  Start")
        self.btn_pause.config(state="disabled", text="⏸  Pause")
        self.btn_stop.config( state="disabled")

    # =========================================================================
    # Drawing
    # =========================================================================
    def _ax_reset(self, ax, title=None):
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GREY)
        if title:
            ax.set_title(title, color=FG, fontsize=8, pad=4)

    def _redraw(self):
        xs = self.xs
        if not xs:
            return

        # ── Price ─────────────────────────────────────────────────────────
        self._ax_reset(self.ax_price, "Stock Price  •  mid / bid / ask")
        self.ax_price.plot(xs, self.mid_y, color=BLUE,  lw=1.2, label="mid",  zorder=3)
        self.ax_price.plot(xs, self.bid_y, color=GREEN, lw=0.9, ls="--", label="bid", zorder=2)
        self.ax_price.plot(xs, self.ask_y, color=RED,   lw=0.9, ls="--", label="ask", zorder=2)
        self.ax_price.fill_between(xs, self.bid_y, self.ask_y,
                                   color=TEAL, alpha=0.18, label="spread")
        self.ax_price.legend(fontsize=6.5, facecolor=PANEL, labelcolor=FG,
                             framealpha=0.7, loc="upper left")
        self.fig_price._cv.draw_idle()

        # ── Inventory ─────────────────────────────────────────────────────
        self._ax_reset(self.ax_inv, "Inventory")
        bar_colors = [GREEN if v >= 0 else RED for v in self.inv_y]
        self.ax_inv.bar(xs, self.inv_y, color=bar_colors, alpha=0.75,
                        width=max(1, len(xs) // WINDOW + 1))
        self.ax_inv.axhline(0,     color=GREY,  lw=0.8)
        self.ax_inv.axhline( 100,  color=RED,   lw=0.7, ls=":", alpha=0.6, label="+max")
        self.ax_inv.axhline(-100,  color=GREEN, lw=0.7, ls=":", alpha=0.6, label="-max")
        self.ax_inv.legend(fontsize=6, facecolor=PANEL, labelcolor=FG,
                           framealpha=0.7, loc="upper left")
        self.fig_inv._cv.draw_idle()

        # ── Step reward ───────────────────────────────────────────────────
        self._ax_reset(self.ax_rew, "Step Reward")
        bar_colors = [GREEN if v >= 0 else RED for v in self.rew_y]
        self.ax_rew.bar(xs, self.rew_y, color=bar_colors, alpha=0.75,
                        width=max(1, len(xs) // WINDOW + 1))
        self.ax_rew.axhline(0, color=GREY, lw=0.8)
        self.fig_rew._cv.draw_idle()

        # ── Equity & Cash ─────────────────────────────────────────────────
        self._ax_reset(self.ax_eq, "Equity & Cash")
        self.ax_eq.plot(xs, self.eq_y,   color=YELLOW, lw=1.2, label="equity")
        self.ax_eq.plot(xs, self.cash_y, color=MAUVE,  lw=0.9, ls="--", label="cash")
        self.ax_eq.axhline(0, color=GREY, lw=0.8)
        self.ax_eq.legend(fontsize=6.5, facecolor=PANEL, labelcolor=FG,
                          framealpha=0.7, loc="upper left")
        self.fig_eq._cv.draw_idle()

        # ── Action heatmap ────────────────────────────────────────────────
        self._ax_reset(self.ax_heat, "Action Heatmap  (bid ticks × ask ticks)")
        norm = self.act_counts / (self.act_counts.sum() + 1e-9)
        self.ax_heat.imshow(norm, cmap="plasma", aspect="auto",
                            vmin=0, vmax=max(norm.max(), 1e-6))
        self.ax_heat.set_xticks(range(5))
        self.ax_heat.set_xticklabels(["1", "2", "3", "4", "5"], fontsize=7, color=FG)
        self.ax_heat.set_yticks(range(5))
        self.ax_heat.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7, color=FG)
        self.ax_heat.set_xlabel("Ask ticks offset", color=FG, fontsize=7)
        self.ax_heat.set_ylabel("Bid ticks offset", color=FG, fontsize=7)
        for i in range(5):
            for j in range(5):
                self.ax_heat.text(j, i, f"{norm[i,j]:.2f}",
                                  ha="center", va="center",
                                  fontsize=6, color="white", fontweight="bold")
        self.fig_heat._cv.draw_idle()

        # ── Episode rewards ───────────────────────────────────────────────
        if self.ep_rewards:
            self._ax_reset(self.ax_ep, "Episode Total Reward")
            ep_xs = list(range(1, len(self.ep_rewards) + 1))
            bar_c = [GREEN if v >= 0 else RED for v in self.ep_rewards]
            self.ax_ep.bar(ep_xs, self.ep_rewards, color=bar_c, alpha=0.7)
            self.ax_ep.axhline(0, color=GREY, lw=0.8)
            if len(self.ep_rewards) >= 10:
                rm = np.convolve(self.ep_rewards, np.ones(10) / 10, mode="valid")
                self.ax_ep.plot(range(10, len(self.ep_rewards) + 1), rm,
                                color=YELLOW, lw=1.4, label="10-ep avg")
                self.ax_ep.legend(fontsize=6.5, facecolor=PANEL, labelcolor=FG,
                                  framealpha=0.7, loc="upper left")
            self.fig_ep._cv.draw_idle()

    # =========================================================================
    # Log helper
    # =========================================================================
    def _log(self, text, tag=""):
        self.log.config(state="normal")
        self.log.insert("end", text, tag)
        # trim to last 600 lines
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 600:
            self.log.delete("1.0", f"{lines - 600}.0")
        self.log.see("end")
        self.log.config(state="disabled")

    # =========================================================================
    # Controls
    # =========================================================================
    def _start(self):
        if self._running:
            return
        self._reset_data()
        self._running  = True
        self._paused   = False
        self._pause_ev.set()
        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal",  text="⏸  Pause")
        self.btn_stop.config( state="normal")
        threading.Thread(target=self._train_loop, daemon=True).start()

    def _pause(self):
        if not self._running:
            return
        if self._paused:
            self._paused = False
            self._pause_ev.set()
            self.btn_pause.config(text="⏸  Pause")
        else:
            self._paused = True
            self._pause_ev.clear()
            self.btn_pause.config(text="▶  Resume")

    def _stop(self):
        self._running = False
        self._pause_ev.set()
        self.btn_start.config(state="normal",   text="▶  Start")
        self.btn_pause.config(state="disabled", text="⏸  Pause")
        self.btn_stop.config( state="disabled")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    TrainingGUI(root)
    root.mainloop()
