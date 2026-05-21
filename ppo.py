"""
PPO Wordle Trainer — improved

Fixes vs original:
  ✓ Action masking  — impossible words zeroed out before sampling & eval
  ✓ ENTROPY_COEF 0.02→0.05 — more exploration early on
  ✓ state_dim from env  — no hardcoded mismatch
  ✓ MAX_GUESSES = 10 (set in env) + efficiency-shaped reward (set in env)
  ✓ Advantage normalisation per rollout (unchanged, already good)
  ✓ Eval uses masking too for fair comparison
"""

import os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from words import get_default_words
from env import WordleEnv, GREEN, YELLOW, GREY

EMOJI = {GREY: "⬛", YELLOW: "🟨", GREEN: "🟩"}

# ── Hyper-parameters ──────────────────────────────────────────────────────────
LR              = 3e-4
GAMMA           = 0.99
LAM             = 0.95
CLIP_EPS        = 0.2
ENTROPY_COEF    = 0.05   # ↑ from 0.02 — encourages exploration early
VF_COEF         = 0.5
N_EPOCHS        = 4
BATCH_SIZE      = 128
ROLLOUT_STEPS   = 1024
TOTAL_STEPS     = 300_000
LOG_INTERVAL    = 10
HIDDEN          = 256
DEVICE          = "cpu"
CKPT_DIR        = "checkpoints"
SHOW_GAME_EVERY = 500
WORDLE_NUMBER   = 1796


# ── Model ─────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.5)
                nn.init.zeros_(m.bias)

    def forward(self, s):
        h = self.trunk(s)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, s, mask: torch.Tensor = None):
        logits, v = self.forward(s)
        if mask is not None:
            logits = logits.clone()
            logits[~mask] = -1e9
        dist = torch.distributions.Categorical(logits=logits)
        a    = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), v

    def evaluate(self, s, a, mask: torch.Tensor = None):
        logits, v = self.forward(s)
        if mask is not None:
            logits = logits.clone()
            logits[~mask] = -1e9
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(a), dist.entropy(), v


# ── Rollout buffer ─────────────────────────────────────────────────────────────

@dataclass
class Buffer:
    states:    list = field(default_factory=list)
    actions:   list = field(default_factory=list)
    rewards:   list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values:    list = field(default_factory=list)
    dones:     list = field(default_factory=list)
    masks:     list = field(default_factory=list)   # ← action masks per step

    def add(self, s, a, r, lp, v, d, m):
        self.states.append(s);    self.actions.append(a)
        self.rewards.append(r);   self.log_probs.append(lp)
        self.values.append(v);    self.dones.append(d)
        self.masks.append(m)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


# ── GAE ───────────────────────────────────────────────────────────────────────

def gae(rewards, values, dones, last_v, gamma, lam):
    T        = len(rewards)
    rewards  = np.array(rewards,  np.float32)
    values   = np.array(values,   np.float32)
    dones    = np.array(dones,    np.float32)
    adv      = np.zeros(T,        np.float32)
    g        = 0.0
    vals_ext = np.append(values, last_v)
    for t in reversed(range(T)):
        delta  = rewards[t] + gamma * vals_ext[t+1] * (1 - dones[t]) - vals_ext[t]
        g      = delta + gamma * lam * (1 - dones[t]) * g
        adv[t] = g
    return adv + values, adv


# ── Display ───────────────────────────────────────────────────────────────────

def print_game(env: WordleEnv, episode_num: int, mode: str = "PPO"):
    n   = len(env.guesses_made)
    won = bool(env.guesses_made) and env.guesses_made[-1] == env.target
    score = f"{n}/10" if won else "X/10"
    wnum  = WORDLE_NUMBER + (episode_num // 100)

    print(f"\nWordle {wnum} {score}  [{mode} | ep {episode_num:,}]")
    for guess, fb in zip(env.guesses_made, env.feedback_history):
        tiles = "".join(EMOJI[f] for f in fb)
        print(f"  {tiles}  {guess.upper()}")
    if won:
        print(f"  ✅  Answer: {env.target.upper()}  (solved in {n})")
    else:
        print(f"  ❌  Answer was: {env.target.upper()}")
    print()


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    words, answers = get_default_words()
    env   = WordleEnv(words, answers)
    model = ActorCritic(env.state_dim, env.action_dim, HIDDEN).to(DEVICE)
    opt   = Adam(model.parameters(), lr=LR)
    buf   = Buffer()
    dev   = torch.device(DEVICE)

    ep_rewards, ep_wins              = [], []
    steps, updates, episodes         = 0, 0, 0
    state = env.reset()
    ep_r  = 0.0

    print("=" * 66)
    print("  PPO Wordle Trainer  (with action masking + shaped reward)")
    print(f"  Vocab: {len(words):,}  |  Answers: {len(answers):,}  |  "
          f"Target steps: {TOTAL_STEPS:,}")
    print(f"  rollout={ROLLOUT_STEPS}  epochs={N_EPOCHS}  clip={CLIP_EPS}  lr={LR}")
    print(f"  gamma={GAMMA}  lambda={LAM}  vf_coef={VF_COEF}  "
          f"ent_coef={ENTROPY_COEF}")
    print(f"  hidden={HIDDEN}  state_dim={env.state_dim}  actions={env.action_dim}")
    print("=" * 66)

    t0 = time.time()

    while steps < TOTAL_STEPS:
        model.eval()
        with torch.no_grad():
            for _ in range(ROLLOUT_STEPS):
                s_t  = torch.tensor(state, dtype=torch.float32, device=dev).unsqueeze(0)

                # ── action masking ────────────────────────────────────────────
                raw_mask = env.get_valid_mask()              # numpy bool array
                mask_t   = torch.tensor(raw_mask, dtype=torch.bool, device=dev)
                a, lp, _, v = model.act(s_t, mask_t.unsqueeze(0))
                # ─────────────────────────────────────────────────────────────

                next_s, r, done, info = env.step(a.item())
                ep_r += r
                buf.add(state, a.item(), r, lp.item(), v.item(),
                        float(done), raw_mask)
                state = next_s

                if done:
                    episodes += 1
                    ep_rewards.append(ep_r)
                    ep_wins.append(int(info["won"]))
                    if episodes % SHOW_GAME_EVERY == 0:
                        print_game(env, episodes, mode="PPO")
                    ep_r  = 0.0
                    state = env.reset()

            s_t    = torch.tensor(state, dtype=torch.float32, device=dev).unsqueeze(0)
            _, lv  = model(s_t)
            last_v = lv.item()

        returns, advs = gae(buf.rewards, buf.values, buf.dones, last_v, GAMMA, LAM)

        S    = torch.tensor(np.array(buf.states),    dtype=torch.float32, device=dev)
        A    = torch.tensor(buf.actions,             dtype=torch.long,    device=dev)
        LP   = torch.tensor(buf.log_probs,           dtype=torch.float32, device=dev)
        R    = torch.tensor(returns,                 dtype=torch.float32, device=dev)
        ADV  = torch.tensor(advs,                    dtype=torch.float32, device=dev)
        MASKS= torch.tensor(np.array(buf.masks),     dtype=torch.bool,    device=dev)
        ADV  = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        N          = len(S)
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for _ in range(N_EPOCHS):
            idx = torch.randperm(N, device=dev)
            for start in range(0, N, BATCH_SIZE):
                b              = idx[start : start + BATCH_SIZE]
                new_lp, ent, val = model.evaluate(S[b], A[b], MASKS[b])
                ratio          = (new_lp - LP[b]).exp()
                s1             = ratio * ADV[b]
                s2             = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * ADV[b]
                p_loss         = -torch.min(s1, s2).mean()
                v_loss         = F.mse_loss(val, R[b])
                e_loss         = -ent.mean()
                loss           = p_loss + VF_COEF * v_loss + ENTROPY_COEF * e_loss
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                total_loss += loss.item()
                n_batches  += 1

        buf.clear()
        steps   += ROLLOUT_STEPS
        updates += 1

        if updates % LOG_INTERVAL == 0:
            recent_wins = ep_wins[-1_000:]
            recent_r    = ep_rewards[-1_000:]
            wr       = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0.0
            avg      = sum(recent_r)    / len(recent_r)          if recent_r    else 0.0
            elapsed  = time.time() - t0
            sps      = steps / elapsed
            avg_loss = total_loss / max(1, n_batches)
            print(
                f"[PPO]  step {steps:>8,} | ep {episodes:>6,} | upd {updates:>4} | "
                f"win {wr:5.1f}% | avg_r {avg:+.3f} | "
                f"loss {avg_loss:.4f} | {sps:.0f} s/s"
            )

    # ── Save checkpoint ───────────────────────────────────────────────────────
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = os.path.join(CKPT_DIR, "ppo.pt")
    torch.save({"model": model.state_dict(), "words": words, "answers": answers}, ckpt)
    print(f"\n💾  Checkpoint saved → {ckpt}")
    print(f"⏱   Total training time: {(time.time()-t0)/60:.1f} min")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  PPO Final Eval — 500 greedy games")
    print("=" * 66)

    model.eval()
    wins, scores = 0, []
    hist = [0] * 12   # index = n_guesses; index 11 = failed

    with torch.no_grad():
        for _ in range(500):
            s    = env.reset()
            done = False
            while not done:
                s_t    = torch.tensor(s, dtype=torch.float32, device=dev).unsqueeze(0)
                logits, _ = model(s_t)
                mask   = torch.tensor(env.get_valid_mask(), dtype=torch.bool, device=dev)
                logits[0, ~mask] = -1e9
                act    = logits.argmax(dim=-1).item()
                s, _, done, info = env.step(act)
            ng = info["n_guesses"]
            if info["won"]:
                wins += 1; scores.append(ng)
                hist[ng] = hist[ng] + 1
            else:
                hist[11] += 1

    wr  = wins / 500 * 100
    avg = np.mean(scores) if scores else float("nan")
    print(f"\n  Results: {wins}/500 won  ({wr:.1f}%)  |  Avg guesses: {avg:.2f}\n")
    for g in range(1, 11):
        bar = "█" * hist[g]
        pct = hist[g] / 500 * 100
        print(f"  {g:2d}/10  {bar:<40}  {hist[g]:3d}  ({pct:.1f}%)")
    bar = "░" * hist[11]
    pct = hist[11] / 500 * 100
    print(f"   X/10  {bar:<40}  {hist[11]:3d}  ({pct:.1f}%)")

    print("\n  ── 5 sample greedy games ──")
    for i in range(5):
        s    = env.reset()
        done = False
        with torch.no_grad():
            while not done:
                s_t    = torch.tensor(s, dtype=torch.float32, device=dev).unsqueeze(0)
                logits, _ = model(s_t)
                mask   = torch.tensor(env.get_valid_mask(), dtype=torch.bool, device=dev)
                logits[0, ~mask] = -1e9
                act    = logits.argmax(dim=-1).item()
                s, _, done, _ = env.step(act)
        print_game(env, i, mode="PPO-greedy")


if __name__ == "__main__":
    train()