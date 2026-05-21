"""
GRPO Wordle Trainer — improved

Fixes vs original:
  ✓ Action masking  — impossible words zeroed out before sampling
  ✓ GROUP_SIZE 8→16 — lower-variance within-group baseline
  ✓ ENTROPY_COEF 0.02→0.05 — more exploration early on
  ✓ state_dim from env  — no hardcoded mismatch
  ✓ MAX_GUESSES = 10 (set in env) + efficiency-shaped reward (set in env)
  ✓ Eval prints guess-distribution breakdown
"""

import os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from words import get_default_words
from env import WordleEnv, GREEN, YELLOW, GREY

EMOJI = {GREY: "⬛", YELLOW: "🟨", GREEN: "🟩"}

# ── Hyper-parameters ──────────────────────────────────────────────────────────
LR             = 3e-4
CLIP_EPS       = 0.2
ENTROPY_COEF   = 0.05    # ↑ from 0.02 — encourages exploration early
GROUP_SIZE     = 16      # ↑ from 8  — lower-variance group baseline
N_EPOCHS       = 4
BATCH_SIZE     = 128
EPS_PER_UPDATE = 64
TOTAL_EPISODES = 300_000
LOG_INTERVAL   = 10
HIDDEN         = 256
DEVICE         = "cpu"
CKPT_DIR       = "checkpoints"
SHOW_GAME_EVERY= 500
WORDLE_NUMBER  = 1796


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


# ── Display ───────────────────────────────────────────────────────────────────

def print_game(env: WordleEnv, episode_num: int, mode: str = "GRPO"):
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


# ── Data collection ───────────────────────────────────────────────────────────

def collect_groups(
    model, env: WordleEnv, dev, n_episodes: int, group_size: int,
    total_episodes: int, show_every: int
):
    all_s, all_a, all_lp, all_adv = [], [], [], []
    ep_rewards, ep_wins = [], []
    show_env, show_ep   = None, 0

    for ep_i in range(n_episodes):
        target    = random.choice(env.answer_list)
        group_r   = []
        group_traj = []

        for _ in range(group_size):
            state = env.reset(target=target)
            traj_s, traj_a, traj_lp = [], [], []
            ep_r, done = 0.0, False

            while not done:
                s_t = torch.tensor(state, dtype=torch.float32, device=dev).unsqueeze(0)

                # ── action masking ────────────────────────────────────────────
                with torch.no_grad():
                    logits, _ = model(s_t)
                mask = torch.tensor(env.get_valid_mask(), dtype=torch.bool, device=dev)
                logits = logits.clone()
                logits[0, ~mask] = -1e9
                # ─────────────────────────────────────────────────────────────

                dist = torch.distributions.Categorical(logits=logits)
                a    = dist.sample()
                lp   = dist.log_prob(a).item()
                state, r, done, info = env.step(a.item())
                traj_s.append(state); traj_a.append(a.item()); traj_lp.append(lp)
                ep_r += r

            group_r.append(ep_r)
            group_traj.append((traj_s, traj_a, traj_lp))
            ep_wins.append(int(info["won"]))

        ep_total = total_episodes + ep_i * group_size
        if (ep_total % show_every) < group_size:
            show_env = env
            show_ep  = ep_total

        ep_rewards.extend(group_r)

        # GRPO advantage: normalise within the group
        gr   = np.array(group_r, np.float32)
        norm = (gr - gr.mean()) / (gr.std() + 1e-8)

        for g_i, (ts, ta, tlp) in enumerate(group_traj):
            adv = float(norm[g_i])
            for s, a, lp in zip(ts, ta, tlp):
                all_s.append(s); all_a.append(a)
                all_lp.append(lp); all_adv.append(adv)

    return (
        np.array(all_s,   np.float32),
        np.array(all_a,   np.int64),
        np.array(all_lp,  np.float32),
        np.array(all_adv, np.float32),
        ep_rewards, ep_wins, show_env, show_ep,
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    words, answers = get_default_words()
    env   = WordleEnv(words, answers)
    model = ActorCritic(env.state_dim, env.action_dim, HIDDEN).to(DEVICE)
    opt   = Adam(model.parameters(), lr=LR)
    dev   = torch.device(DEVICE)

    all_rewards, all_wins = [], []
    total_eps, updates    = 0, 0

    print("=" * 66)
    print("  GRPO Wordle Trainer  (with action masking + shaped reward)")
    print(f"  Vocab: {len(words):,}  |  Answers: {len(answers):,}  |  Target eps: {TOTAL_EPISODES:,}")
    print(f"  group_size={GROUP_SIZE}  eps_per_update={EPS_PER_UPDATE}  "
          f"clip={CLIP_EPS}  lr={LR}")
    print(f"  hidden={HIDDEN}  state_dim={env.state_dim}  actions={env.action_dim}")
    print(f"  entropy_coef={ENTROPY_COEF}  max_guesses={env.action_dim and 10}")
    print("=" * 66)

    t0 = time.time()

    while total_eps < TOTAL_EPISODES:
        model.eval()
        S_np, A_np, LP_np, ADV_np, ep_r, ep_w, show_env, show_ep = collect_groups(
            model, env, dev,
            EPS_PER_UPDATE, GROUP_SIZE,
            total_eps, SHOW_GAME_EVERY,
        )

        if show_env is not None:
            print_game(show_env, show_ep, mode="GRPO")

        all_rewards.extend(ep_r)
        all_wins.extend(ep_w)
        total_eps += EPS_PER_UPDATE * GROUP_SIZE
        updates   += 1

        S   = torch.tensor(S_np,   dtype=torch.float32, device=dev)
        A   = torch.tensor(A_np,   dtype=torch.long,    device=dev)
        LP  = torch.tensor(LP_np,  dtype=torch.float32, device=dev)
        ADV = torch.tensor(ADV_np, dtype=torch.float32, device=dev)
        N   = len(S)

        model.train()
        total_loss = 0.0
        n_batches  = 0
        for _ in range(N_EPOCHS):
            idx = torch.randperm(N, device=dev)
            for start in range(0, N, BATCH_SIZE):
                b         = idx[start : start + BATCH_SIZE]
                logits, _ = model(S[b])
                dist      = torch.distributions.Categorical(logits=logits)
                new_lp    = dist.log_prob(A[b])
                ent       = dist.entropy()
                ratio     = (new_lp - LP[b]).exp()
                s1        = ratio * ADV[b]
                s2        = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * ADV[b]
                p_loss    = -torch.min(s1, s2).mean()
                e_loss    = -ent.mean()
                loss      = p_loss + ENTROPY_COEF * e_loss
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                total_loss += loss.item()
                n_batches  += 1

        if updates % LOG_INTERVAL == 0:
            recent_wins = all_wins[-1_000:]
            recent_r    = all_rewards[-1_000:]
            wr       = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0.0
            avg      = sum(recent_r)    / len(recent_r)          if recent_r    else 0.0
            elapsed  = time.time() - t0
            eps_s    = total_eps / elapsed
            avg_loss = total_loss / max(1, n_batches)
            print(
                f"[GRPO] ep {total_eps:>8,} | upd {updates:>4} | "
                f"win {wr:5.1f}% | avg_r {avg:+.3f} | "
                f"loss {avg_loss:.4f} | {eps_s:.0f} ep/s"
            )

    # ── Save checkpoint ───────────────────────────────────────────────────────
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = os.path.join(CKPT_DIR, "grpo.pt")
    torch.save({"model": model.state_dict(), "words": words, "answers": answers}, ckpt)
    print(f"\n💾  Checkpoint saved → {ckpt}")
    print(f"⏱   Total training time: {(time.time()-t0)/60:.1f} min")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  GRPO Final Eval — 500 greedy games")
    print("=" * 66)

    model.eval()
    wins, scores = 0, []
    hist = [0] * 12   # index = n_guesses (0 unused); index 11 = failed

    with torch.no_grad():
        for _ in range(500):
            s    = env.reset()
            done = False
            while not done:
                s_t   = torch.tensor(s, dtype=torch.float32, device=dev).unsqueeze(0)
                logits, _ = model(s_t)
                mask  = torch.tensor(env.get_valid_mask(), dtype=torch.bool, device=dev)
                logits[0, ~mask] = -1e9
                act   = logits.argmax(dim=-1).item()
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
                s_t   = torch.tensor(s, dtype=torch.float32, device=dev).unsqueeze(0)
                logits, _ = model(s_t)
                mask  = torch.tensor(env.get_valid_mask(), dtype=torch.bool, device=dev)
                logits[0, ~mask] = -1e9
                act   = logits.argmax(dim=-1).item()
                s, _, done, _ = env.step(act)
        print_game(env, i, mode="GRPO-greedy")


if __name__ == "__main__":
    train()