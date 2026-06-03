import os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dataclasses import dataclass, field
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from words import get_split_words
from env import WordleEnv, GREEN, YELLOW, GREY
from metrics import (
    MetricsLogger, EntropyTracker, full_eval,
    build_frequency_tiers, win_rate_by_tier,
    quick_eval, EVAL_CHECKPOINTS,
)

EMOJI = {GREY: "⬛", YELLOW: "🟨", GREEN: "🟩"}
LR             = 3e-4
GAMMA          = 0.99
LAM            = 0.95
CLIP_EPS       = 0.2
ENTROPY_COEF   = 0.05
VF_COEF        = 0.5
N_EPOCHS       = 4
BATCH_SIZE     = 128
ROLLOUT_STEPS  = 1024
TOTAL_STEPS    = 300_000
LOG_INTERVAL   = 10
HIDDEN         = 256
DEVICE         = "cpu"
CKPT_DIR       = "checkpoints"
SHOW_GAME_EVERY= 500
TEST_FRACTION  = 0.1
SEED           = 42
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
        return dist.log_prob(a), dist.entropy(), 
@dataclass
class Buffer:
    states    : list = field(default_factory=list)
    actions   : list = field(default_factory=list)
    rewards   : list = field(default_factory=list)
    log_probs : list = field(default_factory=list)
    values    : list = field(default_factory=list)
    dones     : list = field(default_factory=list)
    masks     : list = field(default_factory=list)

    def add(self, s, a, r, lp, v, d, m):
        self.states.append(s);    self.actions.append(a)
        self.rewards.append(r);   self.log_probs.append(lp)
        self.values.append(v);    self.dones.append(d)
        self.masks.append(m)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)
def gae(rewards, values, dones, last_v, gamma, lam):
    T        = len(rewards)
    rewards  = np.array(rewards,  np.float32)
    values   = np.array(values,   np.float32)
    dones    = np.array(dones,    np.float32)
    adv      = np.zeros(T,        np.float32)
    g        = 0.0
    vals_ext = np.append(values, last_v)
    for t in reversed(range(T)):
        delta  = (rewards[t]
                  + gamma * vals_ext[t + 1] * (1 - dones[t])
                  - vals_ext[t])
        g      = delta + gamma * lam * (1 - dones[t]) * g
        adv[t] = g
    return adv + values, adv
def print_game(env: WordleEnv, episode_num: int):
    n   = len(env.guesses_made)
    won = bool(env.guesses_made) and env.guesses_made[-1] == env.target
    score = f"{n}/6" if won else "X/6"
    print(f"\nWordle {score}  [PPO | ep {episode_num:,}]")
    for guess, fb in zip(env.guesses_made, env.feedback_history):
        tiles = "".join(EMOJI[f] for f in fb)
        print(f"  {tiles}  {guess.upper()}")
    if won:
        print(f"  ✅  Answer: {env.target.upper()}  (solved in {n})")
    else:
        print(f"  ❌  Answer was: {env.target.upper()}")
    print()
def train():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    words, train_answers, test_answers = get_split_words(
        test_fraction=TEST_FRACTION, seed=SEED
    )

    train_env = WordleEnv(words, train_answers)
    eval_env  = WordleEnv(words, train_answers, test_answers=test_answers)

    model  = ActorCritic(train_env.state_dim, train_env.action_dim, HIDDEN).to(DEVICE)
    opt    = Adam(model.parameters(), lr=LR)
    buf    = Buffer()
    dev    = torch.device(DEVICE)
    logger = MetricsLogger("ppo")
    logger.log_hyperparams(
        lr=LR, gamma=GAMMA, lam=LAM, clip_eps=CLIP_EPS,
        entropy_coef=ENTROPY_COEF, vf_coef=VF_COEF,
        n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
        rollout_steps=ROLLOUT_STEPS, total_steps=TOTAL_STEPS,
        hidden=HIDDEN, test_fraction=TEST_FRACTION, seed=SEED,
        state_dim=train_env.state_dim, action_dim=train_env.action_dim,
    )

    entropy_tracker = EntropyTracker()
    ep_rewards, ep_wins              = [], []
    steps, updates, episodes         = 0, 0, 0
    next_ckpt_idx                    = 0
    state = train_env.reset()
    ep_r  = 0.0

    print("=" * 70)
    print("  PPO Wordle Trainer")
    print(f"  Vocab: {len(words):,} | Train answers: {len(train_answers):,} | "
          f"Test: {len(test_answers):,}")
    print(f"  rollout={ROLLOUT_STEPS} epochs={N_EPOCHS} clip={CLIP_EPS} "
          f"lr={LR} ent={ENTROPY_COEF}")
    print(f"  gamma={GAMMA} lambda={LAM} vf={VF_COEF}")
    print(f"  state_dim={train_env.state_dim} actions={train_env.action_dim}")
    print("=" * 70)

    t0 = time.time()

    while steps < TOTAL_STEPS:
        model.eval()
        with torch.no_grad():
            for _ in range(ROLLOUT_STEPS):
                step_num = len(train_env.guesses_made)
                s_t      = torch.tensor(
                    state, dtype=torch.float32, device=dev
                ).unsqueeze(0)

                raw_mask = train_env.get_valid_mask()
                mask_t   = torch.tensor(raw_mask, dtype=torch.bool, device=dev)
                a, lp, ent, v = model.act(s_t, mask_t.unsqueeze(0))
                entropy_tracker.record(step_num, ent.item())
                next_s, r, done, info = train_env.step(a.item())
                ep_r += r
                buf.add(state, a.item(), r, lp.item(), v.item(),
                        float(done), raw_mask)
                state = next_s
                if done:
                    episodes += 1
                    ep_rewards.append(ep_r)
                    ep_wins.append(int(info["won"]))
                    if episodes % SHOW_GAME_EVERY == 0:
                        print_game(train_env, episodes)
                    ep_r  = 0.0
                    state = train_env.reset()

            s_t   = torch.tensor(
                state, dtype=torch.float32, device=dev
            ).unsqueeze(0)
            _, lv = model(s_t)
            last_v = lv.item()

        returns, advs = gae(
            buf.rewards, buf.values, buf.dones, last_v, GAMMA, LAM
        )

        S     = torch.tensor(np.array(buf.states),  dtype=torch.float32, device=dev)
        A     = torch.tensor(buf.actions,            dtype=torch.long,    device=dev)
        LP    = torch.tensor(buf.log_probs,          dtype=torch.float32, device=dev)
        R     = torch.tensor(returns,                dtype=torch.float32, device=dev)
        ADV   = torch.tensor(advs,                   dtype=torch.float32, device=dev)
        MASKS = torch.tensor(np.array(buf.masks),    dtype=torch.bool,    device=dev)
        ADV   = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        N          = len(S)
        model.train()
        total_loss, n_batches = 0.0, 0

        for _ in range(N_EPOCHS):
            idx = torch.randperm(N, device=dev)
            for start in range(0, N, BATCH_SIZE):
                b              = idx[start: start + BATCH_SIZE]
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
        checkpoints = EVAL_CHECKPOINTS
        while (next_ckpt_idx < len(checkpoints)
               and episodes >= checkpoints[next_ckpt_idx]):
            train_wr = quick_eval(model, eval_env, dev, n=200, test=False)
            test_wr  = quick_eval(model, eval_env, dev, n=200, test=True)
            logger.log_checkpoint(episodes, train_wr, test_wr)
            print(
                f"  [CKPT] ep {episodes:>8,} | "
                f"train_wr {train_wr*100:.1f}% | test_wr {test_wr*100:.1f}%"
            )
            next_ckpt_idx += 1

        if updates % LOG_INTERVAL == 0:
            recent_wins = ep_wins[-1_000:]
            recent_r    = ep_rewards[-1_000:]
            wr       = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0
            avg      = sum(recent_r)    / len(recent_r)          if recent_r    else 0
            elapsed  = time.time() - t0
            sps      = steps / elapsed
            avg_loss = total_loss / max(1, n_batches)
            ent_g0   = np.mean(entropy_tracker.data.get(0, [0]))
            print(
                f"[PPO]  step {steps:>8,} | ep {episodes:>6,} | upd {updates:>4} | "
                f"win {wr:5.1f}% | avg_r {avg:+.3f} | "
                f"loss {avg_loss:.4f} | ent@g0 {ent_g0:.2f} | {sps:.0f} s/s"
            )
            entropy_tracker.reset()
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, "ppo.pt")
    torch.save({
        "model"         : model.state_dict(),
        "words"         : words,
        "train_answers" : train_answers,
        "test_answers"  : test_answers,
    }, ckpt_path)
    print(f"\n💾  Checkpoint saved → {ckpt_path}")
    print(f"⏱   Training time: {(time.time()-t0)/60:.1f} min")
    print("\n" + "=" * 70)
    print("  PPO Final Eval — 500 greedy games on HELD-OUT test answers")
    print("=" * 70)

    eval_results = full_eval(model, eval_env, dev, n=500, test=True, algo_name="ppo")
    logger.log_final_eval(eval_results)

    wr  = eval_results["win_rate"] * 100
    avg = eval_results["avg_guesses"]
    print(f"\n  Win rate: {wr:.1f}%  |  Avg guesses: {avg:.2f}")

    print("\n  Constraint reduction by guess position (mean fraction eliminated):")
    for i, r in enumerate(eval_results["constraint_reduction_by_pos"]):
        bar = "█" * int(r * 40)
        print(f"  Guess {i+1}  {bar:<40}  {r:.3f}")

    print("\n  Policy entropy by guess position:")
    for pos, ent in sorted(eval_results["entropy_by_pos"].items()):
        bar = "█" * int(ent / 2)
        print(f"  Guess {int(pos)+1}  {bar:<40}  {ent:.2f}")

    fg = eval_results["first_guess_stats"]
    print(f"\n  First-guess diversity — entropy: {fg['entropy']:.3f} | "
          f"unique: {fg['n_unique']} | collapse: {fg['collapse_flag']}")
    print(f"  Top-5 opening words: {fg['top_5']}")

    print("\n  Guess distribution:")
    hist = eval_results["guess_histogram"]
    for g in range(1, 7):
        bar = "█" * hist[g] if g < len(hist) else ""
        pct = hist[g] / 500 * 100 if g < len(hist) else 0
        print(f"  {g}/6  {bar:<40}  {hist[g] if g < len(hist) else 0:3d}  ({pct:.1f}%)")
    fail = hist[0]
    print(f"  X/6  {'░' * fail:<40}  {fail:3d}  ({fail/500*100:.1f}%)")
    print("\n  Win rate by word frequency tier (test set):")
    tiers = build_frequency_tiers(test_answers)
    tier_results = win_rate_by_tier(model, eval_env, dev, tiers, torch, test=True)
    logger.log_tier_results(tier_results)
    for tier_name, wr_t in tier_results.items():
        print(f"    {tier_name:<8}: {wr_t*100:.1f}%")
    train_final_wr = quick_eval(model, eval_env, dev, n=500, test=False)
    test_final_wr  = eval_results["win_rate"]
    gap = train_final_wr - test_final_wr
    print(f"\n  Generalisation — train: {train_final_wr*100:.1f}% | "
          f"test: {test_final_wr*100:.1f}% | gap: {gap*100:.1f}pp")

    logger.save(os.path.join(CKPT_DIR, "ppo_metrics.json"))
    print("\n  ── 5 sample greedy games (test answers) ──")
    for i in range(5):
        s = eval_env.reset(test=True)
        done = False
        with torch.no_grad():
            while not done:
                s_t    = torch.tensor(
                    s, dtype=torch.float32, device=dev
                ).unsqueeze(0)
                logits, _ = model(s_t)
                mask = torch.tensor(
                    eval_env.get_valid_mask(), dtype=torch.bool, device=dev
                )
                logits[0, ~mask] = -1e9
                act  = logits.argmax(dim=-1).item()
                s, _, done, _ = eval_env.step(act)
        print_game(eval_env, i)


if __name__ == "__main__":
    train()
