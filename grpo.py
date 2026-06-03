import os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
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
CLIP_EPS       = 0.2
ENTROPY_COEF   = 0.05
GROUP_SIZE     = 16
N_EPOCHS       = 4
BATCH_SIZE     = 128
EPS_PER_UPDATE = 64
TOTAL_EPISODES = 300_000
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
def print_game(env: WordleEnv, episode_num: int):
    n   = len(env.guesses_made)
    won = bool(env.guesses_made) and env.guesses_made[-1] == env.target
    score = f"{n}/6" if won else "X/6"
    print(f"\nWordle {score}  [GRPO | ep {episode_num:,}]")
    for guess, fb in zip(env.guesses_made, env.feedback_history):
        tiles = "".join(EMOJI[f] for f in fb)
        print(f"  {tiles}  {guess.upper()}")
    if won:
        print(f"GOOO NEPTUNE: {env.target.upper()}  (solved in {n})")
    else:
        print(f"OOPSIE DAISY :((: {env.target.upper()}")
    print()

def collect_groups(
    model, env: WordleEnv, dev, n_episodes: int, group_size: int,
    total_episodes: int, show_every: int,
    entropy_tracker: EntropyTracker,
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
                step_num = len(env.guesses_made)
                s_t = torch.tensor(
                    state, dtype=torch.float32, device=dev
                ).unsqueeze(0)

                with torch.no_grad():
                    logits, _ = model(s_t)
                mask = torch.tensor(
                    env.get_valid_mask(), dtype=torch.bool, device=dev
                )
                logits = logits.clone()
                logits[0, ~mask] = -1e9

                dist = torch.distributions.Categorical(logits=logits)
                a    = dist.sample()
                lp   = dist.log_prob(a).item()

                # Track entropy per guess position
                entropy_tracker.record(step_num, dist.entropy().item())

                state, r, done, info = env.step(a.item())
                traj_s.append(state)
                traj_a.append(a.item())
                traj_lp.append(lp)
                ep_r += r

            group_r.append(ep_r)
            group_traj.append((traj_s, traj_a, traj_lp))
            ep_wins.append(int(info["won"]))

        ep_total = total_episodes + ep_i * group_size
        if (ep_total % show_every) < group_size:
            show_env = env
            show_ep  = ep_total

        ep_rewards.extend(group_r)

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
    dev    = torch.device(DEVICE)
    logger = MetricsLogger("grpo")
    logger.log_hyperparams(
        lr=LR, clip_eps=CLIP_EPS, entropy_coef=ENTROPY_COEF,
        group_size=GROUP_SIZE, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
        eps_per_update=EPS_PER_UPDATE, total_episodes=TOTAL_EPISODES,
        hidden=HIDDEN, test_fraction=TEST_FRACTION, seed=SEED,
        state_dim=train_env.state_dim, action_dim=train_env.action_dim,
    )

    entropy_tracker = EntropyTracker()
    all_rewards, all_wins = [], []
    total_eps, updates    = 0, 0
    next_ckpt_idx         = 0

    print("=" * 70)
    print("  GRPO Wordle Trainer")
    print(f"  Vocab: {len(words):,} | Train answers: {len(train_answers):,} | "
          f"Test: {len(test_answers):,}")
    print(f"  group={GROUP_SIZE} eps/upd={EPS_PER_UPDATE} clip={CLIP_EPS} "
          f"lr={LR} ent={ENTROPY_COEF}")
    print(f"  state_dim={train_env.state_dim} actions={train_env.action_dim}")
    print("=" * 70)

    t0 = time.time()

    while total_eps < TOTAL_EPISODES:
        model.eval()
        S_np, A_np, LP_np, ADV_np, ep_r, ep_w, show_env, show_ep = collect_groups(
            model, train_env, dev,
            EPS_PER_UPDATE, GROUP_SIZE,
            total_eps, SHOW_GAME_EVERY,
            entropy_tracker,
        )

        if show_env is not None:
            print_game(show_env, show_ep)

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
        total_loss, n_batches = 0.0, 0
        for _ in range(N_EPOCHS):
            idx = torch.randperm(N, device=dev)
            for start in range(0, N, BATCH_SIZE):
                b         = idx[start: start + BATCH_SIZE]
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
        checkpoints = EVAL_CHECKPOINTS
        while (next_ckpt_idx < len(checkpoints)
               and total_eps >= checkpoints[next_ckpt_idx]):
            train_wr = quick_eval(model, eval_env, dev, n=200, test=False)
            test_wr  = quick_eval(model, eval_env, dev, n=200, test=True)
            logger.log_checkpoint(total_eps, train_wr, test_wr)
            print(
                f"  [CKPT] ep {total_eps:>8,} | "
                f"train_wr {train_wr*100:.1f}% | test_wr {test_wr*100:.1f}%"
            )
            next_ckpt_idx += 1

        if updates % LOG_INTERVAL == 0:
            recent_wins = all_wins[-1_000:]
            recent_r    = all_rewards[-1_000:]
            wr       = sum(recent_wins) / len(recent_wins) * 100
            avg      = sum(recent_r)    / len(recent_r)
            elapsed  = time.time() - t0
            eps_s    = total_eps / elapsed
            avg_loss = total_loss / max(1, n_batches)
            ent_g0 = np.mean(entropy_tracker.data.get(0, [0]))
            print(
                f"[GRPO] ep {total_eps:>8,} | upd {updates:>4} | "
                f"win {wr:5.1f}% | avg_r {avg:+.3f} | "
                f"loss {avg_loss:.4f} | ent@g0 {ent_g0:.2f} | {eps_s:.0f} ep/s"
            )
            entropy_tracker.reset()
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, "grpo.pt")
    torch.save({
        "model"         : model.state_dict(),
        "words"         : words,
        "train_answers" : train_answers,
        "test_answers"  : test_answers,
    }, ckpt_path)
    print(f"\n💾  Checkpoint saved → {ckpt_path}")
    print(f"⏱   Training time: {(time.time()-t0)/60:.1f} min")
    print("\n" + "=" * 70)
    print("  GRPO Final Eval — 500 greedy games on HELD-OUT test answers")
    print("=" * 70)

    eval_results = full_eval(model, eval_env, dev, n=500, test=True, algo_name="grpo")
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

    logger.save(os.path.join(CKPT_DIR, "grpo_metrics.json"))
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
