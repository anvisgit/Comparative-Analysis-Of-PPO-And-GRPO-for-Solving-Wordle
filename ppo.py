import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import parse_args, ExperimentConfig
from words import get_split_words
from env import WordleEnv, GREEN, YELLOW, GREY
from model import build_model
from metrics import MetricsLogger, EntropyTracker, full_eval, build_frequency_tiers, win_rate_by_tier, quick_eval, EVAL_CHECKPOINTS

EMOJI = {GREY: "⬛", YELLOW: "🟨", GREEN: "🟩"}


class Buffer:
    def __init__(self):
        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.dones, self.masks = [], [], [], []

    def add(self, state, action, reward, logp, value, done, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(logp)
        self.values.append(value)
        self.dones.append(done)
        self.masks.append(mask)

    def clear(self): self.__init__()
    def __len__(self): return len(self.states)


def gae(rewards, values, dones, last_value, gamma, lam):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    values_ext = np.append(values, last_value)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_ext[t + 1] * (1.0 - dones[t]) - values_ext[t]
        last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages + values, advantages


def print_game(env, episode_num):
    won = bool(env.guesses_made) and env.guesses_made[-1] == env.target
    display = "%d/6" % len(env.guesses_made) if won else "X/6"
    print("\nWordle %s  [PPO | ep %s]" % (display, format(episode_num, ",")))
    for guess, feedback in zip(env.guesses_made, env.feedback_history):
        print("  %s  %s" % ("".join(EMOJI[f] for f in feedback), guess.upper()))
    if not won:
        print("  FAILED  Answer was: %s" % env.target.upper())


def _print_eval_summary(results):
    print("\n  Win rate: %.1f%%  |  Avg guesses: %.2f" % (results["win_rate"] * 100, results["avg_guesses"]))
    print("\n  Constraint reduction per guess:")
    for idx, value in enumerate(results["constraint_reduction_by_pos"]):
        print("    Guess %d  %s  %.3f" % (idx + 1, "█" * int(value * 40), value))
    print("\n  Policy entropy per guess:")
    for pos, value in sorted(results["entropy_by_pos"].items(), key=lambda x: int(x[0])):
        print("    Guess %d  %s  %.2f" % (int(pos) + 1, "█" * int(value / 2), value))
    fg = results["first_guess_stats"]
    print("\n  First-guess entropy: %.3f | unique: %d | collapse: %s" % (fg["entropy"], fg["n_unique"], fg["collapse_flag"]))


def train(cfg: ExperimentConfig = None):
    if cfg is None:
        cfg = parse_args(algo="ppo")
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    words, train_answers, test_answers = get_split_words(
        test_fraction=cfg.test_fraction, seed=cfg.seed, vocab_type=cfg.vocab_type
    )
    train_env = WordleEnv(words, train_answers, cfg=cfg)
    eval_env = WordleEnv(words, train_answers, test_answers=test_answers, cfg=cfg)
    model = build_model(cfg, train_env.state_dim, train_env.action_dim)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    buffer = Buffer()
    device = torch.device(cfg.device)
    logger = MetricsLogger(cfg.algo, cfg.run_name)
    logger.log_hyperparams(**{k: v for k, v in cfg.__dict__.items() if not k.startswith("_")})
    entropy_tracker = EntropyTracker()

    episode_rewards, episode_wins = [], []
    steps, updates, episodes = 0, 0, 0
    next_ckpt_idx = 0
    state = train_env.reset()
    episode_reward = 0.0

    print("=" * 70)
    print("  PPO | run=%s | rollout=%d | reward=%s | attn=%s" % (cfg.run_name, cfg.rollout_steps, cfg.reward_type, cfg.use_attention))
    print("  train=%d | test=%d | vocab=%d | state=%d" % (len(train_answers), len(test_answers), len(words), train_env.state_dim))
    print("=" * 70)
    start_time = time.time()

    while steps < cfg.total_steps:
        model.eval()
        with torch.no_grad():
            for _ in range(cfg.rollout_steps):
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                mask = torch.tensor(train_env.get_valid_mask(), dtype=torch.bool, device=device)
                action, log_prob, entropy, value = model.act(s_t, mask.unsqueeze(0))
                entropy_tracker.record(len(train_env.guesses_made), entropy.item())
                next_state, reward, done, info = train_env.step(action.item())
                episode_reward += reward
                buffer.add(state, action.item(), reward, log_prob.item(), value.item(), float(done), train_env.get_valid_mask())
                state = next_state
                if done:
                    episodes += 1
                    episode_rewards.append(episode_reward)
                    episode_wins.append(int(info["won"]))
                    if episodes % cfg.show_game_every == 0:
                        print_game(train_env, episodes)
                    episode_reward = 0.0
                    state = train_env.reset()
            _, last_value = model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
            last_value = last_value.item()

        returns, advantages = gae(buffer.rewards, buffer.values, buffer.dones, last_value, cfg.gamma, cfg.lam)
        states = torch.tensor(np.array(buffer.states), dtype=torch.float32, device=device)
        actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        masks_t = torch.tensor(np.array(buffer.masks), dtype=torch.bool, device=device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        model.train()
        total_loss = 0.0
        batch_count = 0
        for _ in range(cfg.n_epochs):
            order = torch.randperm(len(states), device=device)
            for start in range(0, len(states), cfg.batch_size):
                batch_idx = order[start:start + cfg.batch_size]
                new_log_probs, entropy, values = model.evaluate(states[batch_idx], actions[batch_idx], masks_t[batch_idx])
                ratio = (new_log_probs - old_log_probs[batch_idx]).exp()
                surr1 = ratio * advantages_t[batch_idx]
                surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages_t[batch_idx]
                loss = (-torch.min(surr1, surr2).mean()
                        + cfg.vf_coef * F.mse_loss(values, returns_t[batch_idx])
                        - cfg.entropy_coef * entropy.mean())
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1

        buffer.clear()
        steps += cfg.rollout_steps
        updates += 1

        while next_ckpt_idx < len(EVAL_CHECKPOINTS) and episodes >= EVAL_CHECKPOINTS[next_ckpt_idx]:
            train_wr = quick_eval(model, eval_env, device, n=200, test=False)
            test_wr = quick_eval(model, eval_env, device, n=200, test=True)
            logger.log_checkpoint(episodes, train_wr, test_wr)
            print("  [CKPT] ep %d | train %.1f%% | test %.1f%%" % (episodes, train_wr * 100, test_wr * 100))
            next_ckpt_idx += 1

        if updates % cfg.log_interval == 0:
            recent_wins = episode_wins[-1000:]
            recent_rewards = episode_rewards[-1000:]
            win_rate = sum(recent_wins) / max(1, len(recent_wins)) * 100
            avg_reward = sum(recent_rewards) / max(1, len(recent_rewards))
            entropy_g0 = float(np.mean(entropy_tracker.data.get(0, [0])))
            elapsed = time.time() - start_time
            print("[PPO]  step %d | ep %d | upd %d | win %.1f%% | avg_r %.3f | loss %.4f | ent@g0 %.2f | %d s/s" % (
                steps, episodes, updates, win_rate, avg_reward,
                total_loss / max(1, batch_count), entropy_g0, int(steps / elapsed)
            ))
            entropy_tracker.reset()

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_path = cfg.ckpt_path(".pt")
    torch.save({"model": model.state_dict(), "words": words, "train_answers": train_answers,
                "test_answers": test_answers, "cfg": cfg.__dict__}, ckpt_path)
    print("\nCheckpoint saved → %s" % ckpt_path)
    print("Training time: %.1f min" % ((time.time() - start_time) / 60.0))

    print("\n" + "=" * 70)
    print("  PPO Final Eval — 500 greedy games on held-out test answers")
    print("=" * 70)
    results = full_eval(model, eval_env, device, n=500, test=True, algo_name="ppo")
    logger.log_final_eval(results)
    _print_eval_summary(results)

    tiers = build_frequency_tiers(test_answers)
    tier_results = win_rate_by_tier(model, eval_env, device, tiers, test=True)
    logger.log_tier_results(tier_results)
    print("\n  Win rate by frequency tier:")
    for tier_name, wr in tier_results.items():
        print("    %s: %.1f%%" % (tier_name, wr * 100))

    train_wr = quick_eval(model, eval_env, device, n=500, test=False)
    test_wr = results["win_rate"]
    print("\n  Generalisation — train: %.1f%% | test: %.1f%% | gap: %.1fpp" % (
        train_wr * 100, test_wr * 100, (train_wr - test_wr) * 100
    ))

    logger.save(cfg.ckpt_path("_metrics.json"))
    return model, logger


if __name__ == "__main__":
    train()
