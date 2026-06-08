import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import parse_args, ExperimentConfig
from words import get_split_words
from env import WordleEnv, GREEN, YELLOW, GREY
from model import build_model
from adversary import WordAdversary
from metrics import MetricsLogger, EntropyTracker, full_eval, build_frequency_tiers, win_rate_by_tier, quick_eval, EVAL_CHECKPOINTS

EMOJI = {GREY: "⬛", YELLOW: "🟨", GREEN: "🟩"}


def print_game(env, episode_num):
    won = bool(env.guesses_made) and env.guesses_made[-1] == env.target
    display = "%d/6" % len(env.guesses_made) if won else "X/6"
    print("\nWordle %s  [GRPO | ep %s]" % (display, format(episode_num, ",")))
    for guess, feedback in zip(env.guesses_made, env.feedback_history):
        print("  %s  %s" % ("".join(EMOJI[f] for f in feedback), guess.upper()))
    if not won:
        print("  FAILED  Answer was: %s" % env.target.upper())


def collect_groups(model, env, device, cfg, entropy_tracker, adversary=None, answer_indices=None):
    all_states, all_actions, all_logps, all_advantages = [], [], [], []
    episode_rewards, episode_wins = [], []
    chosen_adv_indices, adv_rewards = [], []

    for _ in range(cfg.eps_per_update):
        if adversary is not None and answer_indices:
            target_indices = adversary.sample_targets(answer_indices, cfg.group_size, temperature=1.0)
            targets = [env.words[i] for i in target_indices]
        else:
            targets = [random.choice(env.answer_list) for _ in range(cfg.group_size)]
            target_indices = [env.word_to_idx.get(t, 0) for t in targets]

        group_rewards = []
        group_trajectories = []
        for group_idx in range(cfg.group_size):
            state = env.reset(target=targets[group_idx % len(targets)])
            traj_states, traj_actions, traj_logps = [], [], []
            episode_reward = 0.0
            done = False
            while not done:
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = model(s_t)
                mask = torch.tensor(env.get_valid_mask(), dtype=torch.bool, device=device)
                logits = logits.masked_fill(~mask, -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                entropy_tracker.record(len(env.guesses_made), dist.entropy().item())
                state, reward, done, info = env.step(action.item())
                traj_states.append(state)
                traj_actions.append(action.item())
                traj_logps.append(dist.log_prob(action).item())
                episode_reward += reward
            group_rewards.append(episode_reward)
            group_trajectories.append((traj_states, traj_actions, traj_logps))
            episode_wins.append(int(info["won"]))
            if adversary is not None:
                chosen_adv_indices.append(target_indices[group_idx % len(target_indices)])
                adv_rewards.append(episode_reward)

        episode_rewards.extend(group_rewards)
        rewards = np.array(group_rewards, dtype=np.float32)
        rewards = rewards - rewards.mean() if rewards.std() < 1e-8 else (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for idx, (traj_states, traj_actions, traj_logps) in enumerate(group_trajectories):
            adv = float(rewards[idx])
            for state, action, logp in zip(traj_states, traj_actions, traj_logps):
                all_states.append(state)
                all_actions.append(action)
                all_logps.append(logp)
                all_advantages.append(adv)

    return (
        np.array(all_states, dtype=np.float32),
        np.array(all_actions, dtype=np.int64),
        np.array(all_logps, dtype=np.float32),
        np.array(all_advantages, dtype=np.float32),
        episode_rewards,
        episode_wins,
        chosen_adv_indices,
        adv_rewards,
    )


def _print_eval_summary(results):
    print("\n  Win rate: %.1f%%  |  Avg guesses: %.2f" % (results["win_rate"] * 100, results["avg_guesses"] or 0))
    print("\n  Constraint reduction per guess:")
    for idx, value in enumerate(results["constraint_reduction_by_pos"]):
        print("    Guess %d  %s  %.3f" % (idx + 1, "█" * int(value * 40), value))
    print("\n  Policy entropy per guess:")
    for pos, value in sorted(results["entropy_by_pos"].items(), key=lambda x: int(x[0])):
        print("    Guess %d  %s  %.2f" % (int(pos) + 1, "█" * int(value / 2), value))
    fg = results["first_guess_stats"]
    print("\n  First-guess entropy: %.3f | unique: %d | collapse: %s" % (fg["entropy"], fg["n_unique"], fg["collapse_flag"]))


def train(cfg=None):
    if cfg is None:
        cfg = parse_args(algo="grpo")
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    words, train_answers, test_answers = get_split_words(
        test_fraction=cfg.test_fraction, split_seed=cfg.split_seed
    )
    train_env = WordleEnv(words, train_answers, cfg=cfg)
    test_env = WordleEnv(words, test_answers, cfg=cfg)
    model = build_model(cfg, train_env.state_dim, train_env.action_dim)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    device = torch.device(cfg.device)
    logger = MetricsLogger(cfg.algo, cfg.run_name)
    logger.log_hyperparams(**{k: v for k, v in cfg.__dict__.items() if not k.startswith("_")})

    answer_indices = [train_env.word_to_idx[w] for w in train_answers if w in train_env.word_to_idx]
    adversary = WordAdversary(answer_indices)
    adv_optimizer = Adam(adversary.parameters(), lr=1e-3)
    entropy_tracker = EntropyTracker()

    all_rewards, all_wins = [], []
    total_episodes = 0
    updates = 0
    next_ckpt_idx = 0

    print("=" * 70)
    print("  GRPO | run=%s | group=%d | reward=%s | attn=%s" % (cfg.run_name, cfg.group_size, cfg.reward_type, cfg.use_attention))
    print("  train=%d | test=%d | vocab=%d | state=%d" % (len(train_answers), len(test_answers), len(words), train_env.state_dim))
    print("=" * 70)
    start_time = time.time()

    while total_episodes < cfg.total_episodes:
        model.eval()
        states_np, actions_np, logps_np, adv_np, ep_rewards, ep_wins, adv_idx, adv_rew = collect_groups(
            model, train_env, device, cfg, entropy_tracker, adversary=adversary, answer_indices=answer_indices
        )
        if adv_idx:
            adversary.update(adv_optimizer, adv_idx, adv_rew)
        all_rewards.extend(ep_rewards)
        all_wins.extend(ep_wins)
        total_episodes += cfg.eps_per_update * cfg.group_size
        updates += 1

        states = torch.tensor(states_np, dtype=torch.float32, device=device)
        actions = torch.tensor(actions_np, dtype=torch.long, device=device)
        old_logps = torch.tensor(logps_np, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(adv_np, dtype=torch.float32, device=device)

        model.train()
        total_loss = 0.0
        batch_count = 0
        for _ in range(cfg.n_epochs):
            order = torch.randperm(len(states), device=device)
            for start in range(0, len(states), cfg.batch_size):
                batch_idx = order[start:start + cfg.batch_size]
                logits, _ = model(states[batch_idx])  
                dist = torch.distributions.Categorical(logits=logits)
                new_logps = dist.log_prob(actions[batch_idx])
                pg_loss = -(new_logps * advantages_t[batch_idx]).mean()
                entropy_loss = -cfg.entropy_coef * dist.entropy().mean()
                kl = (old_logps[batch_idx] - new_logps).mean()
                loss = pg_loss + entropy_loss + 0.01 * kl
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1

        while next_ckpt_idx < len(EVAL_CHECKPOINTS) and total_episodes >= EVAL_CHECKPOINTS[next_ckpt_idx]:
            train_wr = quick_eval(model, train_env, device)
            test_wr = quick_eval(model, test_env, device)
            logger.log_checkpoint(total_episodes, train_wr, test_wr)
            print("  [CKPT] ep %d | train %.1f%% | test %.1f%%" % (total_episodes, train_wr * 100, test_wr * 100))
            next_ckpt_idx += 1

        if updates % cfg.log_interval == 0:
            recent_wins = all_wins[-1000:]
            recent_rewards = all_rewards[-1000:]
            win_rate = sum(recent_wins) / max(1, len(recent_wins)) * 100
            avg_reward = sum(recent_rewards) / max(1, len(recent_rewards))
            entropy_g0 = float(np.mean(entropy_tracker.data.get(0, [0])))
            elapsed = time.time() - start_time
            print("[GRPO] ep %d | upd %d | win %.1f%% | avg_r %.3f | loss %.4f | ent@g0 %.2f | %d ep/s" % (
                total_episodes, updates, win_rate, avg_reward,
                total_loss / max(1, batch_count), entropy_g0, int(total_episodes / elapsed)
            ))
            entropy_tracker.reset()

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_path = cfg.ckpt_path(".pt")
    torch.save({"model": model.state_dict(), "words": words, "train_answers": train_answers,
                "test_answers": test_answers, "cfg": cfg.__dict__}, ckpt_path)
    print("\nCheckpoint saved → %s" % ckpt_path)
    print("Training time: %.1f min" % ((time.time() - start_time) / 60.0))

    print("\n" + "=" * 70)
    print("  GRPO Final Eval — exhaustive greedy on held-out test answers")
    print("=" * 70)
    results = full_eval(model, test_env, device, algo_name="grpo")
    logger.log_final_eval(results)
    _print_eval_summary(results)

    tiers = build_frequency_tiers(test_answers)
    tier_results = win_rate_by_tier(model, test_env, device, tiers)
    logger.log_tier_results(tier_results)
    print("\n  Win rate by frequency tier:")
    for tier_name, wr in tier_results.items():
        print("    %s: %.1f%%" % (tier_name, wr * 100))

    train_wr = quick_eval(model, train_env, device)
    test_wr = results["win_rate"]
    print("\n  Generalisation — train: %.1f%% | test: %.1f%% | gap: %.1fpp" % (
        train_wr * 100, test_wr * 100, (train_wr - test_wr) * 100
    ))

    logger.save(cfg.ckpt_path("_metrics.json"))
    return model, logger


if __name__ == "__main__":
    train()
