import os
import json
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from wordfreq import word_frequency
    WORDFREQ_AVAILABLE = True
except ImportError:
    WORDFREQ_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from env import WordleEnv, is_consistent

EVAL_CHECKPOINTS = [2000, 5000, 10000, 20000, 50000, 100000, 200000, 300000]


def constraint_reduction_rates(env):
    rates = []
    prev_pool = list(env.answer_list)
    for step_idx in range(len(env.guesses_made)):
        new_pool = [
            w for w in prev_pool
            if is_consistent(w, env.guesses_made[:step_idx + 1], env.feedback_history[:step_idx + 1])
        ]
        rates.append(float((len(prev_pool) - len(new_pool)) / max(1, len(prev_pool))))
        prev_pool = new_pool
    return rates


def mean_constraint_reduction(all_rates, max_steps=6):
    acc = np.zeros(max_steps)
    count = np.zeros(max_steps, dtype=int)
    for rates in all_rates:
        for i, r in enumerate(rates):
            if i < max_steps:
                acc[i] += r
                count[i] += 1
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, acc / count, 0.0)


@dataclass
class EntropyTracker:
    data: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))

    def record(self, pos, val):
        self.data[pos].append(val)

    def reset(self):
        self.data = defaultdict(list)


def first_guess_stats(first_guesses):
    counts = Counter(first_guesses)
    n = max(1, len(first_guesses))
    probs = np.array([c / n for c in counts.values()])
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    top_5 = counts.most_common(5)
    top1_frac = top_5[0][1] / n if top_5 else 0.0
    return {
        "entropy": round(entropy, 4),
        "top_5": top_5,
        "n_unique": len(counts),
        "n_episodes": n,
        "collapse_flag": top1_frac > 0.5,
        "top1_fraction": round(top1_frac, 4),
    }


def build_frequency_tiers(answers, n_tiers=3):
    tier_names = ["common", "medium", "rare"]
    scored = sorted(answers, key=lambda w: word_frequency(w, "en"), reverse=True) if WORDFREQ_AVAILABLE else sorted(answers)
    n = len(scored)
    size = n // n_tiers
    return {
        name: scored[i * size: (i + 1) * size if i < n_tiers - 1 else n]
        for i, name in enumerate(tier_names)
    }


def win_rate_by_tier(model, test_env, dev, tiers):
    import torch
    model.eval()
    results = {}
    pool = set(test_env.answer_list)
    with torch.no_grad():
        for tier_name, word_list in tiers.items():
            wins, total = 0, 0
            for target in word_list:
                if target not in pool:
                    continue
                s, done = test_env.reset(target=target), False
                while not done:
                    s_t = torch.tensor(s, dtype=torch.float32, device=dev).unsqueeze(0)
                    logits, _ = model(s_t)
                    mask = torch.tensor(test_env.get_valid_mask(), dtype=torch.bool, device=dev)
                    logits[0, ~mask] = -1e9
                    s, _, done, info = test_env.step(logits.argmax(dim=-1).item())
                wins += int(info["won"])
                total += 1
            results[tier_name] = round(wins / max(1, total), 4)
    return results


def quick_eval(model, test_env, dev):
    import torch
    model.eval()
    wins = 0
    targets = list(test_env.answer_list)
    with torch.no_grad():
        for target in targets:
            s, done = test_env.reset(target=target), False
            while not done:
                s_t = torch.tensor(s, dtype=torch.float32, device=dev).unsqueeze(0)
                logits, _ = model(s_t)
                mask = torch.tensor(test_env.get_valid_mask(), dtype=torch.bool, device=dev)
                logits[0, ~mask] = -1e9
                s, _, done, info = test_env.step(logits.argmax(dim=-1).item())
            wins += int(info["won"])
    return wins / len(targets)


def full_eval(model, test_env, dev, algo_name="agent"):
    import torch
    import torch.nn.functional as F
    model.eval()
    wins, scores = 0, []
    guess_hist = [0] * 8
    first_guesses = []
    all_csr = []
    entropy_by_pos = defaultdict(list)
    targets = list(test_env.answer_list)
    with torch.no_grad():
        for target in targets:
            s, done = test_env.reset(target=target), False
            while not done:
                step_num = len(test_env.guesses_made)
                s_t = torch.tensor(s, dtype=torch.float32, device=dev).unsqueeze(0)
                logits, _ = model(s_t)
                mask = torch.tensor(test_env.get_valid_mask(), dtype=torch.bool, device=dev)
                masked = logits.clone()
                masked[0, ~mask] = -1e9
                probs = F.softmax(masked, dim=-1)
                entropy_by_pos[step_num].append(-(probs * torch.log(probs + 1e-12)).sum().item())
                s, _, done, info = test_env.step(masked.argmax(dim=-1).item())
            if test_env.guesses_made:
                first_guesses.append(test_env.guesses_made[0])
            ng = info["n_guesses"]
            if info["won"]:
                wins += 1
                scores.append(ng)
                if ng < len(guess_hist):
                    guess_hist[ng] += 1
            else:
                guess_hist[0] += 1
            all_csr.append(constraint_reduction_rates(test_env))
    n = len(targets)
    return {
        "algo": algo_name,
        "n_eval": n,
        "win_rate": round(wins / n, 4),
        "avg_guesses": round(float(np.mean(scores)), 4) if scores else None,
        "guess_histogram": guess_hist,
        "constraint_reduction_by_pos": mean_constraint_reduction(all_csr).tolist(),
        "entropy_by_pos": {k: float(np.mean(v)) for k, v in entropy_by_pos.items()},
        "first_guess_stats": first_guess_stats(first_guesses),
    }


def hypothesis_test(ppo_g1_rates, grpo_g1_rates, alpha=0.05):
    if not SCIPY_AVAILABLE:
        return {"error": "scipy not available"}
    ppo_arr = np.array(ppo_g1_rates, dtype=float)
    grpo_arr = np.array(grpo_g1_rates, dtype=float)
    t_stat, p_two = scipy_stats.ttest_ind(grpo_arr, ppo_arr, equal_var=False)
    p_one = p_two / 2 if t_stat > 0 else 1.0
    pooled_std = np.sqrt((ppo_arr.std() ** 2 + grpo_arr.std() ** 2) / 2)
    cohens_d = (grpo_arr.mean() - ppo_arr.mean()) / (pooled_std + 1e-12)
    abs_d = abs(cohens_d)
    effect_size = "large" if abs_d >= 0.8 else ("medium" if abs_d >= 0.5 else "small")
    return {
        "ppo_mean": round(float(ppo_arr.mean()), 4),
        "grpo_mean": round(float(grpo_arr.mean()), 4),
        "ppo_std": round(float(ppo_arr.std()), 4),
        "grpo_std": round(float(grpo_arr.std()), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value_one_tailed": round(float(p_one), 4),
        "p_value": round(float(p_one), 4),
        "cohens_d": round(float(cohens_d), 4),
        "effect_size": effect_size,
        "significant": bool(p_one < alpha),
        "alpha": alpha,
    }


class MetricsLogger:
    def __init__(self, algo, run_name="default"):
        self.algo = algo
        self.run_name = run_name
        self.sample_efficiency: List[Dict] = []
        self.final_eval: Optional[Dict] = None
        self.tier_results: Optional[Dict] = None
        self.hyperparams: Dict = {}
        self.adversary_ranking: Optional[List] = None

    def log_checkpoint(self, episodes, train_wr, test_wr):
        self.sample_efficiency.append({"episodes": episodes, "train_wr": round(train_wr, 4), "test_wr": round(test_wr, 4)})

    def log_final_eval(self, d): self.final_eval = d
    def log_tier_results(self, d): self.tier_results = d
    def log_hyperparams(self, **kwargs): self.hyperparams = kwargs
    def log_adversary_ranking(self, ranking): self.adversary_ranking = ranking

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "algo": self.algo,
                "run_name": self.run_name,
                "hyperparams": self.hyperparams,
                "sample_efficiency": self.sample_efficiency,
                "final_eval": self.final_eval,
                "tier_results": self.tier_results,
                "adversary_ranking": self.adversary_ranking,
            }, f, indent=2)
