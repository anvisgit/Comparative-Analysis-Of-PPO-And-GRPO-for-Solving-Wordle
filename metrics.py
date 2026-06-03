import json
import os
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    from wordfreq import word_frequency
    WORDFREQ_AVAILABLE = True
except ImportError:
    WORDFREQ_AVAILABLE = False

from env import WordleEnv, is_consistent, compute_feedback, GREEN
def constraint_reduction_rates(env: WordleEnv) -> List[float]:
    rates     = []
    prev_pool = list(env.answer_list)

    for step_idx, (guess, feedback) in enumerate(
        zip(env.guesses_made, env.feedback_history)
    ):
        guesses_so_far   = env.guesses_made[: step_idx + 1]
        feedbacks_so_far = env.feedback_history[: step_idx + 1]

        new_pool = [
            w for w in prev_pool
            if is_consistent(w, guesses_so_far, feedbacks_so_far)
        ]

        eliminated  = len(prev_pool) - len(new_pool)
        rate        = eliminated / max(1, len(prev_pool))
        rates.append(float(rate))
        prev_pool   = new_pool

    return rates


def mean_constraint_reduction(
    all_rates: List[List[float]], max_steps: int = 6
) -> np.ndarray:
    acc   = np.zeros(max_steps)
    count = np.zeros(max_steps, dtype=int)
    for rates in all_rates:
        for i, r in enumerate(rates):
            if i < max_steps:
                acc[i]   += r
                count[i] += 1
    with np.errstate(invalid="ignore"):
        result = np.where(count > 0, acc / count, 0.0)
    return result
@dataclass
class EntropyTracker:
    data: Dict[int, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def record(self, guess_pos: int, entropy: float):
        self.data[guess_pos].append(entropy)

    def mean_by_position(self, max_steps: int = 6) -> np.ndarray:
        result = np.zeros(max_steps)
        for i in range(max_steps):
            vals = self.data.get(i, [])
            result[i] = float(np.mean(vals)) if vals else 0.0
        return result

    def reset(self):
        self.data = defaultdict(list)

def first_guess_stats(first_guesses: List[str]) -> Dict:
    counts = Counter(first_guesses)
    n      = len(first_guesses)

    probs   = np.array([c / n for c in counts.values()])
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    top_5 = counts.most_common(5)
    top1_frac = top_5[0][1] / n if top_5 else 0.0

    return {
        "entropy"       : round(entropy, 4),
        "top_5"         : top_5,
        "n_unique"      : len(counts),
        "n_episodes"    : n,
        "collapse_flag" : top1_frac > 0.5,
        "top1_fraction" : round(top1_frac, 4),
    }

def build_frequency_tiers(
    answers: List[str], n_tiers: int = 3
) -> Dict[str, List[str]]:
    tier_names = ["common", "medium", "rare"]

    if WORDFREQ_AVAILABLE:
        scored = sorted(
            answers,
            key=lambda w: word_frequency(w, "en"),
            reverse=True,
        )
    else:
        print("[metrics] wordfreq not available — using alphabetical proxy split.")
        scored = sorted(answers)

    n = len(scored)
    size = n // n_tiers
    tiers = {}
    for i, name in enumerate(tier_names):
        start = i * size
        end   = (i + 1) * size if i < n_tiers - 1 else n
        tiers[name] = scored[start:end]

    return tiers


def win_rate_by_tier(
    model,
    env: WordleEnv,
    dev,
    tiers: Dict[str, List[str]],
    torch_module,
    test: bool = True,
) -> Dict[str, float]:
    import torch

    model.eval()
    results = {}

    for tier_name, word_list in tiers.items():
        wins = 0
        total = 0
        with torch.no_grad():
            for target in word_list:
              if test and target not in (env.test_answers or []):
                    continue
                if not test and target not in env.answer_list:
                    continue

                s    = env.reset(target=target)
                done = False
                while not done:
                    s_t    = torch.tensor(
                        s, dtype=torch.float32, device=dev
                    ).unsqueeze(0)
                    logits, _ = model(s_t)
                    mask = torch.tensor(
                        env.get_valid_mask(), dtype=torch.bool, device=dev
                    )
                    logits[0, ~mask] = -1e9
                    act  = logits.argmax(dim=-1).item()
                    s, _, done, info = env.step(act)

                if info["won"]:
                    wins += 1
                total += 1

        results[tier_name] = round(wins / max(1, total), 4)

    return results
EVAL_CHECKPOINTS = [
    1_000, 5_000, 10_000, 25_000, 50_000,
    100_000, 150_000, 200_000, 250_000, 300_000,
]


def quick_eval(
    model,
    env: WordleEnv,
    dev,
    n: int = 200,
    test: bool = True,
) -> float:
    import torch

    model.eval()
    wins = 0
    with torch.no_grad():
        for _ in range(n):
            s    = env.reset(test=test)
            done = False
            while not done:
                s_t    = torch.tensor(
                    s, dtype=torch.float32, device=dev
                ).unsqueeze(0)
                logits, _ = model(s_t)
                mask = torch.tensor(
                    env.get_valid_mask(), dtype=torch.bool, device=dev
                )
                logits[0, ~mask] = -1e9
                act  = logits.argmax(dim=-1).item()
                s, _, done, info = env.step(act)
            wins += int(info["won"])
    return wins / n
def full_eval(
    model,
    env: WordleEnv,
    dev,
    n: int = 500,
    test: bool = True,
    algo_name: str = "agent",
) -> Dict:
    import torch

    model.eval()
    wins, scores   = 0, []
    guess_hist     = [0] * (env.action_dim and 8)  
    first_guesses  = []
    all_csr        = []          
    entropy_by_pos = defaultdict(list)

    with torch.no_grad():
        for _ in range(n):
            s    = env.reset(test=test)
            done = False

            while not done:
                step_num = len(env.guesses_made)
                s_t      = torch.tensor(
                    s, dtype=torch.float32, device=dev
                ).unsqueeze(0)
                logits, _ = model(s_t)
                mask = torch.tensor(
                    env.get_valid_mask(), dtype=torch.bool, device=dev
                )
                masked_logits = logits.clone()
                masked_logits[0, ~mask] = -1e9
                import torch.nn.functional as F
                probs   = F.softmax(masked_logits, dim=-1)
                log_p   = torch.log(probs + 1e-12)
                entropy = -(probs * log_p).sum().item()
                entropy_by_pos[step_num].append(entropy)

                act  = masked_logits.argmax(dim=-1).item()
                s, _, done, info = env.step(act)
            if env.guesses_made:
                first_guesses.append(env.guesses_made[0])

            ng = info["n_guesses"]
            if info["won"]:
                wins += 1
                scores.append(ng)
                if ng < len(guess_hist):
                    guess_hist[ng] += 1
            else:
                guess_hist[0] += 1  
            all_csr.append(constraint_reduction_rates(env))
    mean_csr = mean_constraint_reduction(all_csr, max_steps=6).tolist()
    mean_ent = {
        k: float(np.mean(v)) for k, v in entropy_by_pos.items()
    }
    fg_stats = first_guess_stats(first_guesses)

    return {
        "algo"               : algo_name,
        "n_eval"             : n,
        "win_rate"           : round(wins / n, 4),
        "avg_guesses"        : round(float(np.mean(scores)), 4) if scores else None,
        "guess_histogram"    : guess_hist,
        "constraint_reduction_by_pos" : mean_csr,
        "entropy_by_pos"     : mean_ent,
        "first_guess_stats"  : fg_stats,
    }
class MetricsLogger:
    def __init__(self, algo: str):
        self.algo = algo
        self.sample_efficiency: List[Dict] = []   
        self.final_eval: Optional[Dict]    = None
        self.tier_results: Optional[Dict]  = None
        self.hyperparams: Dict             = {}

    def log_checkpoint(
        self,
        episodes: int,
        train_wr: float,
        test_wr: float,
    ):
        self.sample_efficiency.append({
            "episodes" : episodes,
            "train_wr" : round(train_wr, 4),
            "test_wr"  : round(test_wr,  4),
        })

    def log_final_eval(self, eval_dict: Dict):
        self.final_eval = eval_dict

    def log_tier_results(self, tier_dict: Dict):
        self.tier_results = tier_dict

    def log_hyperparams(self, **kwargs):
        self.hyperparams = kwargs

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "algo"               : self.algo,
            "hyperparams"        : self.hyperparams,
            "sample_efficiency"  : self.sample_efficiency,
            "final_eval"         : self.final_eval,
            "tier_results"       : self.tier_results,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[metrics] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "MetricsLogger":
        with open(path) as f:
            data = json.load(f)
        obj = cls(data["algo"])
        obj.hyperparams       = data.get("hyperparams", {})
        obj.sample_efficiency = data.get("sample_efficiency", [])
        obj.final_eval        = data.get("final_eval")
        obj.tier_results      = data.get("tier_results")
        return obj
