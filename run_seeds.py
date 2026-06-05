"""
run_seeds.py — multi-seed runner for both PPO and GRPO.

Runs N independent seeds and aggregates results with mean ± std.
This is the minimum bar for any publishable comparison.

Usage:
    python run_seeds.py --algo grpo --n_seeds 5 --total_episodes 300000
    python run_seeds.py --algo ppo  --n_seeds 5 --total_steps 300000
    python run_seeds.py --algo both --n_seeds 5   # runs both sequentially
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ExperimentConfig


def run_seeds(algo: str, n_seeds: int, base_cfg: ExperimentConfig):
    """Run n_seeds independent trials and return list of MetricsLogger objects."""
    loggers = []

    for seed in range(n_seeds):
        cfg          = ExperimentConfig(**base_cfg.__dict__)
        cfg.seed     = seed
        cfg.run_name = f"{algo}_seed{seed}"

        print("\n" + "=" * 60)
        print(f"  {algo.upper()} — seed {seed}/{n_seeds-1}")
        print("=" * 60)

        if algo == "grpo":
            from grpo import train
        else:
            from ppo import train

        _, logger = train(cfg)
        loggers.append(logger)

    return loggers


def aggregate_results(loggers) -> dict:
    """
    Aggregate metrics across seeds.
    Returns dict with mean ± std for each key metric.
    """
    # Win rates
    win_rates = [
        lg.final_eval["win_rate"] for lg in loggers
        if lg.final_eval is not None
    ]
    avg_guesses = [
        lg.final_eval["avg_guesses"] for lg in loggers
        if lg.final_eval is not None and lg.final_eval.get("avg_guesses")
    ]

    # Constraint reduction at guess 1 (mean across episodes per seed)
    g1_csr = [
        lg.final_eval["constraint_reduction_by_pos"][0]
        for lg in loggers
        if lg.final_eval and lg.final_eval.get("constraint_reduction_by_pos")
    ]

    # Entropy at guess 0 per seed
    ent_g0 = [
        lg.final_eval["entropy_by_pos"].get("0", 0)
        for lg in loggers
        if lg.final_eval and lg.final_eval.get("entropy_by_pos")
    ]

    # Sample efficiency: align by checkpoint, average across seeds
    all_se = [lg.sample_efficiency for lg in loggers if lg.sample_efficiency]
    se_agg = {}
    if all_se:
        all_eps = sorted({d["episodes"] for se in all_se for d in se})
        for ep in all_eps:
            vals = [
                d["test_wr"] for se in all_se
                for d in se if d["episodes"] == ep
            ]
            if vals:
                se_agg[ep] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std" : round(float(np.std(vals)),  4),
                    "n"   : len(vals),
                }

    return {
        "n_seeds"       : len(loggers),
        "win_rate"      : {
            "mean": round(float(np.mean(win_rates)), 4),
            "std" : round(float(np.std(win_rates)),  4),
            "all" : win_rates,
        },
        "avg_guesses"   : {
            "mean": round(float(np.mean(avg_guesses)), 4) if avg_guesses else None,
            "std" : round(float(np.std(avg_guesses)),  4) if avg_guesses else None,
        },
        "g1_csr"        : {
            "mean": round(float(np.mean(g1_csr)), 4) if g1_csr else None,
            "std" : round(float(np.std(g1_csr)),  4) if g1_csr else None,
            "all" : g1_csr,
        },
        "entropy_g0"    : {
            "mean": round(float(np.mean(ent_g0)), 4) if ent_g0 else None,
            "std" : round(float(np.std(ent_g0)),  4) if ent_g0 else None,
        },
        "sample_efficiency_by_episode": se_agg,
    }


def print_aggregate(agg: dict, algo: str):
    print(f"\n{'='*60}")
    print(f"  {algo.upper()} — {agg['n_seeds']} seeds aggregated")
    print(f"{'='*60}")
    wr = agg["win_rate"]
    print(f"  Win rate:       {wr['mean']*100:.1f}% ± {wr['std']*100:.1f}%")
    ag = agg["avg_guesses"]
    if ag["mean"]:
        print(f"  Avg guesses:    {ag['mean']:.2f} ± {ag['std']:.2f}")
    g1 = agg["g1_csr"]
    if g1["mean"]:
        print(f"  G1 CSR:         {g1['mean']:.3f} ± {g1['std']:.3f}")
    en = agg["entropy_g0"]
    if en["mean"]:
        print(f"  Entropy@g0:     {en['mean']:.2f} ± {en['std']:.2f}")

    se = agg["sample_efficiency_by_episode"]
    if se:
        print("\n  Sample efficiency (test win rate at checkpoints):")
        print(f"  {'Episodes':>10}  {'Mean':>8}  {'Std':>8}")
        for ep, v in sorted(se.items()):
            print(f"  {ep:>10,}  {v['mean']*100:>7.1f}%  ±{v['std']*100:.1f}%")


def main():
    p = argparse.ArgumentParser("Multi-seed runner")
    p.add_argument("--algo",            default="both",
                   choices=["ppo", "grpo", "both"])
    p.add_argument("--n_seeds",         type=int,   default=5)
    p.add_argument("--total_episodes",  type=int,   default=300_000)
    p.add_argument("--total_steps",     type=int,   default=300_000)
    p.add_argument("--group_size",      type=int,   default=16)
    p.add_argument("--hidden",          type=int,   default=256)
    p.add_argument("--reward_type",     default="shaped")
    p.add_argument("--include_rem",
                   type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--ckpt_dir",        default="checkpoints")
    p.add_argument("--device",          default="cpu")
    args = p.parse_args()

    algos = ["ppo", "grpo"] if args.algo == "both" else [args.algo]
    aggregated = {}

    for algo in algos:
        base_cfg = ExperimentConfig(
            algo           = algo,
            total_episodes = args.total_episodes,
            total_steps    = args.total_steps,
            group_size     = args.group_size,
            hidden         = args.hidden,
            reward_type    = args.reward_type,
            include_rem    = args.include_rem,
            ckpt_dir       = args.ckpt_dir,
            device         = args.device,
        )
        loggers           = run_seeds(algo, args.n_seeds, base_cfg)
        agg               = aggregate_results(loggers)
        aggregated[algo]  = agg
        print_aggregate(agg, algo)

        # Save aggregate JSON
        out_path = os.path.join(args.ckpt_dir, f"{algo}_aggregate.json")
        os.makedirs(args.ckpt_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\n  Saved → {out_path}")

    # Cross-algo comparison if both ran
    if "ppo" in aggregated and "grpo" in aggregated:
        _cross_compare(aggregated["ppo"], aggregated["grpo"])


def _cross_compare(ppo_agg: dict, grpo_agg: dict):
    print(f"\n{'='*60}")
    print("  PPO vs GRPO — Multi-seed summary")
    print(f"{'='*60}")

    def fmt(d, pct=False):
        if not d or d["mean"] is None:
            return "   —   "
        scale = 100 if pct else 1
        fmt_s = f"{d['mean']*scale:.1f}" + ("%" if pct else "")
        return f"{fmt_s} ±{d['std']*scale:.1f}"

    rows = [
        ("Win rate",     "win_rate",   True),
        ("Avg guesses",  "avg_guesses", False),
        ("G1 CSR",       "g1_csr",     False),
        ("Entropy@g0",   "entropy_g0", False),
    ]
    print(f"  {'Metric':20}  {'PPO':>16}  {'GRPO':>16}")
    print(f"  {'─'*20}  {'─'*16}  {'─'*16}")
    for label, key, pct in rows:
        p = fmt(ppo_agg.get(key), pct)
        g = fmt(grpo_agg.get(key), pct)
        print(f"  {label:20}  {p:>16}  {g:>16}")

    # Statistical test on G1 CSR if raw values available
    try:
        from metrics import hypothesis_test
        ppo_g1  = ppo_agg.get("g1_csr", {}).get("all", [])
        grpo_g1 = grpo_agg.get("g1_csr", {}).get("all", [])
        if ppo_g1 and grpo_g1:
            result = hypothesis_test(ppo_g1, grpo_g1)
            print(f"\n  Thesis test (Welch t-test, H1: GRPO G1-CSR > PPO):")
            print(f"    t={result['t_statistic']:.3f}  "
                  f"p={result['p_value']:.4f}  "
                  f"d={result['cohens_d']:.3f} ({result['effect_size']})")
            sig = "✅ SIGNIFICANT" if result["significant"] else "❌ NOT SIGNIFICANT"
            print(f"    {sig} at α=0.05")
    except Exception as e:
        print(f"\n  Stats test skipped: {e}")


if __name__ == "__main__":
    main()