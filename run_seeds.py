import argparse
import json
import os
import sys
import numpy as np
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ExperimentConfig


def run_seeds(algo, n_seeds, base_cfg):
    loggers = []
    for seed in range(n_seeds):
        cfg = ExperimentConfig(**asdict(base_cfg))
        cfg.seed = seed
        cfg.split_seed = seed
        cfg.run_name = f"{algo}_seed{seed}"
        print("\n" + "=" * 60)
        print(f"  {algo.upper()} — train_seed={seed} split_seed={seed}")
        print("=" * 60)
        if algo == "grpo":
            from grpo import train
        else:
            from ppo import train
        _, logger = train(cfg)
        loggers.append(logger)
    return loggers


def aggregate_results(loggers):
    win_rates = [lg.final_eval["win_rate"] for lg in loggers if lg.final_eval]
    avg_guesses = [lg.final_eval["avg_guesses"] for lg in loggers if lg.final_eval and lg.final_eval.get("avg_guesses")]
    g1_csr = [lg.final_eval["constraint_reduction_by_pos"][0] for lg in loggers
               if lg.final_eval and lg.final_eval.get("constraint_reduction_by_pos")]
    ent_g0 = [lg.final_eval["entropy_by_pos"].get("0", 0) for lg in loggers
               if lg.final_eval and lg.final_eval.get("entropy_by_pos")]

    all_se = [lg.sample_efficiency for lg in loggers if lg.sample_efficiency]
    se_agg = {}
    if all_se:
        all_eps = sorted({d["episodes"] for se in all_se for d in se})
        for ep in all_eps:
            vals = [d["test_wr"] for se in all_se for d in se if d["episodes"] == ep]
            if vals:
                se_agg[ep] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4), "n": len(vals)}

    def _agg(arr):
        return {"mean": round(float(np.mean(arr)), 4), "std": round(float(np.std(arr)), 4)} if arr else {"mean": None, "std": None}

    return {
        "n_seeds": len(loggers),
        "win_rate": {**_agg(win_rates), "all": win_rates},
        "avg_guesses": _agg(avg_guesses),
        "g1_csr": {**_agg(g1_csr), "all": g1_csr},
        "entropy_g0": _agg(ent_g0),
        "sample_efficiency_by_episode": se_agg,
    }


def print_aggregate(agg, algo):
    print(f"\n{'='*60}\n  {algo.upper()} — {agg['n_seeds']} seeds aggregated\n{'='*60}")
    wr = agg["win_rate"]
    print(f"  Win rate:    {wr['mean']*100:.1f}% ± {wr['std']*100:.1f}%")
    for key, label in [("avg_guesses", "Avg guesses"), ("g1_csr", "G1 CSR"), ("entropy_g0", "Entropy@g0")]:
        d = agg[key]
        if d["mean"] is not None:
            print(f"  {label:12} {d['mean']:.3f} ± {d['std']:.3f}")
    se = agg["sample_efficiency_by_episode"]
    if se:
        print("\n  Sample efficiency (test win rate at checkpoints):")
        print(f"  {'Episodes':>10}  {'Mean':>8}  {'Std':>8}")
        for ep, v in sorted(se.items()):
            print(f"  {ep:>10,}  {v['mean']*100:>7.1f}%  ±{v['std']*100:.1f}%")


def _cross_compare(ppo_agg, grpo_agg):
    print(f"\n{'='*60}\n  PPO vs GRPO — Multi-seed summary\n{'='*60}")

    def fmt(d, pct=False):
        if not d or d["mean"] is None:
            return "   —   "
        s = 100 if pct else 1
        return f"{d['mean']*s:.1f}{'%' if pct else ''} ±{d['std']*s:.1f}"

    rows = [("Win rate", "win_rate", True), ("Avg guesses", "avg_guesses", False),
            ("G1 CSR", "g1_csr", False), ("Entropy@g0", "entropy_g0", False)]
    print(f"  {'Metric':20}  {'PPO':>16}  {'GRPO':>16}")
    print(f"  {'─'*20}  {'─'*16}  {'─'*16}")
    for label, key, pct in rows:
        print(f"  {label:20}  {fmt(ppo_agg.get(key), pct):>16}  {fmt(grpo_agg.get(key), pct):>16}")

    try:
        from metrics import hypothesis_test
        ppo_g1 = ppo_agg.get("g1_csr", {}).get("all", [])
        grpo_g1 = grpo_agg.get("g1_csr", {}).get("all", [])
        if ppo_g1 and grpo_g1:
            r = hypothesis_test(ppo_g1, grpo_g1)
            sig = "✅ SIGNIFICANT" if r["significant"] else "❌ NOT SIGNIFICANT"
            print(f"\n  Welch t-test (H1: GRPO G1-CSR > PPO):")
            print(f"    t={r['t_statistic']:.3f}  p={r['p_value']:.4f}  d={r['cohens_d']:.3f} ({r['effect_size']})")
            print(f"    {sig} at α=0.05")
    except Exception as e:
        print(f"\n  Stats test skipped: {e}")


def main():
    p = argparse.ArgumentParser("Multi-seed runner")
    p.add_argument("--algo",           default="both", choices=["ppo", "grpo", "both"])
    p.add_argument("--n_seeds",        type=int,   default=5)
    p.add_argument("--total_episodes", type=int,   default=300_000)
    p.add_argument("--total_steps",    type=int,   default=300_000)
    p.add_argument("--group_size",     type=int,   default=16)
    p.add_argument("--hidden",         type=int,   default=256)
    p.add_argument("--reward_type",    default="shaped")
    p.add_argument("--ckpt_dir",       default="checkpoints")
    p.add_argument("--device",         default="cpu")
    args = p.parse_args()

    algos = ["ppo", "grpo"] if args.algo == "both" else [args.algo]
    aggregated = {}
    for algo in algos:
        base_cfg = ExperimentConfig(
            algo=algo, total_episodes=args.total_episodes, total_steps=args.total_steps,
            group_size=args.group_size, hidden=args.hidden, reward_type=args.reward_type,
            ckpt_dir=args.ckpt_dir, device=args.device,
        )
        loggers = run_seeds(algo, args.n_seeds, base_cfg)
        agg = aggregate_results(loggers)
        aggregated[algo] = agg
        print_aggregate(agg, algo)
        out_path = os.path.join(args.ckpt_dir, f"{algo}_aggregate.json")
        os.makedirs(args.ckpt_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\n  Saved → {out_path}")

    if "ppo" in aggregated and "grpo" in aggregated:
        _cross_compare(aggregated["ppo"], aggregated["grpo"])


if __name__ == "__main__":
    main()
