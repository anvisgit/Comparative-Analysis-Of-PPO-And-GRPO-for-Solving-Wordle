import json
import os
import sys

CKPT_DIR = "checkpoints"
PPO_LOG  = os.path.join(CKPT_DIR, "ppo_default_metrics.json")
GRPO_LOG = os.path.join(CKPT_DIR, "grpo_default_metrics.json")


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(v, pct=False):
    if v is None:
        return "  —   "
    return f"{v*100:5.1f}%" if pct else f"{v:.4f}"


def print_header(title):
    print(f"\n{'─'*66}\n  {title}\n{'─'*66}")


def main():
    missing = [p for p in [PPO_LOG, GRPO_LOG] if not os.path.exists(p)]
    if missing:
        print(f"Missing log files: {missing}")
        print("Run ppo.py and grpo.py first.")
        sys.exit(1)

    ppo  = load(PPO_LOG)
    grpo = load(GRPO_LOG)

    print_header("1. Sample Efficiency  (test win rate at each checkpoint)")
    print(f"  {'Episodes':>10}  {'PPO test':>10}  {'GRPO test':>10}  {'Δ (GRPO-PPO)':>14}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")
    ppo_se  = {d["episodes"]: d["test_wr"] for d in ppo.get("sample_efficiency",  [])}
    grpo_se = {d["episodes"]: d["test_wr"] for d in grpo.get("sample_efficiency", [])}
    for ep in sorted(set(list(ppo_se) + list(grpo_se))):
        p, g = ppo_se.get(ep), grpo_se.get(ep)
        delta = f"{(g-p)*100:+.1f}pp" if (p is not None and g is not None) else "  —  "
        print(f"  {ep:>10,}  {fmt(p, pct=True):>10}  {fmt(g, pct=True):>10}  {delta:>14}")

    print_header("2. Constraint Reduction per Guess  (mean fraction eliminated)")
    print(f"  {'Guess':>6}  {'PPO':>8}  {'GRPO':>8}  {'Δ':>10}  {'GRPO wins?':>10}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")
    ppo_cr  = ppo.get("final_eval",  {}).get("constraint_reduction_by_pos", [])
    grpo_cr = grpo.get("final_eval", {}).get("constraint_reduction_by_pos", [])
    for i in range(max(len(ppo_cr), len(grpo_cr))):
        p = ppo_cr[i]  if i < len(ppo_cr)  else None
        g = grpo_cr[i] if i < len(grpo_cr) else None
        delta  = f"{(g-p):+.3f}" if (p and g) else "  —  "
        winner = "✓" if (p and g and g > p) else ("✗" if (p and g) else "")
        print(f"  {i+1:>6}  {f'{p:.3f}' if p else '  —  ':>8}  {f'{g:.3f}' if g else '  —  ':>8}  {delta:>10}  {winner:>10}")

    print_header("3. Policy Entropy per Guess Position")
    print(f"  {'Guess':>6}  {'PPO':>8}  {'GRPO':>8}  {'Δ':>10}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}")
    ppo_ent  = ppo.get("final_eval",  {}).get("entropy_by_pos", {})
    grpo_ent = grpo.get("final_eval", {}).get("entropy_by_pos", {})
    for pos in sorted(set(list(ppo_ent) + list(grpo_ent)), key=int):
        p, g = ppo_ent.get(pos), grpo_ent.get(pos)
        delta = f"{(g-p):+.2f}" if (p and g) else "  —  "
        print(f"  {int(pos)+1:>6}  {f'{p:.2f}' if p else '  —  ':>8}  {f'{g:.2f}' if g else '  —  ':>8}  {delta:>10}")

    print_header("4. First-Guess Diversity")
    for name, data in [("PPO", ppo), ("GRPO", grpo)]:
        fg = data.get("final_eval", {}).get("first_guess_stats", {})
        if fg:
            print(f"  {name}: entropy={fg['entropy']:.3f} | unique={fg['n_unique']} | collapse={fg['collapse_flag']} | top1={fg['top1_fraction']*100:.1f}%")
            print(f"    top-5: {fg['top_5']}")

    print_header("5. Win Rate by Word Frequency Tier  (test set)")
    print(f"  {'Tier':>8}  {'PPO':>8}  {'GRPO':>8}  {'Δ':>10}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")
    ppo_tiers  = ppo.get("tier_results",  {}) or {}
    grpo_tiers = grpo.get("tier_results", {}) or {}
    for tier in ["common", "medium", "rare"]:
        p, g = ppo_tiers.get(tier), grpo_tiers.get(tier)
        delta = f"{(g-p)*100:+.1f}pp" if (p is not None and g is not None) else "  —  "
        print(f"  {tier:>8}  {fmt(p, pct=True):>8}  {fmt(g, pct=True):>8}  {delta:>10}")

    print_header("6. Final Eval Summary")
    print(f"  {'Metric':30}  {'PPO':>10}  {'GRPO':>10}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*10}")
    pfe, gfe = ppo.get("final_eval", {}), grpo.get("final_eval", {})
    for label, p, g, is_pct in [
        ("Win rate (test)", pfe.get("win_rate"),    gfe.get("win_rate"),    True),
        ("Avg guesses",     pfe.get("avg_guesses"), gfe.get("avg_guesses"), False),
    ]:
        print(f"  {label:30}  {fmt(p, pct=is_pct):>10}  {fmt(g, pct=is_pct):>10}")

    print_header("7. Thesis Check — GRPO information-seeking at guess 1")
    ppo_g1  = ppo_cr[0]  if ppo_cr  else None
    grpo_g1 = grpo_cr[0] if grpo_cr else None
    if ppo_g1 is not None and grpo_g1 is not None:
        delta = grpo_g1 - ppo_g1
        if delta > 0.05:
            verdict = "SUPPORTED — group baseline encouraging information-seeking behaviour."
        elif delta > 0:
            verdict = f"WEAK — GRPO has higher guess-1 CSR ({delta:+.3f}) but effect size is small."
        else:
            verdict = "NOT SUPPORTED — PPO guess-1 CSR ≥ GRPO."
        print(f"  PPO  guess-1 CSR: {ppo_g1:.3f}")
        print(f"  GRPO guess-1 CSR: {grpo_g1:.3f}")
        print(f"  Δ = {delta:+.3f}\n  Verdict: {verdict}")
    else:
        print("  Insufficient data — run both trainers first.")
    print()


if __name__ == "__main__":
    main()
