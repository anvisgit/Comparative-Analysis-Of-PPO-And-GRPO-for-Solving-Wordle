import json
import os
import sys

CKPT_DIR = "checkpoints"
PPO_LOG  = os.path.join(CKPT_DIR, "ppo_metrics.json")
GRPO_LOG = os.path.join(CKPT_DIR, "grpo_metrics.json")


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(v, pct=False):
    if v is None:
        return "  —   "
    if pct:
        return f"{v*100:5.1f}%"
    return f"{v:.4f}"


def print_header(title):
    print(f"\n{'─'*66}")
    print(f"  {title}")
    print(f"{'─'*66}")


def main():
    missing = [p for p in [PPO_LOG, GRPO_LOG] if not os.path.exists(p)]
    if missing:
        print(f"Missing log files: {missing}")
        print("Run train_ppo.py and train_grpo.py first.")
        sys.exit(1)

    ppo  = load(PPO_LOG)
    grpo = load(GRPO_LOG)
    print_header("1. Sample Efficiency  (test win rate at each checkpoint)")
    print(f"  {'Episodes':>10}  {'PPO test':>10}  {'GRPO test':>10}  {'Δ (GRPO-PPO)':>14}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")

    ppo_se  = {d["episodes"]: d["test_wr"]  for d in ppo.get("sample_efficiency",  [])}
    grpo_se = {d["episodes"]: d["test_wr"]  for d in grpo.get("sample_efficiency", [])}
    all_eps = sorted(set(list(ppo_se.keys()) + list(grpo_se.keys())))

    for ep in all_eps:
        p = ppo_se.get(ep)
        g = grpo_se.get(ep)
        delta = f"{(g-p)*100:+.1f}pp" if (p is not None and g is not None) else "  —  "
        pstr  = fmt(p, pct=True) if p is not None else "   —   "
        gstr  = fmt(g, pct=True) if g is not None else "   —   "
        print(f"  {ep:>10,}  {pstr:>10}  {gstr:>10}  {delta:>14}")
    print_header("2. Constraint Reduction per Guess  (mean fraction eliminated)")
    print(f"  {'Guess':>6}  {'PPO':>8}  {'GRPO':>8}  {'Δ':>10}  {'GRPO wins?':>10}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")

    ppo_cr  = ppo.get("final_eval",  {}).get("constraint_reduction_by_pos", [])
    grpo_cr = grpo.get("final_eval", {}).get("constraint_reduction_by_pos", [])

    for i in range(max(len(ppo_cr), len(grpo_cr))):
        p = ppo_cr[i]  if i < len(ppo_cr)  else None
        g = grpo_cr[i] if i < len(grpo_cr) else None
        delta = f"{(g-p):+.3f}" if (p and g) else "  —  "
        winner = "✓" if (p and g and g > p) else ("✗" if (p and g) else "")
        pstr = f"{p:.3f}" if p is not None else "  —  "
        gstr = f"{g:.3f}" if g is not None else "  —  "
        print(f"  {i+1:>6}  {pstr:>8}  {gstr:>8}  {delta:>10}  {winner:>10}")
    print_header("3. Policy Entropy per Guess Position  (higher = more exploratory)")
    print(f"  {'Guess':>6}  {'PPO':>8}  {'GRPO':>8}  {'Δ':>10}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}")

    ppo_ent  = ppo.get("final_eval",  {}).get("entropy_by_pos", {})
    grpo_ent = grpo.get("final_eval", {}).get("entropy_by_pos", {})
    all_pos  = sorted(
        set(list(ppo_ent.keys()) + list(grpo_ent.keys())),
        key=lambda x: int(x)
    )
    for pos in all_pos:
        p = ppo_ent.get(pos)
        g = grpo_ent.get(pos)
        delta = f"{(g-p):+.2f}" if (p and g) else "  —  "
        pstr  = f"{p:.2f}" if p else "  —  "
        gstr  = f"{g:.2f}" if g else "  —  "
        print(f"  {int(pos)+1:>6}  {pstr:>8}  {gstr:>8}  {delta:>10}")
    print_header("4. First-Guess Diversity")
    for name, data in [("PPO", ppo), ("GRPO", grpo)]:
        fg = data.get("final_eval", {}).get("first_guess_stats", {})
        if fg:
            print(f"  {name}: entropy={fg.get('entropy'):.3f} | "
                  f"unique={fg.get('n_unique')} | "
                  f"collapse={fg.get('collapse_flag')} | "
                  f"top1={fg.get('top1_fraction')*100:.1f}%")
            print(f"    top-5: {fg.get('top_5')}")
    print_header("5. Win Rate by Word Frequency Tier  (test set)")
    print(f"  {'Tier':>8}  {'PPO':>8}  {'GRPO':>8}  {'Δ':>10}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")

    ppo_tiers  = ppo.get("tier_results",  {}) or {}
    grpo_tiers = grpo.get("tier_results", {}) or {}
    all_tiers  = ["common", "medium", "rare"]

    for tier in all_tiers:
        p = ppo_tiers.get(tier)
        g = grpo_tiers.get(tier)
        delta = f"{(g-p)*100:+.1f}pp" if (p is not None and g is not None) else "  —  "
        pstr  = fmt(p, pct=True) if p is not None else "   —   "
        gstr  = fmt(g, pct=True) if g is not None else "   —   "
        print(f"  {tier:>8}  {pstr:>8}  {gstr:>8}  {delta:>10}")
    print_header("6. Final Eval Summary")
    print(f"  {'Metric':30}  {'PPO':>10}  {'GRPO':>10}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*10}")

    pfe  = ppo.get("final_eval",  {})
    gfe  = grpo.get("final_eval", {})

    rows = [
        ("Win rate (test)",   pfe.get("win_rate"),  gfe.get("win_rate"),  True),
        ("Avg guesses",       pfe.get("avg_guesses"), gfe.get("avg_guesses"), False),
    ]
    for label, p, g, is_pct in rows:
        pstr = fmt(p, pct=is_pct) if p is not None else "   —   "
        gstr = fmt(g, pct=is_pct) if g is not None else "   —   "
        print(f"  {label:30}  {pstr:>10}  {gstr:>10}")
    print_header("7. Thesis Check — GRPO information-seeking at guess 1")
    ppo_g1  = ppo_cr[0]  if len(ppo_cr)  > 0 else None
    grpo_g1 = grpo_cr[0] if len(grpo_cr) > 0 else None

    if ppo_g1 is not None and grpo_g1 is not None:
        delta = grpo_g1 - ppo_g1
        if delta > 0.05:
            verdict = (
                " SUPPORTED"
                "     consistent with group baseline encouraging information-"
                "seeking behaviour."
            )
        elif delta > 0:
            verdict = (
                " WEAK — GRPO has higher guess-1 constraint reduction "
                f"({delta:+.3f}) but\n"
                "     effect size is small. Needs more runs or seeds."
            )
        else:
            verdict = (
                " NOT SUPPORTED — PPO guess-1 constraint reduction ≥ GRPO.\n"
                "     Consider revisiting group size, entropy coef, or total episodes."
            )
        print(f"  PPO  guess-1 constraint reduction: {ppo_g1:.3f}")
        print(f"  GRPO guess-1 constraint reduction: {grpo_g1:.3f}")
        print(f"  Δ = {delta:+.3f}")
        print(f"\n  Verdict: {verdict}")
    else:
        print("  Insufficient data — run both trainers first.")

    print()


if __name__ == "__main__":
    main()
