"""Analysis: Is n=3 seeds enough for the claims being made?

The ladder uses 3 seeds (42, 7, 1) with same folds. The aggregate table shows mean ± std.
Let's assess whether this provides sufficient statistical power for the claims.
"""
import json
from pathlib import Path
import numpy as np

def load_all_seeds(step):
    """Load all 3 seeds for a step."""
    return [json.loads((Path(f"../{step}/seed{s}.json").read_text()))
            for s in [42, 7, 1]]

def analyze_statistical_power():
    print("STATISTICAL POWER ANALYSIS")
    print("=" * 70)
    print()

    # Load all results
    s0 = load_all_seeds("S0")
    s1 = load_all_seeds("S1")
    s4b = load_all_seeds("S4b")
    s4c = load_all_seeds("S4c")

    # Key metrics
    metrics = [
        ("flip_same_height", lambda x: x["distractor"]["same_height"]["flip_rate"]),
        ("wrongacc_same_height", lambda x: x["distractor"]["same_height"]["wrong_accept_rate"]),
        ("noise_ret", lambda x: x["noise_sigma1_gap_retention"]),
        ("openset_fa_m", lambda x: [r for r in x["open_set"] if r["held_class"] == "m"][0]["false_accept_rate_T_fit"]),
        ("sep_r/u", lambda x: x["hard_pairs"]["r/u"]),
        ("sep_g/n", lambda x: x["hard_pairs"]["g/n"]),
    ]

    print("Key comparisons with n=3 seeds:")
    print()

    for name, getter in metrics:
        s0_vals = [getter(s) for s in s0]
        s1_vals = [getter(s) for s in s1]
        s4b_vals = [getter(s) for s in s4b]
        s4c_vals = [getter(s) for s in s4c] if name != "openset_fa_m" else None

        print(f"{name}:")
        print(f"  S0:  {np.mean(s0_vals):.3f} ± {np.std(s0_vals):.3f}")
        print(f"  S1:  {np.mean(s1_vals):.3f} ± {np.std(s1_vals):.3f}")
        print(f"  S4b: {np.mean(s4b_vals):.3f} ± {np.std(s4b_vals):.3f}")
        if s4c_vals:
            print(f"  S4c: {np.mean(s4c_vals):.3f} ± {np.std(s4c_vals):.3f}")

        # Check if S1 -> S4b difference exceeds combined std
        diff = abs(np.mean(s1_vals) - np.mean(s4b_vals))
        combined_std = np.sqrt(np.std(s1_vals)**2 + np.std(s4b_vals)**2)

        if combined_std > 0:
            effect_size = diff / combined_std
            print(f"  S1->S4b: Δ={diff:.3f}, effect size={effect_size:.2f}")
            if effect_size > 1.5:
                print(f"  ✓ STRONG effect (>1.5 combined std)")
            elif effect_size > 1.0:
                print(f"  ~ MODERATE effect (>1.0 combined std)")
            else:
                print(f"  ✗ WEAK effect (<1.0 combined std) - UNDERPOWERED")
        print()

    print()
    print("ASSESSMENT:")
    print("=" * 70)
    print()
    print("Is n=3 seeds enough?")
    print()
    print("DEPENDS on effect size:")
    print("  - For LARGE effects (>2 std): n=3 may suffice")
    print("  - For MODERATE effects (1-2 std): n=3 is MARGINAL")
    print("  - For SMALL effects (<1 std): n=3 is UNDERPOWERED")
    print()
    print("From the results above:")
    print()
    print("WELL-POWERED claims (large effect):")
    print("  - flip_same_height: S1=0.115 -> S4b=0.026 (strong drop)")
    print("  - wrongacc_same_height: S1=0.055 -> S4b=0.010 (strong drop)")
    print()
    print("UNDERPOWERED claims (small/moderate effect):")
    print("  - Hard pair separations (r/u, g/n): high variance, overlapping std")
    print("  - noise_ret drop: S1=0.886 -> S4b=0.726 is significant, BUT")
    print("    we showed this is likely a metric artifact, not real robustness")
    print()
    print("CONCLUSION:")
    print("  n=3 seeds is SUFFICIENT for the main claim (distractor flip reduction)")
    print("  but INSUFFICIENT for confident claims about hard pair improvements.")
    print("  The user's rule requires 'significant' improvement - for distractor")
    print("  flip this is met, but hard pairs show high variance across seeds.")

if __name__ == "__main__":
    analyze_statistical_power()
