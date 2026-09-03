# ========================= PASTE INTO A NEW CELL ==============================
# Diagnostic: did anything actually finish, and what did it say?
# Costs nothing, loads no model, safe to run any time.
# ==============================================================================
import os, glob, json

NAMES = ["positive_control.json", "exp_B_raw_instructions.json", "exp_D_probe.json",
         "exp_E_objective_comparison.json", "exp_A_seed1.json", "exp_A_seed2.json"]
ROOTS = [".", "/content", "/content/DiaLense_PartII",
         "/content/drive/MyDrive/DiaLense_PartII",
         "/kaggle/working", "/kaggle/working/DiaLense_PartII"]

print("="*72); print("WHAT EXISTS"); print("="*72)
found = {}
for n in NAMES:
    hits = [p for r in ROOTS for p in glob.glob(os.path.join(r, n)) if os.path.isfile(p)]
    if hits:
        p = hits[0]; found[n] = p
        print(f"  FOUND    {n:34s} {os.path.getsize(p):>8,} bytes   {p}")
    else:
        print(f"  missing  {n}")

print("\nadapters:")
for pat in ["dialense_lora", "dialense_lora_hard", "dialense_lora_hard_ckpt",
            "dialense_lora_seed1", "dialense_lora_seed2"]:
    hits = [p for r in ROOTS for p in glob.glob(os.path.join(r, pat))
            if os.path.isfile(os.path.join(p, "adapter_config.json"))]
    print(f"  {'FOUND   ' if hits else 'missing '} {pat:26s} {hits[0] if hits else ''}")

# ---------------------------------------------------------------- EXP_B ------
if "exp_B_raw_instructions.json" in found:
    d = json.load(open(found["exp_B_raw_instructions.json"]))
    print("\n" + "="*72); print("EXP_B — DID THE EFFECT SURVIVE THE CORPUS'S OWN INSTRUCTIONS?")
    print("="*72)
    print(f"positive control (this run's own scenarios): {d['controls']['positive']:.5f} bits")
    print(f"negative control: {d['controls']['negative']:.3e}   scenarios: {d['controls']['n_scenarios']}")
    for arm in ("DESC", "CONV", "RAW"):
        a = d["arms"][arm]; s = a["_summary"]
        print(f"\n{arm}   {s['n_positive']}/5 dimensions positive   "
              f"sign-test p = {s['sign_test_p']:.4f}   median ratio {s['median_ratio']:.1f}x")
        for dim in [k for k in a if not k.startswith("_")]:
            r = a[dim]
            print(f"   {dim:26s} style {r['d_style']:.5f}  placebo {r['d_placebo']:.5f}  "
                  f"ratio {r['ratio']:5.1f}x  ({r['pct_of_positive']:4.1f}% of ctrl)"
                  f"{'' if r['placebo_matched'] else '   [NO PLACEBO MATCH]'}")
    rc = d.get("desc_vs_raw_rank_correlation", {})
    if rc: print(f"\nDESC vs RAW dimension ordering: rho = {rc['rho']:+.3f} (n={rc['n']}, p={rc['p']:.3f})")
    print("\nCONTENT-CONFOUND AUDIT (goes in the paper either way):")
    for dim, v in d["content_confound_audit"].items():
        if v["only_high"] or v["only_low"]:
            print(f"   {dim:26s} high-only {v['only_high']}  low-only {v['only_low']}")

# ---------------------------------------------------------------- EXP_E ------
if "exp_E_objective_comparison.json" in found:
    d = json.load(open(found["exp_E_objective_comparison.json"]))
    print("\n" + "="*72); print("EXP_E — OBJECTIVE COMPARISON"); print("="*72)
    print(f"teacher entropy {d['teacher_entropy_bits']:.3f} bits | "
          f"this run's positive control {d['positive_control']:.5f} bits")
    print(f"{'':16s} {'entropy':>9} {'top-1':>8} {'p>.99':>8} {'style gap':>11} {'drift':>9}")
    for k in ("base", "soft_target", "hard_target"):
        m = d[k]
        print(f"{k:16s} {m['entropy_bits']:>9.3f} {m['top1_prob']:>8.3f} "
              f"{m['frac_positions_p_gt_0p99']:>8.3f} {m['style_gap_mean']:>11.5f} "
              f"{m.get('neutral_drift', float('nan')):>9.5f}")
    print(f"\nreduction  soft {d['reduction_soft_pct']:+.1f}%   hard {d['reduction_hard_pct']:+.1f}%")

# ---------------------------------------------------------------- EXP_D ------
if "exp_D_probe.json" in found:
    d = json.load(open(found["exp_D_probe.json"]))
    print("\n" + "="*72); print("EXP_D — ERASED OR SUPPRESSED?"); print("="*72)
    print(f"mean AUC drop, final position {d['mean_auc_drop_final']:+.3f}")
    print(f"mean AUC drop, cue positions  {d['mean_auc_drop_cue']:+.3f}")
    print(f"content-control drop          {d['content_control_drop']:+.3f}  (should be ~0)")
    print(f"\n{d['verdict']}")

# ---------------------------------------------------------------- EXP_A ------
seeds = [found[n] for n in ("exp_A_seed1.json", "exp_A_seed2.json") if n in found]
if seeds:
    print("\n" + "="*72); print("EXP_A — SEEDS"); print("="*72)
    for p in seeds:
        r = json.load(open(p))
        print(f"  seed {r['seed']}: {r['reduction_pct']:+.1f}%   drift {r['neutral_drift']:.5f}")

if not found:
    print("\n" + "="*72)
    print("NOTHING FOUND. So the cell did not finish, whatever the screen showed.")
    print("="*72)
    print("""Check, in this order:
  1. Is the cell still running? Look for the spinner and an empty [ ] instead of
     a number like [3] to its left.
  2. Did the runtime restart? Files in /content are wiped. Runtime > Manage
     sessions will show whether the session is new.
  3. Did you paste the whole script? The last line should be
     print("saved exp_B_raw_instructions.json")
  4. Re-run it and DO NOT clear the output. The first thing it prints is
     "installing deps..." within a few seconds -- if you do not see that,
     the cell is not executing.""")
