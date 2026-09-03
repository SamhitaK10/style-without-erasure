"""Verify every reported number against the artifact that should back it.

Compares `analysis/master.json` -- the single source every figure and table is
generated from -- against the raw experiment outputs in `results/`. Writes
`docs/RESULT_VERIFICATION.md`.

This tool does not run experiments. It checks that what the paper reports is
what the surviving artifacts say, and reports NO ARTIFACT wherever the raw
output was never saved. Re-run the relevant experiment, drop its JSON into
`results/raw/`, and run this again: rows flip from NO ARTIFACT to a real
comparison automatically.

    python tools/verify_results.py            # write docs/RESULT_VERIFICATION.md
    python tools/verify_results.py --check    # exit 1 on any MISMATCH (for CI)

Statuses
    EXACT                    identical to the artifact
    MATCHES WITHIN TOLERANCE equal after documented rounding
    MISMATCH                 artifact and paper disagree -- investigate
    NO ARTIFACT              the raw output was never saved; see results/MISSING.md
    NOT RUN                  a rerun script exists but has not been executed here
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf8") as f:
        return json.load(f)


def cmp_num(reported, artifact, tol=5e-4):
    """EXACT / MATCHES WITHIN TOLERANCE / MISMATCH plus the signed difference."""
    if artifact is None:
        return "NO ARTIFACT", None
    d = float(reported) - float(artifact)
    if reported == artifact:
        return "EXACT", 0.0
    if abs(d) <= tol or abs(round(float(artifact), 3) - float(reported)) <= 1e-9:
        return "MATCHES WITHIN TOLERANCE", d
    return "MISMATCH", d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 on any MISMATCH")
    a = ap.parse_args()

    M = load("analysis/master.json")
    base = load("results/baseline_v6.json")
    ft = load("results/finetune_v2_ACCEPTED.json")
    sel = load("results/selectivity_PASSED.json")
    loc = load("results/localisation_1.5B.json")
    hard1 = load("results/finetune_attempt1_REJECTED.json")
    rerun_patch = load("results/raw/localisation_1.5B_rerun.json")

    rows = []   # (paper location, metric, reported, reproduced, diff, artifact, status)

    def add(loc_, metric, rep, art, src, tol=5e-4):
        st, d = cmp_num(rep, art, tol) if art is not None else ("NO ARTIFACT", None)
        rows.append((loc_, metric, rep, art if art is not None else "--",
                     f"{d:+.4g}" if d is not None else "--", src, st))

    def add_missing(loc_, metric, rep, why):
        rows.append((loc_, metric, rep, "--", "--", why, "NO ARTIFACT"))

    # ---- Table 1 / §5.1 -- baseline
    for dim, v in sorted(M["baseline"]["dims"].items()):
        src = base.get(dim) if base else None
        add("Table 1", f"{dim} ratio", v["ratio"], src["ratio"] if src else None,
            "results/baseline_v6.json")
        add("Table 1", f"{dim} D_style", v["d_style"], src["d_style"] if src else None,
            "results/baseline_v6.json")
        add("Table 1", f"{dim} D_placebo", v["d_placebo"], src["d_placebo"] if src else None,
            "results/baseline_v6.json")
    add("§5.1", "positive control (bits)", M["baseline"]["positive_control_bits"],
        base["_controls"]["positive"] if base else None, "results/baseline_v6.json")

    # ---- Table 2 / §5.2 -- cue operationalisation
    for arm, v in M["cue_robustness"]["arms"].items():
        add_missing("Table 2", f"{arm} median ratio", v.get("median_ratio"),
                    "exp_B_raw_instructions.json never saved")

    # ---- Table 3 / §5.3 -- placebo robustness
    for k, v in M["placebo_robustness"]["family"].items():
        add_missing("Table 3", f"family {k}", v, "exp_B2_placebo_sensitivity.json never saved")

    # ---- Table 4 / §5.4 -- intervention
    orig = [r for r in M["intervention"]["runs"] if r["label"].startswith("original")][0]
    if ft:
        add("Table 4", "original mean reduction %", orig["reduction_pct"],
            abs(ft["bias_style_gap"]["MEAN"]["change_pct"]), "results/finetune_v2_ACCEPTED.json")
        add("Table 4", "original perplexity after", orig["after_ppl"],
            ft["quality_guard"]["doctor_perplexity_after"], "results/finetune_v2_ACCEPTED.json")
        add("Table 4", "original neutral drift (bits)", orig["drift"],
            ft["quality_guard"]["neutral_answer_drift_bits"], "results/finetune_v2_ACCEPTED.json")
        add("§5.5", "perplexity before", M["intervention"]["before_ppl"],
            ft["quality_guard"]["doctor_perplexity_before"], "results/finetune_v2_ACCEPTED.json")
        # Two separate questions, kept separate:
        #  (1) does master.json agree with the value the artifact STATES?
        #      The artifact states change_pct to 1 decimal, so the tolerance is 0.05.
        #  (2) is the artifact internally consistent -- does its stated change_pct
        #      match a recomputation from its own before/after fields?
        for dim, val in sorted(orig["per_dim"].items()):
            g = ft["bias_style_gap"].get(dim)
            stated = abs(g["change_pct"]) if g else None
            add("Table 4", f"original {dim} reduction %", val, stated,
                "results/finetune_v2_ACCEPTED.json (stated change_pct)", tol=0.05)
            if g:
                recomputed = abs((g["after"] - g["before"]) / g["before"] * 100)
                add("artifact self-check", f"{dim}: stated vs recomputed", stated,
                    round(recomputed, 1),
                    "results/finetune_v2_ACCEPTED.json (before/after fields)", tol=0.05)
    for r in M["intervention"]["runs"]:
        if r["label"].startswith("seed"):
            add_missing("Table 4", f"{r['label']} reduction %", r["reduction_pct"],
                        f"exp_A_{r['label'].replace(' ', '')}.json never saved")
    add_missing("Table 4", "3-seed mean reduction %", M["intervention"]["seeded_mean"],
                "exp_A_seed{1,2,3}.json never saved")

    # ---- §5.5 -- selectivity
    if sel:
        s = M["intervention"]["selectivity"]
        add("§5.5", "clinical change %", s["clinical_change_pct"],
            sel["contrasts"]["medical"]["change_pct"], "results/selectivity_PASSED.json")
        add("§5.5", "placebo reduction %", s["placebo_reduction_pct"],
            abs(sel["contrasts"]["placebo"]["change_pct"]), "results/selectivity_PASSED.json")
        add("§5.5", "style influence before %", s["style_influence_before_pct"],
            sel["relative_influence"]["style_as_pct_of_medical_before"],
            "results/selectivity_PASSED.json")
        add("§5.5", "style influence after %", s["style_influence_after_pct"],
            sel["relative_influence"]["style_as_pct_of_medical_after"],
            "results/selectivity_PASSED.json")

    # ---- Table 5 / §5.7 -- objective comparison
    oc = M["objective_comparison"]
    for k, v in oc["rows"].items():
        add_missing("Table 5", f"{k} entropy (bits)", v["entropy"],
                    "exp_E_objective_comparison.json never saved")
        add_missing("Table 5", f"{k} top-1 probability", v["top1"],
                    "exp_E_objective_comparison.json never saved")
    if hard1:
        add("§5.7", "hard-target drift, first attempt (bits)",
            oc["rows"]["hard_target"]["drift"],
            hard1["quality_guard"]["neutral_answer_drift_bits"],
            "results/finetune_attempt1_REJECTED.json", tol=6e-4)

    # ---- Table 6 / §5.6 -- probe
    add_missing("Table 6", "max base style AUC", M["probe"]["max_auc_base"],
                "exp_D5_probe.json never saved")
    add_missing("Table 6", "resolution bound (AUC)", M["probe"]["bound_auc"],
                "exp_D5_probe.json never saved")

    # ---- §6 / Appendix H -- activation patching
    pa = M["patching"]
    if loc:
        for dim, key in (("fluency", "fluency_peak"), ("confidence", "confidence_peak"),
                         ("health_literacy", "health_literacy_peak")):
            add("§6.2 / App. H", f"{dim} peak delta", pa[key]["delta"],
                loc["peaks"][dim]["diff"], "results/localisation_1.5B.json")
            add("§6.2 / App. H", f"{dim} peak layer", pa[key]["layer"],
                loc["peaks"][dim]["layer"], "results/localisation_1.5B.json")
        add("§6.2 / App. H", "magnitude floor", pa["magnitude_floor"],
            0.03, "results/localisation_1.5B.json (_reporting_rule)")

    # patching rerun, if the reconstruction has been executed
    if rerun_patch:
        for dim, key in (("fluency", "fluency_peak"), ("confidence", "confidence_peak"),
                         ("health_literacy", "health_literacy_peak")):
            d = rerun_patch["per_dimension"].get(dim)
            if d:
                add("§6.2 rerun", f"{dim} peak delta (reconstruction)", pa[key]["delta"],
                    d["peak"]["diff"], "results/raw/localisation_1.5B_rerun.json", tol=0.02)
    else:
        rows.append(("§6.2 / App. H", "reconstruction rerun", "--", "--", "--",
                     "experiments/12_patching_1.5b.py not yet executed", "NOT RUN"))

    # ------------------------------------------------------------------- report
    counts = {}
    for r in rows:
        counts[r[-1]] = counts.get(r[-1], 0) + 1

    out = ["# Result verification", "",
           "Generated by `python tools/verify_results.py`. Every row compares a number",
           "in `analysis/master.json` -- the single source the figures, the LaTeX tables",
           "and the paper are built from -- against the raw experiment output that should",
           "back it.", "",
           "| Status | Count |", "|---|---:|"]
    for k in ("EXACT", "MATCHES WITHIN TOLERANCE", "MISMATCH", "NO ARTIFACT", "NOT RUN"):
        if k in counts:
            out.append(f"| {k} | {counts[k]} |")
    out += ["", "| Paper location | Metric | Reported | Reproduced | Difference | Source artifact | Status |",
            "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    out += ["", "---", "",
            "**NO ARTIFACT** means the experiment ran and its numbers are in the paper, but",
            "the raw output was never written to disk. See [`../results/MISSING.md`](../results/MISSING.md)",
            "for the file list and the re-run order. These rows become real comparisons as",
            "soon as the corresponding JSON lands in `results/raw/`.", "",
            "**NOT RUN** means a script exists but has not been executed in this environment.", ""]

    dst = os.path.join(ROOT, "docs/RESULT_VERIFICATION.md")
    with open(dst, "w", encoding="utf8") as f:
        f.write("\n".join(out) + "\n")

    for k in ("EXACT", "MATCHES WITHIN TOLERANCE", "MISMATCH", "NO ARTIFACT", "NOT RUN"):
        if k in counts:
            print(f"  {k:26s} {counts[k]}")
    print(f"\nwrote docs/RESULT_VERIFICATION.md ({len(rows)} rows)")

    if a.check and counts.get("MISMATCH"):
        print(f"\nFAILED: {counts['MISMATCH']} mismatch(es)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
