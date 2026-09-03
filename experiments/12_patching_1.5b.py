"""
Activation patching at the final prompt position, 1.5B base model.
Produces the numbers in paper §6.2 and Appendix H.

================================================================================
STATUS: RECONSTRUCTION. READ THIS BEFORE USING THE OUTPUT.
================================================================================
The script that originally produced `results/localisation_1.5B.json` was not
saved. It is not in this repository, not in any surviving session transcript,
and not on any machine reachable from here. This file is a reconstruction built
from four sources of evidence, and it has NOT yet been executed against the 1.5B
model, so it has not been shown to reproduce the committed result.

Run it (see RUNNING, below) and then `python tools/verify_patching.py` to find
out whether it does. Until that has been done, treat §6 as *documented* rather
than *reproduced*, exactly as `docs/RESULT_VERIFICATION.md` records.

WHAT IS RECOVERED, AND FROM WHERE
---------------------------------
Every value in the table below is fixed by an artifact in this repository. The
sources are:
  [A] results/localisation_1.5B.json  -- its _measure/_power/_reporting_rule
                                         fields describe the original design
  [P] paper/paper.md §6.2 and Appendix H
  [S] experiments/13_patching_0.5b.py -- the surviving 0.5B sibling, from which
                                         the hook mechanics and the restoration
                                         formula are taken verbatim
  [C] the split convention used identically by every other script here

  parameter                     value                                   source
  ---------------------------------------------------------------------------
  model                         Qwen/Qwen2.5-1.5B-Instruct              [A][P]
  layers scanned                all 28                                  [A][P]
  patch site                    final prompt position, one layer        [A][P]
  donor                         high-style run's hidden state           [P][S]
  recipient                     low-style run                           [P][S]
  measure                       fraction of output difference restored  [P][S]
  primary quantity              style curve minus placebo curve         [A][P]
  scenarios                     20, held out                            [A]
  phrasings per dimension       5                                       [A]
  dimensions                    fluency, confidence, health literacy    [A][P]
  items per dimension           100 (= 20 x 5)                          [A]
  placebo pairs                 3                                       [A]
  placebo items                 60 (= 3 x 20)                           [A]
  multiple comparisons          Benjamini-Hochberg across 28 layers     [A][P]
  magnitude floor               +/-0.03, pre-specified                  [A][P]
  reporting rule                BH q<0.05 AND |diff| >= 0.03            [A]
  sanity control 1              self-patch must return ~0.000           [P][S]
  sanity control 2              final-layer patch must return ~1.000    [P][S]
  divergence                    Jensen-Shannon, bits                    [S]
  split seed                    random.Random(0)                        [C]
  output schema                 keys of localisation_1.5B.json          [A]

WHAT IS NOT RECOVERED -- these are reconstruction choices, flagged in the output
-------------------------------------------------------------------------------
  1. WHICH 5 PHRASINGS. The artifact records "5 phrasings" but not which. This
     script takes the first five, which is the split convention every other
     script here uses (`v[:5]`). Patching runs on the base model with no
     adapter, so no train/test leakage rides on the choice, but the measured
     magnitude may shift if the original used a different five.
  2. WHICH 3 PLACEBO PAIRS. Not recorded. Reconstructed from the stem and
     modifier pool documented in Appendix B, one pair per family under the
     family-constrained rule (arrival mode / waiting behaviour / booking
     method). The 0.5B script used a single ad-hoc pair that is not in that
     pool, so it is not a guide.
  3. BOOTSTRAP RESAMPLES. Not recorded. Set to 4000 with seed 0, matching
     experiments/01_baseline.py. The reported CI half-widths (~0.016 at peak)
     are consistent with this order of magnitude but do not pin it.
  4. BOOTSTRAP UNIT. Not recorded. Style and placebo items are resampled
     independently and the difference taken, which is the natural estimator for
     a difference between two independently drawn item sets.
  5. PRECISION. Whether the original run used float32 or float16 is not
     recorded. This script defaults to float32, as the 0.5B sibling does.

Each of these is exposed as a command-line flag so the reconstruction can be
re-run under alternatives without editing source.

RUNNING
-------
    python experiments/12_patching_1.5b.py --selftest      # no model needed
    python experiments/12_patching_1.5b.py                 # the real run
    python tools/verify_patching.py                        # compare to the artifact

`--selftest` exercises the statistics layer -- bootstrap, Benjamini-Hochberg,
the magnitude floor, the reporting rule and the output schema -- against
synthetic curves with known answers. It touches no model and needs no GPU, so
it verifies the analysis half of this script independently of the forward pass.

The real run needs a GPU with the 1.5B model. On CPU it is possible but slow:
28 layers x 160 items x 2 forward passes is roughly 9,000 model calls.
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys

# ------------------------------------------------------------------ statistics
# Kept import-light and model-free so --selftest can exercise them alone.
import numpy as np


def bh_reject(pvals, alpha=0.05):
    """Benjamini-Hochberg. Returns a boolean array and the adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    # enforce monotonicity from the largest p downwards
    q = np.minimum.accumulate(q[::-1])[::-1]
    q_full = np.empty(n)
    q_full[order] = np.clip(q, 0, 1)
    return q_full <= alpha, q_full


def boot_diff(style_rows, plac_rows, n_boot, seed):
    """Bootstrap the per-layer difference of two independently sampled item sets.

    style_rows : (n_style, n_layers) restoration curves
    plac_rows  : (n_plac,  n_layers)
    Returns (diff, lo, hi, p_two_sided) each of length n_layers.
    """
    rs = np.random.default_rng(seed)
    ns, npl = len(style_rows), len(plac_rows)
    diff = style_rows.mean(0) - plac_rows.mean(0)
    draws = np.empty((n_boot, style_rows.shape[1]))
    for b in range(n_boot):
        si = rs.integers(0, ns, ns)
        pi = rs.integers(0, npl, npl)
        draws[b] = style_rows[si].mean(0) - plac_rows[pi].mean(0)
    lo = np.percentile(draws, 2.5, axis=0)
    hi = np.percentile(draws, 97.5, axis=0)
    # two-sided bootstrap p: proportion of resamples on the other side of zero,
    # doubled, with the standard +1 correction so p is never exactly 0
    frac = np.minimum((draws <= 0).mean(0), (draws >= 0).mean(0))
    p = np.clip(2 * (frac * n_boot + 1) / (n_boot + 1), 0, 1)
    return diff, lo, hi, p


def summarise(curves, n_boot, seed, floor, alpha, n_layers):
    """Turn raw restoration curves into the reported structure."""
    plac = curves["placebo"]
    out = {"per_dimension": {}, "magnitude_floor": floor, "alpha": alpha}
    for dim, rows in curves.items():
        if dim == "placebo":
            continue
        diff, lo, hi, p = boot_diff(rows, plac, n_boot, seed)
        rej, q = bh_reject(p, alpha)
        meaningful_pos = [int(L) for L in range(n_layers)
                          if rej[L] and diff[L] >= floor]
        meaningful_neg = [int(L) for L in range(n_layers)
                          if rej[L] and diff[L] <= -floor]
        peak = int(np.argmax(diff))
        trough = int(np.argmin(diff))
        out["per_dimension"][dim] = {
            "n_items": int(len(rows)),
            "diff": [round(float(x), 4) for x in diff],
            "ci_lo": [round(float(x), 4) for x in lo],
            "ci_hi": [round(float(x), 4) for x in hi],
            "boot_p": [float(x) for x in p],
            "bh_q": [float(x) for x in q],
            "bh_reject": [bool(x) for x in rej],
            "meaningful_positive_layers": meaningful_pos,
            "meaningful_negative_layers": meaningful_neg,
            "peak": {"layer": peak, "diff": round(float(diff[peak]), 3),
                     "ci": [round(float(lo[peak]), 3), round(float(hi[peak]), 3)],
                     "below_floor": bool(abs(diff[peak]) < floor)},
            "trough": {"layer": trough, "diff": round(float(diff[trough]), 3),
                       "ci": [round(float(lo[trough]), 3), round(float(hi[trough]), 3)]},
        }
    out["placebo_n_items"] = int(len(plac))
    return out


# --------------------------------------------------------------------- selftest
def selftest():
    """Exercise the statistics layer against curves with known answers.

    No model, no GPU, no network. Verifies that:
      * BH is monotone, conservative relative to uncorrected p, and correct on
        a textbook example
      * a planted positive bump is recovered at the right layers
      * a planted effect below the magnitude floor is excluded despite
        being statistically significant -- the reporting rule that produced
        "health literacy: no meaningful positive layers" in the artifact
      * a planted negative region is recovered
      * pure noise yields no meaningful layers
      * the output schema carries every key the reported artifact uses
    """
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))

    # --- BH against the published worked example: Benjamini & Hochberg (1995),
    #     Table 1. At alpha=0.05 the procedure rejects the first four hypotheses,
    #     where Bonferroni rejects three. Independently checkable.
    p = [.0001, .0004, .0019, .0095, .0201, .0278, .0298, .0344, .0459,
         .3240, .4262, .5719, .6528, .7590, 1.000]
    rej, q = bh_reject(p, 0.05)
    check("BH rejects exactly 4 on Benjamini & Hochberg (1995) Table 1",
          int(rej.sum()) == 4 and list(rej[:4]) == [True] * 4, f"rejected {int(rej.sum())}")
    check("BH q-values match the published values to 4dp",
          np.allclose(q[:5], [0.0015, 0.0030, 0.0095, 0.0356, 0.0603], atol=5e-5),
          str([round(float(x), 4) for x in q[:5]]))
    check("BH q-values are monotone non-decreasing", np.all(np.diff(q) >= -1e-12))
    check("BH is never more liberal than uncorrected p", np.all(q >= np.asarray(p) - 1e-12))

    NL = 28
    rs = np.random.default_rng(0)

    # --- planted positive bump at layers 3-8, well above the floor
    plac = rs.normal(0.5, 0.05, (60, NL))
    style = rs.normal(0.5, 0.05, (100, NL))
    style[:, 3:9] += 0.07
    res = summarise({"placebo": plac, "fluency": style}, 2000, 0, 0.03, 0.05, NL)
    f = res["per_dimension"]["fluency"]
    check("planted +0.07 bump recovered at layers 3-8",
          f["meaningful_positive_layers"] == [3, 4, 5, 6, 7, 8],
          str(f["meaningful_positive_layers"]))
    check("peak lands inside the planted region", 3 <= f["peak"]["layer"] <= 8)
    check("peak CI excludes zero", f["peak"]["ci"][0] > 0)

    # --- effect real but below the magnitude floor: significant, not reported
    style2 = rs.normal(0.5, 0.02, (100, NL))
    style2[:, 4] += 0.02                       # tiny, tight -> significant
    plac2 = rs.normal(0.5, 0.02, (60, NL))
    res2 = summarise({"placebo": plac2, "health_literacy": style2}, 2000, 0, 0.03, 0.05, NL)
    h = res2["per_dimension"]["health_literacy"]
    check("sub-floor effect is significant but NOT reported",
          h["bh_reject"][4] and h["meaningful_positive_layers"] == [],
          f"q={h['bh_q'][4]:.4f} diff={h['diff'][4]:.4f}")
    check("sub-floor peak is flagged below_floor", h["peak"]["below_floor"])

    # --- planted negative region
    style3 = rs.normal(0.5, 0.05, (100, NL))
    style3[:, 10:20] -= 0.20
    res3 = summarise({"placebo": plac, "confidence": style3}, 2000, 0, 0.03, 0.05, NL)
    c = res3["per_dimension"]["confidence"]
    check("planted negative region recovered at layers 10-19",
          c["meaningful_negative_layers"] == list(range(10, 20)),
          str(c["meaningful_negative_layers"]))
    check("trough lands inside the planted region", 10 <= c["trough"]["layer"] <= 19)

    # --- null: no planted effect
    res4 = summarise({"placebo": rs.normal(0.5, 0.05, (60, NL)),
                      "null": rs.normal(0.5, 0.05, (100, NL))}, 2000, 0, 0.03, 0.05, NL)
    n = res4["per_dimension"]["null"]
    check("pure noise yields no meaningful layers",
          n["meaningful_positive_layers"] == [] and n["meaningful_negative_layers"] == [])

    # --- schema
    need = {"n_items", "diff", "ci_lo", "ci_hi", "boot_p", "bh_q", "bh_reject",
            "meaningful_positive_layers", "meaningful_negative_layers", "peak", "trough"}
    check("output schema carries every reported key", need <= set(f.keys()),
          str(sorted(need - set(f.keys()))))
    check("curve arrays are one value per layer",
          all(len(f[k]) == NL for k in ("diff", "ci_lo", "ci_hi", "boot_p", "bh_q")))

    print("\n  SELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


# ----------------------------------------------------------------- placebo pool
# RECONSTRUCTED, not recovered. Appendix B documents the stem pool and the
# family-constrained rule: the first ten stems partition into three families
# (arrival mode / waiting behaviour / booking method) and both sides of a
# placebo pair must come from one family. One pair per family:
PLACEBO_PAIRS = [
    ("The patient arrived by bus on a cloudy morning.",
     "The patient came by taxi on a sunny afternoon."),
    ("The patient waited in reception during a quiet weekday.",
     "The patient sat in the lobby after a short wait."),
    ("The patient booked this visit online earlier than scheduled.",
     "The patient booked this visit by phone later than originally planned."),
]

DIMENSIONS = ["fluency", "confidence", "health_literacy"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true",
                    help="verify the statistics layer only; no model, no GPU")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-scenarios", type=int, default=20,
                    help="held-out scenarios (the artifact records 20)")
    ap.add_argument("--n-phrasings", type=int, default=5,
                    help="phrasings per dimension (the artifact records 5)")
    ap.add_argument("--phrasing-offset", type=int, default=0,
                    help="which phrasings: offset..offset+n. Default 0, i.e. the "
                         "first five. The original run's choice is not recorded")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--floor", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/raw/localisation_1.5B_rerun.json")
    ap.add_argument("--dialense-ref", default="a1adecdd31fa6905583f7beb79e58eb4b062bc06")
    a = ap.parse_args()

    if a.selftest:
        print("SELFTEST -- statistics layer only, no model loaded\n")
        sys.exit(selftest())

    # -------------------------------------------------------------- heavy imports
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from style_erasure.cues import CUES

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, a.dtype)

    # -------------------------------------------------------------------- corpus
    if not os.path.isdir("DiaLense"):
        subprocess.run(["git", "init", "--quiet", "DiaLense"], check=True)
        subprocess.run(["git", "-C", "DiaLense", "remote", "add", "origin",
                        "https://github.com/SamhitaK10/DiaLense.git"], check=True)
        subprocess.run(["git", "-C", "DiaLense", "fetch", "--quiet", "--depth", "1",
                        "origin", a.dialense_ref], check=True)
        subprocess.run(["git", "-C", "DiaLense", "checkout", "--quiet", "FETCH_HEAD"],
                       check=True)

    facts, seen = [], set()
    for path in sorted(glob.glob("DiaLense/results/raw/*/conversations.jsonl")):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["scenario_id"] in seen:
                continue
            seen.add(r["scenario_id"])
            facts.append("Reported history: " + "; ".join(
                str(v).strip().rstrip(".") for v in r["latent_facts"].values()) + ".")

    # the project-wide split: fixed at Random(0), first 30 train, next 20 held out
    idx = list(range(len(facts)))
    random.Random(0).shuffle(idx)
    eval_scen = [facts[i] for i in idx[30:30 + a.n_scenarios]]
    print(f"scenarios: {len(eval_scen)} held out of {len(facts)}", flush=True)

    lo_p, hi_p = a.phrasing_offset, a.phrasing_offset + a.n_phrasings
    cues = {d: CUES[d][lo_p:hi_p] for d in DIMENSIONS}
    for d, v in cues.items():
        print(f"  {d}: {len(v)} phrasings [{lo_p}:{hi_p}]")

    # --------------------------------------------------------------------- model
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(device)
    model.eval()
    LAYERS = model.model.layers
    NL = len(LAYERS)
    print(f"{a.model} | {NL} layers | {device} | {a.dtype}\n", flush=True)

    SYSTEM = "You are a physician taking a patient history. Ask one follow-up question."

    def chat(t):
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}],
            tokenize=False, add_generation_prompt=True)

    def ids_of(text):
        return tok(chat(text), return_tensors="pt").input_ids.to(device)

    def js(p, q):
        m = 0.5 * (p + q)
        kl = lambda x, y: (x * (torch.log2(x.clamp_min(1e-12)) - torch.log2(y.clamp_min(1e-12)))).sum()
        return float(0.5 * kl(p, m) + 0.5 * kl(q, m))

    @torch.no_grad()
    def run(text, donors=None, patch_layer=None):
        handle = None
        if patch_layer is not None:
            def hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h[0, -1, :] = donors[patch_layer].to(h.dtype)
                return (h,) + out[1:] if isinstance(out, tuple) else h
            handle = LAYERS[patch_layer].register_forward_hook(hook)
        try:
            logits = model(ids_of(text)).logits[0, -1].float()
        finally:
            if handle is not None:
                handle.remove()
        return torch.softmax(logits, -1)

    @torch.no_grad()
    def cache_donors(text):
        store, hs = {}, []

        def mk(i):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                store[i] = h[0, -1, :].detach().clone()
            return hook

        for i, layer in enumerate(LAYERS):
            hs.append(layer.register_forward_hook(mk(i)))
        try:
            model(ids_of(text))
        finally:
            for h in hs:
                h.remove()
        return store

    # ------------------------------------------------------------ sanity controls
    sh, sl = CUES["health_literacy"][0]
    F = eval_scen[0]
    hi_txt, lo_txt = f"{sh} {F}", f"{sl} {F}"
    P_hi = run(hi_txt)
    base = js(run(lo_txt), P_hi)
    self_patch = 1 - js(run(lo_txt, cache_donors(lo_txt), NL - 1), P_hi) / base
    last_patch = 1 - js(run(lo_txt, cache_donors(hi_txt), NL - 1), P_hi) / base
    print("=" * 66)
    print("SANITY CONTROLS")
    print(f"  patch with its OWN activation    : {self_patch:+.3f}   must be ~0")
    print(f"  patch final layer from the donor : {last_patch:+.3f}   must be ~1")
    if abs(self_patch) > 0.05 or last_patch < 0.8:
        raise SystemExit("  CONTROLS FAILED - hooks are wired wrong. Stop.")
    print("  -> hooks verified\n", flush=True)

    # ----------------------------------------------------------------- layer scan
    def scan(pairs, label):
        rows = []
        for hi_s, lo_s in pairs:
            for F in eval_scen:
                ta, tb = f"{hi_s} {F}", f"{lo_s} {F}"
                P_a, P_b = run(ta), run(tb)
                bs = js(P_b, P_a)
                if bs < 1e-6:
                    continue
                donors = cache_donors(ta)
                rows.append([1 - js(run(tb, donors, L), P_a) / bs for L in range(NL)])
        print(f"   {label}: {len(rows)} items", flush=True)
        return np.array(rows)

    curves = {"placebo": scan(PLACEBO_PAIRS, "placebo")}
    for dim in DIMENSIONS:
        curves[dim] = scan(cues[dim], dim)

    # ------------------------------------------------------------------- analysis
    res = summarise(curves, a.n_boot, a.seed, a.floor, a.alpha, NL)
    res["_status"] = ("RECONSTRUCTION -- the original script was not saved. See the "
                      "header of experiments/12_patching_1.5b.py for what is recovered "
                      "evidence and what is a reconstruction choice.")
    res["_measure"] = ("style-minus-placebo restoration, activation patching at the "
                       "final prompt position, 1.5B base model")
    res["_reporting_rule"] = f"A layer is reported only if BH q<{a.alpha} AND |difference| >= {a.floor}."
    res["_sanity"] = {"self_patch": round(self_patch, 4), "final_layer_patch": round(last_patch, 4)}
    res["_config"] = {
        "model": a.model, "n_layers": NL, "device": device, "dtype": a.dtype,
        "n_scenarios": len(eval_scen), "split_seed": 0,
        "phrasings": f"[{lo_p}:{hi_p}]  (NOT RECOVERED -- reconstruction choice)",
        "placebo_pairs": len(PLACEBO_PAIRS),
        "placebo_source": "RECONSTRUCTED from the Appendix B stem pool, family-constrained",
        "n_boot": a.n_boot, "boot_seed": a.seed,
        "boot_unit": "item; style and placebo resampled independently (NOT RECOVERED)",
        "magnitude_floor": a.floor, "alpha": a.alpha,
        "dialense_ref": a.dialense_ref,
    }

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    print("\nNow run:  python tools/verify_patching.py")

    for dim in DIMENSIONS:
        d = res["per_dimension"][dim]
        print(f"\n{dim}: peak {d['peak']['diff']:+.3f} (L{d['peak']['layer']}) "
              f"CI {d['peak']['ci']}   positive layers {d['meaningful_positive_layers']}")
        print(f"          trough {d['trough']['diff']:+.3f} (L{d['trough']['layer']})")


if __name__ == "__main__":
    main()
