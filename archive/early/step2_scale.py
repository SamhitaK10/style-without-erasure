"""
STEP 2 - Run the attention test across ALL your matched pairs, with real statistics.

What changed from step 1:
  * many pairs instead of one sentence
  * the PAIR is the unit of analysis, not the layer (layers are not independent)
  * multiple-comparison correction across layers
  * effect sizes and confidence intervals, not just p-values
  * a self-test that proves the pipeline does NOT invent signal

Usage:
    python step2_scale.py --selftest            # no model needed, verifies the stats
    python step2_scale.py --data pairs.jsonl
    python step2_scale.py --data pairs.jsonl --limit 100     # quick trial run

Expected input: JSONL, one matched pair per line:
{"pair_id":"s001",
 "text_high":"I'm a software engineer... chest pain for three days...",
 "cue_high":"I'm a software engineer",
 "text_low":"Sorry my English is not so good... chest pain for three days...",
 "cue_low":"Sorry my English is not so good",
 "control":"chest pain for three days"}

`control` MUST appear verbatim in both texts. That is the whole point.
"""

import argparse, json, sys
import numpy as np
from scipy import stats

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "You are a physician taking a patient history. Ask one follow-up question."


# ---------------------------------------------------------------- statistics
def analyze(effects, layer_names=None, alpha=0.05):
    """
    effects: (n_pairs, n_layers) array of  style_gap - control_gap.
    Positive means the model reacted to speaking style MORE than to identical text.

    One test per layer, across pairs. Pairs are independent; layers are not.
    """
    effects = np.asarray(effects, dtype=float)
    n_pairs, n_layers = effects.shape
    rows = []

    for L in range(n_layers):
        x = effects[:, L]
        x = x[np.isfinite(x)]
        if len(x) < 5:
            continue

        # Wilcoxon: is the median effect above zero? (no normality assumption)
        try:
            p = stats.wilcoxon(x, alternative="greater").pvalue
        except ValueError:      # all zeros
            p = 1.0

        d = x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0   # Cohen's d

        boot = np.random.default_rng(0).choice(x, (4000, len(x)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])

        rows.append(dict(layer=L, n=len(x), mean=x.mean(), ci_lo=lo, ci_hi=hi,
                         cohens_d=d, p_raw=p))

    # Benjamini-Hochberg: 28 tests means ~1.4 false positives at p<.05 uncorrected
    ps = np.array([r["p_raw"] for r in rows])
    order = np.argsort(ps)
    m = len(ps)
    q = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        prev = min(prev, ps[idx] * m / (m - rank + 1))
        q[idx] = prev
    for r, qq in zip(rows, q):
        r["q_bh"] = qq
        r["significant"] = qq < alpha
    return rows


def report(rows, label=""):
    print(f"\n{'layer':>5} {'mean effect':>13} {'95% CI':>24} {'d':>7} {'q(BH)':>9}  sig")
    print("-" * 74)
    for r in rows:
        ci = f"[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]"
        print(f"{r['layer']:>5} {r['mean']:>+13.6f} {ci:>24} {r['cohens_d']:>+7.2f} "
              f"{r['q_bh']:>9.4f}  {'YES' if r['significant'] else '-'}")

    sig = [r for r in rows if r["significant"]]
    n = rows[0]["n"] if rows else 0
    print(f"\n{label}")
    print(f"pairs analyzed: {n}")
    print(f"layers with a real effect after correction: {len(sig)}/{len(rows)}")
    if sig:
        best = max(sig, key=lambda r: abs(r["cohens_d"]))
        print(f"strongest: layer {best['layer']}, d={best['cohens_d']:+.2f}, q={best['q_bh']:.4f}")
        big = [r['layer'] for r in sig if abs(r['cohens_d']) >= 0.5]
        print(f"layers with d>=0.5 (not just significant, actually sizeable): {big or 'none'}")
        if not big:
            print("  -> significant but tiny. With enough pairs, trivial effects pass p-tests.")
            print("     Report the effect size, not the p-value.")
    else:
        print("No layer survives correction. The model does not measurably")
        print("attend to speaking style more than to identical text.")
        print("That is a publishable negative result, not a failed experiment.")


# ---------------------------------------------------------------- self-test
def selftest():
    rng = np.random.default_rng(42)
    n_pairs, n_layers = 200, 28

    print("=" * 74)
    print("SELF-TEST A: data with NO real effect (pure noise)")
    print("A trustworthy pipeline must find nothing here.")
    print("=" * 74)
    report(analyze(rng.normal(0, 1e-3, (n_pairs, n_layers))), "NULL DATA")

    print("\n" + "=" * 74)
    print("SELF-TEST B: planted effect in layers 10-14 only")
    print("A trustworthy pipeline must find those and not the others.")
    print("=" * 74)
    e = rng.normal(0, 1e-3, (n_pairs, n_layers))
    e[:, 10:15] += 5e-4
    report(analyze(e), "PLANTED EFFECT IN LAYERS 10-14")


# ---------------------------------------------------------------- model path
def run_real(data_path, limit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", torch_dtype=torch.float32)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    n_layers = model.config.num_hidden_layers

    def prompt_for(text):
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True)

    def profiles(text, phrases):
        """ONE forward pass, attention profile for several phrases. 2x faster."""
        p = prompt_for(text)
        enc = tok(p, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_attentions=True)
        last = enc["input_ids"].shape[-1] - 1

        results = {}
        for name, phrase in phrases.items():
            if phrase not in p:
                results[name] = None
                continue
            s = p.index(phrase); e = s + len(phrase)
            pos = [i for i, (a, b) in enumerate(offs) if a < e and b > s]
            if not pos:
                results[name] = None
                continue
            prof = [ (out.attentions[L][0][:, last, pos].sum(-1).mean() / len(pos)).item()
                     for L in range(n_layers) ]
            results[name] = np.array(prof)
        return results

    pairs = [json.loads(l) for l in open(data_path) if l.strip()]
    if limit:
        pairs = pairs[:limit]

    effects, skipped = [], 0
    for i, pr in enumerate(pairs):
        try:
            hi = profiles(pr["text_high"], {"cue": pr["cue_high"], "ctl": pr["control"]})
            lo = profiles(pr["text_low"],  {"cue": pr["cue_low"],  "ctl": pr["control"]})
        except Exception as ex:
            skipped += 1; continue
        if any(v is None for v in (*hi.values(), *lo.values())):
            skipped += 1; continue

        style_gap = np.abs(lo["cue"] - hi["cue"])
        ctrl_gap  = np.abs(lo["ctl"] - hi["ctl"])
        effects.append(style_gap - ctrl_gap)

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pairs)} pairs  ({skipped} skipped)", flush=True)

    if len(effects) < 5:
        sys.exit(f"Only {len(effects)} usable pairs. Check that `control` appears "
                 f"verbatim in both texts of each pair.")

    print(f"\nusable pairs: {len(effects)}   skipped: {skipped}")
    eff = np.vstack(effects)
    np.save("effects.npy", eff)
    report(analyze(eff), "REAL DATA")
    print("\nsaved per-pair effects to effects.npy")
    print("Rerun this identical script after fine-tuning and compare.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.data:
        run_real(a.data, a.limit)
    else:
        ap.error("pass --data pairs.jsonl or --selftest")
