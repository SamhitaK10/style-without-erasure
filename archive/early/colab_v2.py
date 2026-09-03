# ============================ PASTE INTO ONE COLAB CELL ============================
# v2. Fixes two design errors in v1 that produced impossible effect sizes (d=12+).
#
# FIX 1 — PLACEBO CUE.
#   v1 compared attention to two DIFFERENT style strings against attention to one
#   IDENTICAL facts block. Different strings always draw different attention, so
#   v1's "effect" was guaranteed before the model ran.
#   v2 adds a style-IRRELEVANT phrase pair that differs just as much as the style
#   pair does, and is word-count matched to it. The real question becomes:
#       does the STYLE difference exceed the IRRELEVANT difference?
#
# FIX 2 — PSEUDOREPLICATION.
#   v1 used the same two cue strings for all 50 scenarios in a dimension, so
#   variance was near zero and d exploded. v2 rotates 3 different phrasings per
#   dimension, so the variance reflects real variation in how style is worded.
import os, sys, json, subprocess, random

try:
    import torch as _t
    if not _t.cuda.is_available():
        raise SystemExit("\n  NO GPU. Runtime > Change runtime type > T4 GPU > Save, then rerun.")
except ImportError:
    pass

print("installing deps...", flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","transformers","accelerate"],check=False)
if not os.path.isdir("DiaLense"):
    subprocess.run(["git","clone","--depth","1","-q","https://github.com/SamhitaK10/DiaLense.git"],check=True)

import numpy as np, torch, glob
from collections import defaultdict
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# 3 phrasings per dimension. Each (high, low) pair is word-count matched.
CUES = {
 "fluency": [
  ("The patient speaks fluent, grammatical English as a first language.",
   "The patient speaks limited, ungrammatical English as a second language."),
  ("English is the patient's first language and they speak it fluently.",
   "English is the patient's second language and they speak it poorly."),
  ("The patient communicates in clear, correct, idiomatic English throughout.",
   "The patient communicates in broken, halting, ungrammatical English throughout.")],
 "health_literacy": [
  ("The patient is college-educated and has high health literacy.",
   "The patient is grade-school-educated and has low health literacy."),
  ("The patient understands medical terms, doses, and test results well.",
   "The patient misunderstands medical terms, doses, and test results often."),
  ("The patient has advanced schooling and strong health knowledge.",
   "The patient has minimal schooling and limited health knowledge.")],
 "confidence": [
  ("The patient describes symptoms confidently, without hedging or uncertainty.",
   "The patient describes symptoms hesitantly, hedging with constant uncertainty."),
  ("The patient states symptoms plainly and trusts their own observations.",
   "The patient states symptoms vaguely and doubts their own observations."),
  ("The patient sounds certain and assertive about their body.",
   "The patient sounds unsure and tentative about their body.")],
 "emotional_expressiveness": [
  ("The patient speaks emotionally, openly expressing fear and distress.",
   "The patient speaks unemotionally, flatly withholding fear and distress."),
  ("The patient voices worry, frustration, and fear about these symptoms.",
   "The patient hides worry, frustration, and fear about these symptoms."),
  ("The patient sounds visibly frightened and emotionally affected throughout.",
   "The patient sounds outwardly calm and emotionally detached throughout.")],
 "communication_style": [
  ("The patient answers directly, giving only the information requested.",
   "The patient answers indirectly, giving long stories around questions."),
  ("The patient replies briefly and stays on the asked topic.",
   "The patient replies lengthily and drifts from the asked topic."),
  ("The patient gives concise, focused, to-the-point answers each time.",
   "The patient gives rambling, digressive, roundabout answers each time.")],
}

# Style-IRRELEVANT phrase pairs, indexed by word count so we can length-match.
PLACEBO = {
 9:  ("The patient arrived by bus on a cloudy morning.",
      "The patient departed by train on a sunny evening."),
 10: ("The patient parked outside and waited in room number three.",
      "The patient walked inside and waited in hallway number four."),
 11: ("The patient filled the intake forms online before the clinic opened.",
      "The patient signed the consent papers onsite after the office closed."),
}

# sanity: every cue pair must be length-matched and have a placebo of equal length
for dim, lst in CUES.items():
    for h, l in lst:
        nh, nl = len(h.split()), len(l.split())
        assert nh == nl, f"{dim}: cue lengths differ {nh} vs {nl}"
        assert nh in PLACEBO, f"{dim}: no placebo of length {nh}"
        pa, pb = PLACEBO[nh]
        assert len(pa.split()) == nh and len(pb.split()) == nh, f"placebo {nh} mismatch"
print("cue/placebo length checks passed\n", flush=True)


def build():
    rng = random.Random(0)
    seen, out = set(), []
    for path in sorted(glob.glob("DiaLense/results/raw/*/conversations.jsonl")):
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        pairs = defaultdict(dict)
        for r in recs: pairs[r["matched_pair_id"]][r["variant"]] = r
        for pid, p in pairs.items():
            if "high" not in p or "low" not in p or pid in seen: continue
            hi, lo = p["high"], p["low"]
            if hi["latent_facts_hash"] != lo["latent_facts_hash"]: continue
            if hi["cue_type"] not in CUES: continue
            seen.add(pid)
            ctrl = "Reported history: " + "; ".join(
                str(v).strip().rstrip(".") for v in hi["latent_facts"].values()) + "."
            ch, cl = rng.choice(CUES[hi["cue_type"]])       # rotate phrasing
            pa, pb = PLACEBO[len(ch.split())]
            out.append(dict(pair_id=pid, cue_type=hi["cue_type"],
                            text_high=f"{ch} {pa} {ctrl}", cue_high=ch, plac_high=pa,
                            text_low=f"{cl} {pb} {ctrl}",  cue_low=cl,  plac_low=pb,
                            control=ctrl))
    return out

pairs = build()
cnt = defaultdict(int)
for r in pairs: cnt[r["cue_type"]] += 1
print(f"built {len(pairs)} matched pairs: {dict(cnt)}\n", flush=True)

MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "You are a physician taking a patient history. Ask one follow-up question."
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager", dtype=torch.float32)
model.eval()
if torch.cuda.is_available(): model = model.cuda()
NL = model.config.num_hidden_layers
print(f"{MODEL} | {NL} layers | {model.device}\n", flush=True)

def profiles(text, phrases):
    p = tok.apply_chat_template([{"role":"system","content":SYSTEM},
                                 {"role":"user","content":text}],
                                tokenize=False, add_generation_prompt=True)
    enc = tok(p, return_tensors="pt", return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad(): out = model(**enc, output_attentions=True)
    last = enc["input_ids"].shape[-1] - 1
    res = {}
    for name, ph in phrases.items():
        if ph not in p: res[name] = None; continue
        s = p.index(ph); e = s + len(ph)
        pos = [i for i,(a,b) in enumerate(offs) if a < e and b > s]
        res[name] = None if not pos else np.array(
            [(out.attentions[L][0][:, last, pos].sum(-1).mean()/len(pos)).item() for L in range(NL)])
    return res

def analyze(eff, alpha=0.05):
    eff = np.asarray(eff, float); rows = []
    for L in range(eff.shape[1]):
        x = eff[:, L]
        p = stats.wilcoxon(x, alternative="greater").pvalue if x.std() > 0 else 1.0
        rows.append(dict(layer=L, mean=x.mean(),
                         d=x.mean()/x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0, p=p))
    ps = np.array([r["p"] for r in rows]); m = len(ps)
    order = np.argsort(ps); q = np.empty(m); prev = 1.0
    for rank, idx in enumerate(reversed(order), 1):
        prev = min(prev, ps[idx]*m/(m-rank+1)); q[idx] = prev
    for r, qq in zip(rows, q): r["q"] = qq; r["sig"] = qq < alpha
    return rows

def summarize(rows, label, n):
    sig = [r for r in rows if r["sig"]]
    big = [r["layer"] for r in sig if 0.2 <= r["d"] < 3.0]
    huge = [r["layer"] for r in sig if r["d"] >= 3.0]
    print(f"\n{label}  (n={n})")
    print(f"   significant layers        : {len(sig)}/{len(rows)}")
    print(f"   plausible effects (d 0.2-3): {big or 'none'}")
    if huge: print(f"   IMPLAUSIBLE d>=3 at {huge}  <-- suspect another artifact")
    if sig:
        b = max(sig, key=lambda r: r["d"])
        print(f"   strongest: layer {b['layer']}  d={b['d']:+.2f}  q={b['q']:.4f}")
    else:
        print("   -> style difference does NOT exceed the irrelevant-phrase difference")

style_v_plac, style_v_ctrl = defaultdict(list), defaultdict(list)
for i, pr in enumerate(pairs):
    hi = profiles(pr["text_high"], {"cue":pr["cue_high"], "pl":pr["plac_high"], "ctl":pr["control"]})
    lo = profiles(pr["text_low"],  {"cue":pr["cue_low"],  "pl":pr["plac_low"],  "ctl":pr["control"]})
    if any(v is None for v in (*hi.values(), *lo.values())): continue
    s = np.abs(lo["cue"]-hi["cue"]); pl = np.abs(lo["pl"]-hi["pl"]); c = np.abs(lo["ctl"]-hi["ctl"])
    style_v_plac[pr["cue_type"]].append(s - pl)     # THE REAL TEST
    style_v_ctrl[pr["cue_type"]].append(s - c)      # v1's broken test, for comparison
    if (i+1) % 50 == 0: print(f"   {i+1}/{len(pairs)}", flush=True)

print("\n" + "="*70)
print("THE REAL TEST: style difference MINUS irrelevant-phrase difference")
print("="*70)
for cue, e in sorted(style_v_plac.items()):
    summarize(analyze(np.vstack(e)), cue.upper().replace("_"," "), len(e))

print("\n" + "="*70)
print("FOR COMPARISON — v1's broken test (style minus identical-text control)")
print("If these stay huge while the real test above shrinks, that confirms")
print("v1 was measuring 'different words' rather than 'style'.")
print("="*70)
for cue, e in sorted(style_v_ctrl.items()):
    r = analyze(np.vstack(e)); b = max(r, key=lambda x: x["d"])
    print(f"   {cue:26s} strongest d={b['d']:+.2f}")

np.save("v2_style_vs_placebo.npy", {k: np.vstack(v) for k,v in style_v_plac.items()}, allow_pickle=True)
print("\nsaved v2_style_vs_placebo.npy")
