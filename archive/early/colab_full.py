import os, sys, json, subprocess

# ---- GPU CHECK FIRST. On CPU this takes ~2.5 hours instead of ~8 minutes. ----
try:
    import torch as _t
    if not _t.cuda.is_available():
        raise SystemExit(
            "\n  NO GPU.\n"
            "  Go to  Runtime > Change runtime type > T4 GPU > Save,\n"
            "  then run this cell again.\n"
            "  (Stopping now so you don't wait 2.5 hours on CPU.)")
except ImportError:
    pass

print("installing deps...", flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","transformers","accelerate"],check=False)
if not os.path.isdir("DiaLense"):
    print("cloning DiaLense...", flush=True)
    subprocess.run(["git","clone","--depth","1","-q","https://github.com/SamhitaK10/DiaLense.git"],check=True)

import numpy as np, torch, glob
from collections import defaultdict
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

CUES = {
 "fluency": ("The patient speaks fluent, grammatical English as a first language.",
             "The patient speaks limited, ungrammatical English as a second language."),
 "health_literacy": ("The patient is college-educated and has high health literacy.",
                     "The patient left school early and has low health literacy."),
 "confidence": ("The patient describes symptoms confidently, without hedging or uncertainty.",
                "The patient describes symptoms hesitantly, hedging with constant uncertainty."),
 "emotional_expressiveness": ("The patient speaks emotionally, openly expressing fear and distress.",
                              "The patient speaks unemotionally, flatly withholding fear and distress."),
 "communication_style": ("The patient answers directly, giving only the information requested.",
                         "The patient answers indirectly, giving long stories around the question."),
}

def build_pairs():
    seen, out = set(), []
    files = sorted(glob.glob("DiaLense/results/raw/*/conversations.jsonl"))
    print("transcript files found:", len(files), flush=True)
    for path in files:
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
            ch, cl = CUES[hi["cue_type"]]
            out.append(dict(pair_id=pid, cue_type=hi["cue_type"],
                            text_high=f"{ch} {ctrl}", cue_high=ch,
                            text_low=f"{cl} {ctrl}",  cue_low=cl, control=ctrl))
    return out

pairs = build_pairs()
by = defaultdict(int)
for r in pairs: by[r["cue_type"]] += 1
print(f"built {len(pairs)} matched pairs: {dict(by)}\n", flush=True)

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
    for name, phrase in phrases.items():
        if phrase not in p: res[name] = None; continue
        s = p.index(phrase); e = s + len(phrase)
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
    big = [r["layer"] for r in sig if r["d"] >= 0.5]
    print(f"\n{label}  (n={n})")
    print(f"   significant layers : {len(sig)}/{len(rows)}")
    print(f"   d>=0.5 (sizeable)  : {big or 'none'}")
    if sig:
        b = max(sig, key=lambda r: r["d"])
        print(f"   strongest: layer {b['layer']}  d={b['d']:+.2f}  q={b['q']:.4f}")
    else:
        print("   -> nothing above the control artifact")

res, skipped = defaultdict(list), 0
for i, pr in enumerate(pairs):
    hi = profiles(pr["text_high"], {"cue": pr["cue_high"], "ctl": pr["control"]})
    lo = profiles(pr["text_low"],  {"cue": pr["cue_low"],  "ctl": pr["control"]})
    if any(v is None for v in (*hi.values(), *lo.values())): skipped += 1; continue
    res[pr["cue_type"]].append(np.abs(lo["cue"]-hi["cue"]) - np.abs(lo["ctl"]-hi["ctl"]))
    if (i+1) % 50 == 0: print(f"   {i+1}/{len(pairs)}", flush=True)

print(f"\ndone. skipped={skipped}")
print("="*64); print("EACH STYLE DIMENSION TESTED SEPARATELY"); print("="*64)
for cue, e in sorted(res.items()):
    if len(e) >= 10: summarize(analyze(np.vstack(e)), cue.upper().replace("_"," "), len(e))
pooled = np.vstack([x for e in res.values() for x in e])
print("\n" + "="*64)
summarize(analyze(pooled), "POOLED (all dimensions)", len(pooled))
np.save("effects_by_cue.npy", {k: np.vstack(v) for k,v in res.items()}, allow_pickle=True)
print("\nsaved effects_by_cue.npy")
