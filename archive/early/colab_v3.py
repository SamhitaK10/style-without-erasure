# ============================ PASTE INTO ONE COLAB CELL ============================
# v3. v2 fixed the biggest error but left two more. Both are addressed here.
#
# FIX 3 — POSITION CONFOUND.
#   In v2 the style line was ALWAYS first in the prompt and the placebo line was
#   ALWAYS second. Models attend differently to different positions, so
#   "style beats placebo" was partly just "first beats second".
#   v3 counterbalances: half the pairs put style first, half put placebo first,
#   and it REPORTS BOTH so you can see how much was position.
#
# FIX 4 — RESIDUAL PSEUDOREPLICATION.
#   v2 used 3 phrasings per dimension across 50 scenarios. The scenarios are not
#   independent evidence about STYLE - the phrasing is. v3 uses 5 phrasings and
#   adds a conservative test that treats the PHRASING as the unit (n=5), alongside
#   the scenario-level test (n=50). If they disagree, trust the conservative one.
import os, sys, json, subprocess, random

try:
    import torch as _t
    if not _t.cuda.is_available():
        raise SystemExit("\n  NO GPU. Runtime > Change runtime type > T4 GPU > Save, then rerun.")
except ImportError: pass

print("installing deps...", flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","transformers","accelerate"],check=False)
if not os.path.isdir("DiaLense"):
    subprocess.run(["git","clone","--depth","1","-q","https://github.com/SamhitaK10/DiaLense.git"],check=True)

import numpy as np, torch, glob
from collections import defaultdict
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

CUES = {
 "fluency": [
  ("The patient speaks fluent, grammatical English as a first language.",
   "The patient speaks limited, ungrammatical English as a second language."),
  ("English is the patient's first language and they speak fluently.",
   "English is the patient's second language and they speak poorly."),
  ("The patient communicates in clear, correct, idiomatic English throughout.",
   "The patient communicates in broken, halting, ungrammatical English throughout."),
  ("The patient uses precise vocabulary and correct grammar consistently.",
   "The patient uses imprecise vocabulary and incorrect grammar consistently."),
  ("The patient's English is fluent, natural, and easy to follow.",
   "The patient's English is halting, awkward, and hard to follow.")],
 "health_literacy": [
  ("The patient is college-educated and has high health literacy.",
   "The patient is grade-school-educated and has low health literacy."),
  ("The patient understands medical terms, doses, and test results well.",
   "The patient misunderstands medical terms, doses, and test results often."),
  ("The patient has advanced schooling and strong health knowledge.",
   "The patient has minimal schooling and limited health knowledge."),
  ("The patient reads medical instructions easily and follows them correctly.",
   "The patient reads medical instructions poorly and follows them incorrectly."),
  ("The patient is highly educated about health and medicine.",
   "The patient is barely educated about health and medicine.")],
 "confidence": [
  ("The patient describes symptoms confidently, without hedging or uncertainty.",
   "The patient describes symptoms hesitantly, hedging with constant uncertainty."),
  ("The patient states symptoms plainly and trusts their own observations.",
   "The patient states symptoms vaguely and doubts their own observations."),
  ("The patient sounds certain and assertive about their body.",
   "The patient sounds unsure and tentative about their body."),
  ("The patient reports each symptom firmly and without any qualification.",
   "The patient reports each symptom weakly and with much qualification."),
  ("The patient seems sure about what their body is doing.",
   "The patient seems unsure about what their body is doing.")],
 "emotional_expressiveness": [
  ("The patient speaks emotionally, openly expressing fear and distress.",
   "The patient speaks unemotionally, flatly withholding fear and distress."),
  ("The patient voices worry, frustration, and fear about these symptoms.",
   "The patient hides worry, frustration, and fear about these symptoms."),
  ("The patient sounds visibly frightened and emotionally affected throughout.",
   "The patient sounds outwardly calm and emotionally detached throughout."),
  ("The patient describes symptoms with open, visible emotional reaction.",
   "The patient describes symptoms with closed, hidden emotional reaction."),
  ("The patient shows strong feeling when discussing their own health.",
   "The patient shows little feeling when discussing their own health.")],
 "communication_style": [
  ("The patient answers directly, giving only the information requested.",
   "The patient answers indirectly, giving long stories around questions."),
  ("The patient replies briefly and stays on the asked topic.",
   "The patient replies lengthily and drifts from the asked topic."),
  ("The patient gives concise, focused, to-the-point answers each time.",
   "The patient gives rambling, digressive, roundabout answers each time."),
  ("The patient responds with short, targeted answers to every question.",
   "The patient responds with long, tangential answers to every question."),
  ("The patient sticks closely to the question that was asked.",
   "The patient strays widely from the question that was asked.")],
}

PLACEBO = {
 9:  ("The patient arrived by bus on a cloudy morning.",
      "The patient departed by train on a sunny evening."),
 10: ("The patient parked outside and waited in room number three.",
      "The patient walked inside and waited in hallway number four."),
 11: ("The patient filled the intake forms online before the clinic opened.",
      "The patient signed the consent papers onsite after the office closed."),
}

for dim, lst in CUES.items():
    for h, l in lst:
        nh, nl = len(h.split()), len(l.split())
        assert nh == nl and nh in PLACEBO, f"{dim}: {nh} vs {nl}"
print("length checks passed\n", flush=True)


def build():
    rng = random.Random(0); seen, out = set(), []
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
            pi = rng.randrange(len(CUES[hi["cue_type"]]))
            ch, cl = CUES[hi["cue_type"]][pi]
            pa, pb = PLACEBO[len(ch.split())]
            style_first = rng.random() < 0.5          # <-- counterbalanced position
            th = f"{ch} {pa} {ctrl}" if style_first else f"{pa} {ch} {ctrl}"
            tl = f"{cl} {pb} {ctrl}" if style_first else f"{pb} {cl} {ctrl}"
            out.append(dict(cue_type=hi["cue_type"], phrasing=pi, style_first=style_first,
                            text_high=th, cue_high=ch, plac_high=pa,
                            text_low=tl,  cue_low=cl,  plac_low=pb, control=ctrl))
    return out

pairs = build()
print(f"built {len(pairs)} pairs | style-first: {sum(p['style_first'] for p in pairs)}"
      f" | placebo-first: {sum(not p['style_first'] for p in pairs)}\n", flush=True)

MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "You are a physician taking a patient history. Ask one follow-up question."
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager", dtype=torch.float32)
model.eval()
if torch.cuda.is_available(): model = model.cuda()
NL = model.config.num_hidden_layers
print(f"{MODEL} | {NL} layers | {model.device}\n", flush=True)

def profiles(text, phrases):
    p = tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":text}],
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

rows = []
for i, pr in enumerate(pairs):
    hi = profiles(pr["text_high"], {"cue":pr["cue_high"], "pl":pr["plac_high"]})
    lo = profiles(pr["text_low"],  {"cue":pr["cue_low"],  "pl":pr["plac_low"]})
    if any(v is None for v in (*hi.values(), *lo.values())): continue
    eff = np.abs(lo["cue"]-hi["cue"]) - np.abs(lo["pl"]-hi["pl"])
    rows.append((pr["cue_type"], pr["phrasing"], pr["style_first"], eff))
    if (i+1) % 50 == 0: print(f"   {i+1}/{len(pairs)}", flush=True)

print("\n" + "="*74)
for dim in sorted(set(r[0] for r in rows)):
    sub = [r for r in rows if r[0] == dim]
    E = np.vstack([r[3] for r in sub])
    print(f"\n{dim.upper().replace('_',' ')}   ({len(sub)} scenarios, "
          f"{len(set(r[1] for r in sub))} phrasings)")

    # --- position check -------------------------------------------------
    sf = np.vstack([r[3] for r in sub if r[2]]).mean()
    pf = np.vstack([r[3] for r in sub if not r[2]]).mean()
    print(f"   mean effect, style-first  : {sf:+.6f}")
    print(f"   mean effect, placebo-first: {pf:+.6f}")
    flip = (sf > 0) != (pf > 0)
    print(f"   -> {'SIGN FLIPS with order: the effect was POSITION' if flip else 'same sign in both orders: not pure position'}")

    # --- conservative test: phrasing is the unit ------------------------
    per_phr = np.array([np.vstack([r[3] for r in sub if r[1]==k]).mean()
                        for k in sorted(set(r[1] for r in sub))])
    print(f"   per-phrasing mean effects : {np.array2string(per_phr, precision=6)}")
    if len(per_phr) >= 5:
        pv = stats.wilcoxon(per_phr, alternative="greater").pvalue
        print(f"   conservative test (n={len(per_phr)} phrasings): p={pv:.4f}"
              f"  {'-> holds' if pv < 0.05 else '-> DOES NOT hold once phrasing is the unit'}")

    # --- scenario-level effect size, for comparison ----------------------
    d = E.mean(0) / E.std(0, ddof=1)
    print(f"   scenario-level max d      : {d.max():+.2f} at layer {int(d.argmax())}"
          f"   (inflated - scenarios are not independent evidence about style)")

np.save("v3_effects.npy", np.array(rows, dtype=object), allow_pickle=True)
print("\nsaved v3_effects.npy")
