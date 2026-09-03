# =========================== PASTE INTO ONE COLAB CELL ===========================
# DiaLense Part II - baseline.  v5.  Three changes from v4:
#
# 1. MEASURES A WHOLE SENTENCE, NOT ONE TOKEN.
#    v4 compared only the next-token distribution. The first word of a clinician
#    question is nearly always "Can"/"Could"/"How", so that metric had a low
#    ceiling and made everything look similar in size. v5 generates a reference
#    question once per scenario from a NEUTRAL prompt (facts only), then asks how
#    differently each condition scores that same question, token by token.
#    Same continuation for every condition, so the comparison stays symmetric.
#
# 2. PLACEBOS ARE GENERATED, THEN MATCHED ON LENGTH *AND* SEMANTIC DISTANCE.
#    v4 hand-wrote 7 placebo pairs and dropped 17 of 25 phrasings that could not
#    be length-matched. v5 generates hundreds of irrelevant sentence pairs and,
#    for each style pair, picks the one that (a) matches BOTH token lengths
#    exactly and (b) has the closest embedding distance between its two sides.
#    (b) matters: v1 failed because "these strings differ" was doing the work.
#    Matching how far apart the two sides are removes that explanation.
#
# 3. REPORTS EVERY EFFECT AS A FRACTION OF THE POSITIVE CONTROL.
#    Absolute bits are meaningless on their own. "22% of what swapping the entire
#    medical history does" is a claim a reader can evaluate.
import os, sys, json, subprocess, random, itertools

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

MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "You are a physician taking a patient history. Ask one follow-up question."
N_SCENARIOS = 30     # raise for tighter estimates, lower for speed
CONT_LEN    = 24     # tokens of the reference question that we score

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

# ---- generate a large pool of clinically irrelevant sentences -----------------
FRAMES = [
 ("The patient {v} by {mode} on a {adj} {when}.",
  dict(v=["arrived","travelled","came","returned"], mode=["bus","train","taxi","tram","car"],
       adj=["cloudy","sunny","rainy","windy","quiet","busy","cold","warm"],
       when=["morning","afternoon","evening","weekday","weekend"])),
 ("The patient waited {adv} in the {place} {room} area.",
  dict(adv=["briefly","quietly","patiently","calmly"], place=["upstairs","downstairs","front","back","side"],
       room=["waiting","reception","seating","lobby"])),
 ("The patient booked this visit through the {chan} {when}.",
  dict(chan=["online portal","phone service","clinic website","front desk","mobile application"],
       when=["yesterday","last week","this morning","last month","two days ago"])),
 ("The patient lives on the {dir} side of the {place}.",
  dict(dir=["north","south","east","west","far north","far south"],
       place=["city","town","river","county","district"])),
 ("The patient brought a {adj} folder of {kind} paperwork today.",
  dict(adj=["thin","thick","worn","new","large"], kind=["previous","earlier","routine","older","spare"])),
]
def expand(frame, slots):
    keys = list(slots)
    return [frame.format(**dict(zip(keys, combo))) for combo in itertools.product(*[slots[k] for k in keys])]

tok = AutoTokenizer.from_pretrained(MODEL)
def ntok(s): return len(tok(s, add_special_tokens=False).input_ids)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()
if torch.cuda.is_available(): model = model.cuda()
EMB = model.get_input_embeddings()

@torch.no_grad()
def embed(s):
    ids = torch.tensor([tok(s, add_special_tokens=False).input_ids]).to(model.device)
    v = EMB(ids)[0].mean(0)
    return (v / v.norm()).float().cpu().numpy()

def cosdist(a, b): return 1.0 - float(np.dot(embed(a), embed(b)))

cands = []
for fr, sl in FRAMES:
    sents = expand(fr, sl)
    for a, b in itertools.combinations(sents, 2):
        if a != b: cands.append((a, b))
random.Random(0).shuffle(cands)
cands = cands[:3000]
by_len = defaultdict(list)
for a, b in cands: by_len[(ntok(a), ntok(b))].append((a, b))
print(f"placebo candidates: {len(cands)} across {len(by_len)} length combinations", flush=True)

# ---- match each style pair to a placebo pair on length AND semantic distance --
matched, dropped = defaultdict(list), []
for dim, lst in CUES.items():
    for sh, sl_ in lst:
        key = (ntok(sh), ntok(sl_))
        pool = by_len.get(key, [])
        if not pool:
            dropped.append((dim, sh, key)); continue
        target = cosdist(sh, sl_)
        best = min(pool[:60], key=lambda ab: abs(cosdist(*ab) - target))
        matched[dim].append((sh, sl_, best[0], best[1], target, cosdist(*best)))

print(f"matched phrasings: { {k: len(v) for k, v in matched.items()} }")
if dropped:
    print(f"dropped {len(dropped)}: {[(d[0], d[2]) for d in dropped]}")
gapz = [abs(m[4]-m[5]) for v in matched.values() for m in v]
print(f"semantic-distance mismatch after matching: mean {np.mean(gapz):.4f}, max {np.max(gapz):.4f}\n", flush=True)


def chat(text):
    return tok.apply_chat_template([{"role":"system","content":SYSTEM},
                                    {"role":"user","content":text}],
                                   tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def reference_question(facts):
    """Greedy continuation from a NEUTRAL prompt: facts only, no descriptors."""
    enc = tok(chat(facts), return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=CONT_LEN, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return out[0, enc.input_ids.shape[-1]:]

@torch.no_grad()
def scored(text, cont_ids):
    """Distributions the model assigns at each position of the SAME question."""
    pid = tok(chat(text), return_tensors="pt").input_ids.to(model.device)
    ids = torch.cat([pid, cont_ids.unsqueeze(0)], dim=1)
    logits = model(ids).logits[0, pid.shape[-1]-1 : -1]        # (CONT_LEN, vocab)
    return torch.softmax(logits.float(), dim=-1)

def js_mean(P, Q):
    """Mean Jensen-Shannon divergence (bits) across the sentence."""
    M = 0.5*(P+Q)
    def kl(A, B):
        return (A * (torch.log2(A.clamp_min(1e-12)) - torch.log2(B.clamp_min(1e-12)))).sum(-1)
    return float((0.5*kl(P, M) + 0.5*kl(Q, M)).mean())

# ---- scenarios ---------------------------------------------------------------
facts, seen = [], set()
for path in sorted(glob.glob("DiaLense/results/raw/*/conversations.jsonl")):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["scenario_id"] in seen: continue
        seen.add(r["scenario_id"])
        facts.append("Reported history: " + "; ".join(
            str(v).strip().rstrip(".") for v in r["latent_facts"].values()) + ".")
facts = facts[:N_SCENARIOS]
refs = [reference_question(F) for F in facts]
print(f"scenarios: {len(facts)} | reference question of scenario 1:")
print(f'   "{tok.decode(refs[0]).strip()}"\n', flush=True)

# ---- controls ----------------------------------------------------------------
sh, sl_, pa, pb, _, _ = matched[list(matched)[0]][0]
b = scored(f"{sh} {pa} {facts[0]}", refs[0])
NEG = js_mean(b, scored(f"{sh} {pa} {facts[0]}", refs[0]))
POS = float(np.mean([js_mean(scored(f"{sh} {pa} {F}", r),
                             scored(f"{sh} {pa} {facts[(i+1) % len(facts)]}", r))
                     for i, (F, r) in enumerate(zip(facts[:10], refs[:10]))]))
print("="*78)
print(f"  NEGATIVE control (swap nothing)        : {NEG:.3e}   must be ~0")
print(f"  POSITIVE control (swap whole history)  : {POS:.5f} bits   must be large")
if NEG > 1e-9 or POS < 1e-4:
    raise SystemExit("  metric untrustworthy - stop")
print("  -> metric works\n", flush=True)

# ---- main loop ---------------------------------------------------------------
rows = []
for dim, phrs in matched.items():
    for k, (sh, sl_, pa, pb, _, _) in enumerate(phrs):
        for F, r in zip(facts, refs):
            ds, dp = [], []
            for style_first in (True, False):
                mk = (lambda s, p: f"{s} {p} {F}") if style_first else (lambda s, p: f"{p} {s} {F}")
                base = scored(mk(sh, pa), r)
                ds.append(js_mean(base, scored(mk(sl_, pa), r)))
                dp.append(js_mean(base, scored(mk(sh, pb), r)))
            rows.append((dim, k, float(np.mean(ds)), float(np.mean(dp)), ds[0], ds[1]))
        print(f"   {dim} {k+1}/{len(phrs)}", flush=True)

print("\n" + "="*78)
print("RESULT  (all effects also shown as % of the positive control)")
print("="*78)
summary = {}
for dim in sorted(matched):
    sub = [r for r in rows if r[0] == dim]
    ds = np.array([r[2] for r in sub]); dp = np.array([r[3] for r in sub])
    eff = ds - dp
    phr = sorted(set(r[1] for r in sub))
    per = np.array([np.mean([r[2]-r[3] for r in sub if r[1] == k]) for k in phr])
    pv = stats.wilcoxon(per, alternative="greater").pvalue if len(per) >= 5 else float("nan")
    boot = np.random.default_rng(0).choice(per, (4000, len(per))).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    s1 = np.mean([r[4] for r in sub]); s2 = np.mean([r[5] for r in sub])
    print(f"\n{dim.upper().replace('_',' ')}   ({len(phr)} phrasings x {len(facts)} scenarios)")
    print(f"   D_style                  : {ds.mean():.5f} bits   ({100*ds.mean()/POS:5.1f}% of positive control)")
    print(f"   D_placebo                : {dp.mean():.5f} bits   ({100*dp.mean()/POS:5.1f}%)")
    print(f"   effect (style - placebo) : {eff.mean():+.5f}   95% CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"   position audit slot1/slot2: {s1:.5f} / {s2:.5f}")
    print(f"   per-phrasing effects     : {np.array2string(per, precision=5)}")
    print(f"   CONSERVATIVE p (n={len(per)})     : {pv:.4f}"
          f"   {'-> ESTABLISHED' if (pv < 0.05 and lo > 0) else '-> not established'}")
    summary[dim] = dict(d_style=float(ds.mean()), d_placebo=float(dp.mean()),
                        effect=float(eff.mean()), ci=[float(lo), float(hi)],
                        pct_of_positive_control=float(100*ds.mean()/POS),
                        p_conservative=float(pv))

summary["_controls"] = dict(negative=float(NEG), positive=float(POS))
json.dump(summary, open("baseline_v5.json", "w"), indent=2)
print("\nsaved baseline_v5.json — rerun this identical cell after fine-tuning.")
