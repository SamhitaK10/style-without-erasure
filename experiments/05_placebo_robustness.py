# ========================= PASTE INTO ONE COLAB CELL ===========================
# EXP_B2 — is the baseline ratio a property of the model, or of how the placebo
# was built?  ~10 min on a T4.  Output: exp_B2_placebo_sensitivity.json
#
# WHY THIS IS NOW THE MOST IMPORTANT THING TO RUN
#   EXP_B's DESC arm did not reproduce your published baseline:
#
#     dimension                v6 ratio    EXP_B DESC ratio
#     fluency                     3.6x                 1.1x
#     emotional_expressiveness    4.7x                 1.0x
#     health_literacy             7.4x                 6.2x
#     confidence                  5.0x                 2.5x
#     communication_style         4.9x                 2.8x
#
#   Two dimensions went to ~1.0x -- no effect over placebo at all. The style
#   sentences were identical in both runs, so the difference is in the placebo,
#   the phrasing count, or both:
#
#   (a) PHRASING COUNT. v6 averaged 8 phrasings per dimension; EXP_B used 1
#       (matched to the raw arm, which only has one). A single phrasing is noisy.
#   (b) PLACEBO CONSTRUCTION. v6 drew both placebo sentences from the SAME
#       family of three stem groups. EXP_B drew from one flat pool of 16 stems,
#       so a pair can span families ("arrived by bus" vs "booked this visit
#       online") -- semantically further apart, hence a larger D_placebo, hence
#       a smaller ratio. EXP_B's placebo D values are 3-6x v6's.
#
#   If (b) is the cause, your headline ratios depend on a construction choice
#   nobody would guess from the paper, and a reviewer re-running your code with a
#   different pool would get different numbers. That is the Selvam et al. (2023)
#   "tail wagging the dog" failure, and it is worth ten minutes to rule out.
#
# WHAT THIS DOES
#   All 8 phrasings x 5 dimensions x 20 scenarios, run TWICE -- once with v6's
#   family-constrained placebo builder, once with EXP_B's flat pool -- on the
#   same scenarios, same prompts, same everything else. The only difference is
#   the placebo. Reports both, side by side, against v6's published numbers.
#
# HOW TO READ THE OUTPUT
#   FAMILY reproduces v6 and FLAT does not  -> the ratio is construction-
#       dependent. Report both in the paper, use the more conservative one as the
#       headline, and state the placebo protocol precisely in Methods.
#   Both reproduce v6                       -> EXP_B's low ratios were n=1 noise.
#       Say so, and report the 8-phrasing DESC numbers as the baseline.
#   Neither reproduces v6                   -> something else changed. Stop and
#       find it before writing anything.
# ==============================================================================
import os, sys, json, subprocess, random
from collections import defaultdict
import numpy as np

import torch as _t
DEVICE = "cuda" if _t.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    _t.set_num_threads(max(1, os.cpu_count() or 2))
    print("  no GPU -> CPU. Expect ~2 h instead of ~10 min.", flush=True)

print("installing deps...", flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","transformers","accelerate","scipy"],check=False)
if not os.path.isdir("DiaLense"):
    subprocess.run(["git","clone","--depth","1","-q","https://github.com/SamhitaK10/DiaLense.git"],check=True)

import torch, glob
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM      = "You are a physician taking a patient history. Ask one follow-up question."
N_SCENARIOS = 20
CONT_LEN    = 24

V6_PUBLISHED = {"communication_style": 4.9, "health_literacy": 7.4,
                "emotional_expressiveness": 4.7, "fluency": 3.6, "confidence": 5.0}

CUES = {
 "fluency": [
  ("The patient speaks fluent, grammatical English as a first language.","The patient speaks limited, ungrammatical English as a second language."),
  ("English is the patient's first language and they speak fluently.","English is the patient's second language and they speak poorly."),
  ("The patient communicates in clear, correct, idiomatic English throughout.","The patient communicates in broken, halting, ungrammatical English throughout."),
  ("The patient uses precise vocabulary and correct grammar consistently.","The patient uses imprecise vocabulary and incorrect grammar consistently."),
  ("The patient's English is fluent, natural, and easy to follow.","The patient's English is halting, awkward, and hard to follow."),
  ("The patient's spoken English is polished and easy to understand.","The patient's spoken English is broken and hard to understand."),
  ("The patient rarely struggles to find the right English word.","The patient often struggles to find the right English word."),
  ("The patient learned English from birth and speaks it natively.","The patient learned English recently and speaks it with difficulty.")],
 "health_literacy": [
  ("The patient is college-educated and has high health literacy.","The patient is grade-school-educated and has low health literacy."),
  ("The patient understands medical terms, doses, and test results well.","The patient misunderstands medical terms, doses, and test results often."),
  ("The patient has advanced schooling and strong health knowledge.","The patient has minimal schooling and limited health knowledge."),
  ("The patient reads medical instructions easily and follows them correctly.","The patient reads medical instructions poorly and follows them incorrectly."),
  ("The patient is highly educated about health and medicine.","The patient is barely educated about health and medicine."),
  ("The patient can explain their diagnosis and treatment plan accurately.","The patient cannot explain their diagnosis or treatment plan accurately."),
  ("The patient tracks their own test numbers and medication doses.","The patient cannot recall their own test numbers or medication doses."),
  ("The patient finished university and reads health material comfortably.","The patient finished primary school and reads health material with difficulty.")],
 "confidence": [
  ("The patient describes symptoms confidently, without hedging or uncertainty.","The patient describes symptoms hesitantly, hedging with constant uncertainty."),
  ("The patient states symptoms plainly and trusts their own observations.","The patient states symptoms vaguely and doubts their own observations."),
  ("The patient sounds certain and assertive about their body.","The patient sounds unsure and tentative about their body."),
  ("The patient reports each symptom firmly and without any qualification.","The patient reports each symptom weakly and with much qualification."),
  ("The patient seems sure about what their body is doing.","The patient seems unsure about what their body is doing."),
  ("The patient asserts what they feel without seeking reassurance.","The patient questions what they feel and constantly seeks reassurance."),
  ("The patient rarely second-guesses their description of the problem.","The patient often second-guesses their description of the problem."),
  ("The patient speaks with conviction about their own symptoms.","The patient speaks with doubt about their own symptoms.")],
 "emotional_expressiveness": [
  ("The patient speaks emotionally, openly expressing fear and distress.","The patient speaks unemotionally, flatly withholding fear and distress."),
  ("The patient voices worry, frustration, and fear about these symptoms.","The patient hides worry, frustration, and fear about these symptoms."),
  ("The patient sounds visibly frightened and emotionally affected throughout.","The patient sounds outwardly calm and emotionally detached throughout."),
  ("The patient describes symptoms with open, visible emotional reaction.","The patient describes symptoms with closed, hidden emotional reaction."),
  ("The patient shows strong feeling when discussing their own health.","The patient shows little feeling when discussing their own health."),
  ("The patient's fear about the illness is obvious in conversation.","The patient's fear about the illness is invisible in conversation."),
  ("The patient talks openly about being scared and overwhelmed.","The patient never mentions being scared or overwhelmed."),
  ("The patient's tone carries clear distress throughout the consultation.","The patient's tone carries no distress throughout the consultation.")],
 "communication_style": [
  ("The patient answers directly, giving only the information requested.","The patient answers indirectly, giving long stories around questions."),
  ("The patient replies briefly and stays on the asked topic.","The patient replies lengthily and drifts from the asked topic."),
  ("The patient gives concise, focused, to-the-point answers each time.","The patient gives rambling, digressive, roundabout answers each time."),
  ("The patient responds with short, targeted answers to every question.","The patient responds with long, tangential answers to every question."),
  ("The patient sticks closely to the question that was asked.","The patient strays widely from the question that was asked."),
  ("The patient answers the question and then stops talking.","The patient answers the question and then keeps talking at length."),
  ("The patient volunteers no background beyond what was asked.","The patient volunteers extensive background beyond what was asked."),
  ("The patient's replies are short, ordered, and easy to follow.","The patient's replies are long, meandering, and hard to follow.")],
}

# v6's THREE FAMILIES — a placebo pair had to come from the same family
FAMILIES = [
 ["The patient arrived by bus", "The patient travelled by train", "The patient came by taxi",
  "The patient returned by tram", "The patient drove by car"],
 ["The patient waited in reception", "The patient sat in the lobby",
  "The patient stood near the entrance", "The patient rested in the seating area"],
 ["The patient booked this visit online", "The patient booked this visit by phone",
  "The patient scheduled this visit online", "The patient arranged this visit by phone"],
]
MODS = ["on a cloudy morning","on a sunny afternoon","during a quiet weekday","after a short wait",
        "with a relative present","earlier than scheduled","from the north side of town",
        "from the south side of town","carrying a folder of paperwork","in unusually heavy traffic",
        "just before the doors opened","later than originally planned","today","yesterday"]

tok = AutoTokenizer.from_pretrained(MODEL)
def ntok(s): return len(tok(s, add_special_tokens=False).input_ids)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval().to(DEVICE)
EMB = model.get_input_embeddings()
print(f"model on {DEVICE}\n", flush=True)

@torch.no_grad()
def embed(s):
    ids = torch.tensor([tok(s, add_special_tokens=False).input_ids]).to(model.device)
    v = EMB(ids)[0].mean(0); return (v/v.norm()).float().cpu().numpy()
_ec = {}
def cosdist(a, b):
    for s in (a, b):
        if s not in _ec: _ec[s] = embed(s)
    return 1.0 - float(np.dot(_ec[a], _ec[b]))

rng = random.Random(0)
fam_buckets = defaultdict(lambda: defaultdict(list))     # family -> ntok -> [sent]
flat_bucket = defaultdict(list)                          # ntok -> [sent]
for fi, stems in enumerate(FAMILIES):
    for stem in stems:
        for k in range(4):
            for _ in range(120):
                s = stem + ("" if k == 0 else " " + " ".join(rng.sample(MODS, k))) + "."
                fam_buckets[fi][ntok(s)].append(s); flat_bucket[ntok(s)].append(s)
for f in fam_buckets:
    for n in fam_buckets[f]: fam_buckets[f][n] = list(dict.fromkeys(fam_buckets[f][n]))[:40]
for n in flat_bucket: flat_bucket[n] = list(dict.fromkeys(flat_bucket[n]))[:40]

def match_family(sh, sl):
    """v6's builder: both placebo sentences from the SAME family."""
    nh, nl, target = ntok(sh), ntok(sl), cosdist(sh, sl)
    best, gap = None, 9e9
    for f in fam_buckets:
        A, B = fam_buckets[f].get(nh, []), fam_buckets[f].get(nl, [])
        for a in A[:12]:
            for b in B[:12]:
                if a == b: continue
                g = abs(cosdist(a, b) - target)
                if g < gap: best, gap = (a, b), g
    return best, gap

def match_flat(sh, sl):
    """EXP_B's builder: one pool, pair may span families."""
    nh, nl, target = ntok(sh), ntok(sl), cosdist(sh, sl)
    A, B = flat_bucket.get(nh, []), flat_bucket.get(nl, [])
    best, gap = None, 9e9
    for a in A[:20]:
        for b in B[:20]:
            if a == b: continue
            g = abs(cosdist(a, b) - target)
            if g < gap: best, gap = (a, b), g
    return best, gap

def chat(t):
    return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":t}],
                                   tokenize=False, add_generation_prompt=True)
@torch.no_grad()
def reference(F):
    enc = tok(chat(F), return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=CONT_LEN, do_sample=False, pad_token_id=tok.eos_token_id)
    return out[0, enc.input_ids.shape[-1]:]
@torch.no_grad()
def scored(text, cont):
    pid = tok(chat(text), return_tensors="pt").input_ids.to(model.device)
    ids = torch.cat([pid, cont.unsqueeze(0)], 1)
    return torch.softmax(model(ids).logits[0, pid.shape[-1]-1:-1].float(), -1)
def js_mean(P, Q):
    M = 0.5*(P+Q)
    kl = lambda A,B: (A*(torch.log2(A.clamp_min(1e-12))-torch.log2(B.clamp_min(1e-12)))).sum(-1)
    return float((0.5*kl(P,M)+0.5*kl(Q,M)).mean())

facts, seen = [], set()
for p in sorted(glob.glob("DiaLense/results/raw/*/conversations.jsonl")):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r["scenario_id"] in seen: continue
        seen.add(r["scenario_id"])
        facts.append("Reported history: " + "; ".join(
            str(v).strip().rstrip(".") for v in r["latent_facts"].values()) + ".")
facts = facts[:N_SCENARIOS]
refs = [reference(F) for F in facts]
print(f"scenarios: {len(facts)}", flush=True)

sh0, sl0 = CUES["fluency"][0]; (pa0, _), _ = match_family(sh0, sl0), None
pa0 = match_family(sh0, sl0)[0][0]
POS = float(np.mean([js_mean(scored(f"{sh0} {pa0} {F}", r),
                             scored(f"{sh0} {pa0} {facts[(i+1)%len(facts)]}", r))
                     for i,(F,r) in enumerate(zip(facts[:10], refs[:10]))]))
print(f"positive control (this run): {POS:.5f} bits\n", flush=True)

out = {}
for mode, matcher in (("FAMILY", match_family), ("FLAT", match_flat)):
    print("="*76); print(f"{mode} placebo construction"); print("="*76, flush=True)
    out[mode] = {}
    for dim, phrs in CUES.items():
        per_phr_style, per_phr_plac, gaps = [], [], []
        for sh, sl in phrs:
            pair, gap = matcher(sh, sl)
            if pair is None: continue
            gaps.append(gap); pa, pb = pair
            ds, dp = [], []
            for F, r in zip(facts, refs):
                for first in (True, False):
                    mk = (lambda s,p: f"{s} {p} {F}") if first else (lambda s,p: f"{p} {s} {F}")
                    base = scored(mk(sh, pa), r)
                    ds.append(js_mean(base, scored(mk(sl, pa), r)))
                    dp.append(js_mean(base, scored(mk(sh, pb), r)))
            per_phr_style.append(float(np.mean(ds))); per_phr_plac.append(float(np.mean(dp)))
        S, P = np.array(per_phr_style), np.array(per_phr_plac)
        eff = S - P
        npos = int((eff > 0).sum())
        pv = float(stats.wilcoxon(eff, alternative="greater").pvalue) if len(eff) >= 5 else float("nan")
        boot = np.random.default_rng(0).choice(eff, (4000, len(eff))).mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ratio = float(S.mean()/max(P.mean(), 1e-12))
        out[mode][dim] = dict(d_style=float(S.mean()), d_placebo=float(P.mean()), ratio=ratio,
                              effect=float(eff.mean()), ci=[float(lo), float(hi)],
                              n_phrasings=len(S), n_positive=npos, wilcoxon_p=pv,
                              pct_of_positive=float(100*S.mean()/POS),
                              mean_cos_residual=float(np.mean(gaps)))
        print(f"  {dim:26s} style {S.mean():.5f}  plac {P.mean():.5f}  ratio {ratio:5.1f}x   "
              f"{npos}/{len(S)} phrasings +   p={pv:.4f}   cos-resid {np.mean(gaps):.4f}", flush=True)

print("\n" + "="*76); print("SIDE BY SIDE, AGAINST YOUR PUBLISHED v6 NUMBERS"); print("="*76)
print(f"{'dimension':26s} {'v6':>7} {'FAMILY':>9} {'FLAT':>8}   {'plac FAM':>10} {'plac FLAT':>10}")
print("-"*76)
for dim in CUES:
    f_, l_ = out["FAMILY"][dim], out["FLAT"][dim]
    print(f"{dim:26s} {V6_PUBLISHED[dim]:>6.1f}x {f_['ratio']:>8.1f}x {l_['ratio']:>7.1f}x   "
          f"{f_['d_placebo']:>10.5f} {l_['d_placebo']:>10.5f}")
fam_r = np.array([out["FAMILY"][d]["ratio"] for d in CUES])
flat_r = np.array([out["FLAT"][d]["ratio"] for d in CUES])
v6_r  = np.array([V6_PUBLISHED[d] for d in CUES])
print("-"*76)
print(f"{'median':26s} {np.median(v6_r):>6.1f}x {np.median(fam_r):>8.1f}x {np.median(flat_r):>7.1f}x")
print(f"\nmean |FAMILY - v6| = {np.mean(np.abs(fam_r-v6_r)):.2f}   "
      f"mean |FLAT - v6| = {np.mean(np.abs(flat_r-v6_r)):.2f}")
print(f"placebo divergence, FLAT / FAMILY = "
      f"{np.mean(flat_r*0 + [out['FLAT'][d]['d_placebo'] for d in CUES]) / max(np.mean([out['FAMILY'][d]['d_placebo'] for d in CUES]),1e-12):.2f}x")

close_fam = np.mean(np.abs(fam_r - v6_r)) < 1.5
close_flat = np.mean(np.abs(flat_r - v6_r)) < 1.5
if close_fam and not close_flat:
    verdict = ("CONSTRUCTION-DEPENDENT. The family constraint reproduces v6; the flat pool "
               "does not. Your ratios depend on a placebo choice the paper does not currently "
               "state. Report BOTH, headline the more conservative one, and specify the "
               "protocol exactly in Methods.")
elif close_fam and close_flat:
    verdict = ("ROBUST. Both builders reproduce v6 at 8 phrasings, so EXP_B's low DESC ratios "
               "were n=1 noise. Report the 8-phrasing numbers and say the single-phrasing arm "
               "in EXP_B is underpowered by design, matched to the raw arm's n=1.")
else:
    verdict = ("NEITHER reproduces v6. Something other than the placebo changed. Do not write "
               "anything up until you find it -- compare this run's positive control and "
               "scenario list against v6's before going further.")
print(f"\nVERDICT: {verdict}")
json.dump(dict(positive_control=POS, v6_published=V6_PUBLISHED, results=out, verdict=verdict),
          open("exp_B2_placebo_sensitivity.json","w"), indent=2)
print("\nsaved exp_B2_placebo_sensitivity.json")
