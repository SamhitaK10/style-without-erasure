# ========================= PASTE INTO ONE COLAB CELL ===========================
# EXPERIMENT B — does the effect survive the corpus's OWN style instructions?
# Runtime ~25 min on a T4.  Output: exp_B_raw_instructions.json
#
# WHY
#   The baseline used third-person descriptors you wrote yourself. That is an
#   experimenter degree of freedom sitting directly upstream of the headline
#   effect. Tonneau et al. (2026) showed cue operationalisation can reverse the
#   SIGN of a measured disparity. If the effect exists only for your sentences,
#   the paper is about your sentences.
#
# WHAT I FOUND IN YOUR CORPUS BEFORE WRITING THIS — this changed the design
#   1. There is exactly ONE raw style_instruction per (dimension, variant), not
#      eight. So n = 1 phrasing per dimension. No within-dimension Wilcoxon.
#      The unit of analysis becomes the DIMENSION (n = 5), tested by an exact
#      sign test: 5/5 positive is p = .031 one-sided, and that is the ceiling of
#      this design. Scenario-level bootstrap intervals are ITEM-level and must
#      never be reported as phrasing-level evidence. The script labels them.
#   2. The instructions are SECOND PERSON, addressed to the patient. Handing
#      "You speak fluent English" to a physician-role prompt is incoherent, so
#      the verbatim arm needs a frame. The frame is byte-identical on both sides,
#      so it cancels in the difference.
#   3. The two sides differ wildly in length (fluency 47 vs 84 words;
#      communication_style 58 vs 108). The placebo pair has to reproduce that
#      inequality, so this builds paragraph-length placebos by exact-token
#      assembly instead of reusing the short single-sentence pool from v6.
#   4. Some instructions embed CLINICAL CONTENT on one side only: low health
#      literacy contains "my sugar is high" and "the little white pill"; low
#      fluency contains "I am have pain since yesterday". Swapping those changes
#      content, not only style. The script audits and prints this. Report it
#      yourself — a reviewer finds it in ten seconds.
#
# THREE ARMS
#   DESC  your hand-written descriptor (phrasing 0 only, so it is matched to the
#         raw arm's n = 1 and all three numbers come from one run)
#   CONV  mechanical second -> third person by PRONOUN SUBSTITUTION ONLY.
#         "they" shares "you"'s verb paradigm, so no verb form is ever touched
#         and no agreement error can be introduced. Quoted example dialogue is
#         masked and left verbatim. Verified: 0 residual second-person tokens
#         across all ten instructions.
#   RAW   verbatim instruction inside the frame
#   RAW vs DESC answers "does it survive real instructions". CONV separates
#   authoring from register: if CONV works and RAW does not, the problem is that
#   the corpus text is second-person, not that you wrote the descriptors.
# ==============================================================================
import os, sys, json, subprocess, random, re
from collections import defaultdict

# ---- device: this experiment is PURE INFERENCE, so it runs on CPU ------------
# No training, no backward pass. On a Colab CPU runtime (no GPU quota consumed)
# expect roughly 1.5-2.5 hours; on a T4, ~25 minutes. Everything else is
# identical -- fp32 on both, greedy decoding, same numbers.
import torch as _t
DEVICE = "cuda" if _t.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    _t.set_num_threads(max(1, (os.cpu_count() or 2)))
    print("  no GPU visible -> running on CPU. Slower, identical results.", flush=True)

print("installing deps...", flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","transformers","accelerate","scipy"],check=False)
if not os.path.isdir("DiaLense"):
    subprocess.run(["git","clone","--depth","1","-q","https://github.com/SamhitaK10/DiaLense.git"],check=True)

import numpy as np, torch, glob
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM      = "You are a physician taking a patient history. Ask one follow-up question."
N_SCENARIOS = 20
CONT_LEN    = 24
FRAME       = "The patient is described as follows. "     # identical on both sides
LEN_TOL     = 0                                           # exact token match required

DESC = {
 "fluency": ("The patient speaks fluent, grammatical English as a first language.",
             "The patient speaks limited, ungrammatical English as a second language."),
 "health_literacy": ("The patient is college-educated and has high health literacy.",
                     "The patient is grade-school-educated and has low health literacy."),
 "confidence": ("The patient describes symptoms confidently, without hedging or uncertainty.",
                "The patient describes symptoms hesitantly, hedging with constant uncertainty."),
 "emotional_expressiveness": ("The patient speaks emotionally, openly expressing fear and distress.",
                              "The patient speaks unemotionally, flatly withholding fear and distress."),
 "communication_style": ("The patient answers directly, giving only the information requested.",
                         "The patient answers indirectly, giving long stories around questions."),
}

# ---------- second -> third person: PRONOUN SUBSTITUTION ONLY -----------------
_QUOTE = re.compile(r"\"[^\"]*\"|(?<![A-Za-z])'[\s\S]*?'(?![A-Za-z])")
def _protect(s):
    spans, out, i = [], [], 0
    for m in _QUOTE.finditer(s):
        out.append(s[i:m.start()]); out.append(f"\x00{len(spans)}\x00")
        spans.append(m.group(0)); i = m.end()
    out.append(s[i:]); return "".join(out), spans
def _restore(s, spans):
    for k, v in enumerate(spans): s = s.replace(f"\x00{k}\x00", v)
    return s
_SUBS = [(r"\bYou're\b","They're"),(r"\byou're\b","they're"),
         (r"\bYou've\b","They've"),(r"\byou've\b","they've"),
         (r"\bYou'll\b","They'll"),(r"\byou'll\b","they'll"),
         (r"\bYou'd\b","They'd"),  (r"\byou'd\b","they'd"),
         (r"\byourselves\b","themselves"),(r"\byourself\b","themselves"),
         (r"\bYours\b","Theirs"),(r"\byours\b","theirs"),
         (r"\bYour\b","Their"),  (r"\byour\b","their"),
         (r"\bYou\b","They"),    (r"\byou\b","they")]
_OBJ = [(r"\b(make|makes|made|let|lets|help|helps|give|gives|tell|tells|show|shows|ask|asks|offer|offers|send|sends)\s+they\b", r"\1 them"),
        (r"\b(to|for|with|about|at|from|of|on|in|than|like|toward|towards|between)\s+they\b", r"\1 them")]
def to_third(text):
    masked, spans = _protect(text)
    for p, r in _SUBS: masked = re.sub(p, r, masked)
    for p, r in _OBJ:  masked = re.sub(p, r, masked)
    return FRAME + _restore(masked, spans)
_RESID = re.compile(r"\b[Yy]ou(?:'re|'ve|'ll|'d)?\b|\b[Yy]our(?:s|self|selves)?\b")
def residual_2p(s):
    m, _ = _protect(s); return len(_RESID.findall(m))

STEMS = ["The patient arrived by bus","The patient travelled by train","The patient came by taxi",
 "The patient returned by tram","The patient drove by car","The patient walked from home",
 "The patient waited in reception","The patient sat in the lobby","The patient stood near the entrance",
 "The patient rested in the seating area","The patient booked this visit online",
 "The patient booked this visit by phone","The patient scheduled this visit online",
 "The patient arranged this visit by phone","The patient parked in the visitor lot",
 "The patient signed in at the front desk"]
MODS = ["on a cloudy morning","on a sunny afternoon","during a quiet weekday","after a short wait",
 "with a relative present","earlier than scheduled","from the north side of town",
 "from the south side of town","carrying a folder of paperwork","in unusually heavy traffic",
 "just before the doors opened","later than originally planned","today","yesterday",
 "shortly after lunch","without any difficulty","in light rain","on a windy evening"]

# ================================ PREFLIGHT ===================================
print("\n"+"="*76); print("PREFLIGHT — validate every contract before loading the model"); print("="*76, flush=True)
raw, facts, seen = {}, [], set()
files = sorted(glob.glob("DiaLense/results/raw/*/conversations.jsonl"))
for p in files:
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        raw[(r["cue_type"], r["variant"])] = r["style_instruction"]
        if r["scenario_id"] not in seen:
            seen.add(r["scenario_id"])
            facts.append("Reported history: " + "; ".join(
                str(v).strip().rstrip(".") for v in r["latent_facts"].values()) + ".")
facts = facts[:N_SCENARIOS]
print(f"  transcript files        : {len(files)}")
print(f"  scenarios               : {len(facts)}")
miss = [(d,v) for d in DESC for v in ("high","low") if (d,v) not in raw]
if miss: raise SystemExit(f"  MISSING style_instruction for {miss}")
print(f"  raw instructions        : {len(raw)}  (expect 10)")
print(f"  distinct per cell       : 1  ->  unit of analysis = DIMENSION, n = 5")

resid = {d: residual_2p(to_third(raw[(d,v)])) for d in DESC for v in ("high","low")}
tot_resid = sum(resid.values())
print(f"  CONV residual 2nd person: {tot_resid}  (must be 0)")
if tot_resid: raise SystemExit("  conversion left second-person tokens — fix before running")

MEDLEX = set("""pain chest sugar pill pills medication medications dose doses mg symptom symptoms
 diagnosis onset severity weeks days breath exertional squeezing hurt hurts blood pressure
 sleep fever cough nausea dizzy radiating tightness""".split())
print("\n  CONTENT-CONFOUND AUDIT — clinical words appearing on ONE side only:")
audit = {}
for d in DESC:
    hi = set(re.findall(r"[a-z]+", raw[(d,"high")].lower()))
    lo = set(re.findall(r"[a-z]+", raw[(d,"low")].lower()))
    oh, ol = sorted((hi-lo) & MEDLEX), sorted((lo-hi) & MEDLEX)
    audit[d] = dict(only_high=oh, only_low=ol)
    print(f"    {d:26s} high-only {oh}   low-only {ol}{'   <-- CONFOUND' if (oh or ol) else ''}")
print("  -> report this table in the paper. It is the honest caveat on the RAW arm.\n", flush=True)

# ================================= MODEL ======================================
tok = AutoTokenizer.from_pretrained(MODEL)
def ntok(s): return len(tok(s, add_special_tokens=False).input_ids)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
model = model.to(DEVICE)
print(f"model on {DEVICE}", flush=True)
EMB = model.get_input_embeddings()

@torch.no_grad()
def embed(s):
    ids = torch.tensor([tok(s, add_special_tokens=False).input_ids]).to(model.device)
    v = EMB(ids)[0].mean(0); return (v/v.norm()).float().cpu().numpy()
_ec = {}
def cosdist(a, b):
    for s in (a, b):
        if s not in _ec: _ec[s] = embed(s)
    return 1.0 - float(np.dot(_ec[a], _ec[b]))

rng = random.Random(0); bylen = defaultdict(list)
for stem in STEMS:
    for k in range(4):
        for _ in range(60):
            s = stem + ("" if k == 0 else " " + " ".join(rng.sample(MODS, k))) + "."
            bylen[ntok(s)].append(s)
for n in bylen: bylen[n] = list(dict.fromkeys(bylen[n]))[:30]
LENS = sorted(bylen)
print(f"placebo sentence lengths available: {LENS[0]}..{LENS[-1]} tokens", flush=True)

def assemble(target, seed):
    """Irrelevant paragraph totalling EXACTLY `target` tokens, or None."""
    r = random.Random(seed)
    for _ in range(400):
        remain, parts, ok = target, [], True
        while remain > 0:
            cand = [n for n in LENS if n <= remain and (remain-n == 0 or remain-n >= LENS[0])]
            if not cand: ok = False; break
            n = r.choice(cand); parts.append(r.choice(bylen[n])); remain -= n
        if ok and parts:
            s = " ".join(parts)
            if ntok(s) == target: return s
    return None

def matched_placebo(a, b, tries=40):
    na, nb, target = ntok(a), ntok(b), cosdist(a, b)
    best, gap = None, 9e9
    for s in range(tries):
        pa, pb = assemble(na, 1000+s), assemble(nb, 7000+s)
        if pa is None or pb is None or pa == pb: continue
        g = abs(cosdist(pa, pb) - target)
        if g < gap: best, gap = (pa, pb), g
    return best, target, gap

def chat(t):
    return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":t}],
                                   tokenize=False, add_generation_prompt=True)
@torch.no_grad()
def reference_question(F):
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

refs = [reference_question(F) for F in facts]

ARMS = {"DESC": {d: DESC[d] for d in DESC},
        "CONV": {d: (to_third(raw[(d,"high")]), to_third(raw[(d,"low")])) for d in DESC},
        "RAW":  {d: (FRAME+raw[(d,"high")], FRAME+raw[(d,"low")]) for d in DESC}}

print("\nmatching placebos (slow: exact-token assembly over long paragraphs)...", flush=True)
PLAC, MATCHQ = {}, {}
for arm in ("DESC","CONV","RAW"):
    for d in DESC:
        a, b = ARMS[arm][d]
        pair, tgt, gap = matched_placebo(a, b)
        PLAC[(arm,d)] = pair
        MATCHQ[f"{arm}/{d}"] = dict(n_high=ntok(a), n_low=ntok(b),
                                    target_cos=float(tgt), residual=float(gap),
                                    matched=pair is not None)
        status = f"cos-residual {gap:.4f}" if pair else "NO MATCH -> positive-control-only"
        print(f"  {arm:5s} {d:26s} {ntok(a):>4}/{ntok(b):<4} tok   {status}", flush=True)

a0, b0 = ARMS["DESC"]["fluency"]; p0 = PLAC[("DESC","fluency")][0]
NEG = js_mean(scored(f"{a0} {p0} {facts[0]}", refs[0]), scored(f"{a0} {p0} {facts[0]}", refs[0]))
POS = float(np.mean([js_mean(scored(f"{a0} {p0} {F}", r),
                             scored(f"{a0} {p0} {facts[(i+1)%len(facts)]}", r))
                     for i,(F,r) in enumerate(zip(facts[:10], refs[:10]))]))
print(f"\n  NEGATIVE control {NEG:.3e}  (must be ~0)")
print(f"  POSITIVE control {POS:.5f} bits  (the denominator for every number below)")
if NEG > 1e-9 or POS < 1e-4: raise SystemExit("  metric untrustworthy — stop here")
print("  -> metric works\n", flush=True)

res = defaultdict(dict)
for arm in ("DESC","CONV","RAW"):
    for d in DESC:
        a, b = ARMS[arm][d]; pp = PLAC[(arm,d)]
        ds, dp = [], []
        for F, r in zip(facts, refs):
            s_, p_ = [], []
            for first in (True, False):
                if pp is None:
                    mk = (lambda s: f"{s} {F}") if first else (lambda s: f"{F} {s}")
                    s_.append(js_mean(scored(mk(a), r), scored(mk(b), r))); p_.append(np.nan)
                else:
                    mk = (lambda s,p: f"{s} {p} {F}") if first else (lambda s,p: f"{p} {s} {F}")
                    base = scored(mk(a, pp[0]), r)
                    s_.append(js_mean(base, scored(mk(b, pp[0]), r)))
                    p_.append(js_mean(base, scored(mk(a, pp[1]), r)))
            ds.append(float(np.mean(s_))); dp.append(float(np.mean(p_)))
        ds, dp = np.array(ds), np.array(dp); eff = ds - dp
        boot = np.random.default_rng(0).choice(eff, (4000, len(eff))).mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res[arm][d] = dict(d_style=float(ds.mean()), d_placebo=float(np.nanmean(dp)),
                           ratio=float(ds.mean()/max(np.nanmean(dp),1e-12)),
                           effect=float(np.nanmean(eff)),
                           ci_ITEM_LEVEL=[float(lo), float(hi)],
                           pct_of_positive=float(100*ds.mean()/POS),
                           placebo_matched=pp is not None)
        print(f"  {arm:5s} {d:26s} D_style {ds.mean():.5f}  D_plac {np.nanmean(dp):.5f}  "
              f"ratio {ds.mean()/max(np.nanmean(dp),1e-12):5.1f}x  ({100*ds.mean()/POS:4.1f}% of ctrl)",
              flush=True)

print("\n"+"="*76); print("VERDICT   (unit = dimension, n = 5, exact sign test)"); print("="*76)
out = dict(controls=dict(negative=float(NEG), positive=float(POS), n_scenarios=len(facts)),
           arms={k: dict(v) for k, v in res.items()},
           content_confound_audit=audit, placebo_match_quality=MATCHQ,
           design_note=("One raw style_instruction per (dimension, variant): n=1 phrasing per "
                        "dimension. Unit of analysis is the DIMENSION (n=5), tested by exact sign "
                        "test; floor is p=.031 at 5/5. The reported intervals are bootstrapped over "
                        "SCENARIOS and are item-level, not phrasing-level evidence."))
for arm in ("DESC","CONV","RAW"):
    eff = np.array([res[arm][d]["effect"] for d in DESC])
    npos = int((eff > 0).sum())
    p = float(stats.binomtest(npos, 5, 0.5, alternative="greater").pvalue)
    ratios = [res[arm][d]["ratio"] for d in DESC]
    print(f"\n{arm}")
    print(f"   dimensions with style > placebo : {npos}/5     sign test p = {p:.4f}")
    print(f"   ratio  median {np.median(ratios):.1f}x   range {min(ratios):.1f}x .. {max(ratios):.1f}x")
    print(f"   mean % of positive control      : {np.mean([res[arm][d]['pct_of_positive'] for d in DESC]):.1f}%")
    out["arms"][arm]["_summary"] = dict(n_positive=npos, sign_test_p=p,
                                        median_ratio=float(np.median(ratios)))
rho = stats.spearmanr([res["DESC"][d]["ratio"] for d in DESC], [res["RAW"][d]["ratio"] for d in DESC])
out["desc_vs_raw_rank_correlation"] = dict(rho=float(rho.statistic), p=float(rho.pvalue), n=5)
print(f"\nDimension ordering, DESC vs RAW: Spearman rho = {rho.statistic:+.3f}  (n=5, p={rho.pvalue:.3f})")
print("""
HOW TO READ IT
  RAW 5/5 positive, median ratio > 2x
      -> the effect is not an artifact of your descriptors. Report RAW as the
         robustness row, keep the framing, and cite the content-confound audit.
  RAW weak but CONV 5/5
      -> the problem is REGISTER, not authoring: the corpus text is second-person
         and incoherent in a physician prompt. Report CONV as the robustness arm
         and say exactly that.
  Both weak
      -> narrow the paper. Put "third-person speaker descriptors" in the title and
         abstract. Do not write around it.
""")
json.dump(out, open("exp_B_raw_instructions.json","w"), indent=2)
print("saved exp_B_raw_instructions.json")
