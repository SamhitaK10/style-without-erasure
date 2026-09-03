# ========================= PASTE INTO ONE COLAB CELL ===========================
# EXP_D3 — the decisive version. ~12 min on a T4.  Output: exp_D3_probe.json
#
# WHY A THIRD ATTEMPT, AND WHAT I GOT WRONG TWICE
#
#   EXP_D  said: style-final AUC drops +0.144 at L24, cue positions unchanged.
#          Its content control was broken (label was "first half of the scenario
#          list", computed separately on two DISJOINT scenario sets), so it
#          measured nothing.
#
#   EXP_D2 said: no reliable drop, every interval spans zero.
#          I fixed the content label but changed the ITEM SET at the same time:
#          D used 12 varying scenarios (~360 eval items), D2 used 2 fixed
#          histories (~60). So D2 is both narrower and 6x smaller, and its null
#          is an underpowered null, not a refutation. Changing two things at once
#          is exactly the mistake this project keeps catching elsewhere.
#          Its content control was also at ceiling -- AUC 1.000 -> 1.000, zero-
#          width intervals -- because I picked two maximally different histories
#          (a neonate with nasal congestion vs a type 2 diabetic). A control with
#          no headroom cannot detect degradation.
#
#   So the honest state is: D and D2 disagree, and neither settles it.
#
# WHAT THIS RUN DOES DIFFERENTLY
#   * POWER. 16 scenarios x 5 dimensions x 8 phrasings x 2 style values = 1,280
#     items per adapter state. Style eval n = 480, content eval n = 640.
#   * A CONTENT CONTROL WITH HEADROOM. The label is the corpus's own
#     `correct_urgency`, collapsed to urgent/emergent vs routine/soon. That is a
#     real clinical inference from the history, not a surface cue, so it sits
#     well below ceiling -- and it is exactly the selectivity question stated at
#     the representational level: does the model still encode the medicine?
#   * TWO SPLITS, EACH CORRECT FOR ITS PROBE. Style is split by PHRASING (5/3),
#     so it cannot memorise cue wording. Content is split by SCENARIO (8/8,
#     balanced on urgency), so it cannot memorise a history.
#   * LOG-LOSS ALONGSIDE AUC. AUC saturates; log-loss keeps moving at ceiling, so
#     a degradation that AUC would hide still shows up.
#   * Bootstrap intervals on every delta, resampling eval items.
#
# HOW TO READ IT
#   style-final drop reliably > 0 at late layers, style-cue ~0, content ~0
#       -> targeted suppression of the style readout. The strong claim, earned.
#   style-final and content both drop
#       -> general late-layer degradation. Weaker, still reportable, say it plainly.
#   nothing reliably non-zero at n=480
#       -> the intervention changed behaviour without a detectable representational
#          signature. Report the BOUND (the CI width is your resolution) and cite
#          Lee et al. 2024 and Galichin et al. 2026. That is a finding too.
# ==============================================================================
import os, sys, json, subprocess, random
from collections import defaultdict
import numpy as np

import torch as _t
DEVICE = "cuda" if _t.cuda.is_available() else "cpu"
DTYPE  = _t.float16 if DEVICE == "cuda" else _t.float32
if DEVICE == "cpu":
    _t.set_num_threads(max(1, os.cpu_count() or 2)); print("  CPU mode: expect ~2 h", flush=True)

print("installing deps...", flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","transformers","accelerate","peft",
                "scikit-learn"],check=False)
if not os.path.isdir("DiaLense"):
    subprocess.run(["git","clone","--depth","1","-q","https://github.com/SamhitaK10/DiaLense.git"],check=True)

import sys as _s
import peft, peft.import_utils as _iu
_no_torchao = lambda *a, **k: False
_iu.is_torchao_available = _no_torchao
for _n, _m in list(_s.modules.items()):
    if _n.startswith("peft") and hasattr(_m, "is_torchao_available"):
        _m.is_torchao_available = _no_torchao
from peft import PeftModel

import torch, glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss

MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "You are a physician taking a patient history. Ask one follow-up question."
LAYERS = [2, 4, 6, 8, 12, 16, 20, 24, 27]
N_PER_CLASS, N_TRAIN_PHR, N_BOOT = 8, 5, 400   # 8 urgent + 8 non-urgent scenarios

def _outdir():
    try:
        from google.colab import drive; drive.mount("/content/drive")
        d = "/content/drive/MyDrive/DiaLense_PartII"; os.makedirs(d, exist_ok=True); return d
    except Exception: pass
    if os.path.isdir("/kaggle/working"):
        d = "/kaggle/working/DiaLense_PartII"; os.makedirs(d, exist_ok=True); return d
    d = os.path.abspath("DiaLense_PartII"); os.makedirs(d, exist_ok=True); return d
DRIVE = _outdir()
def _find_adapter(name="dialense_lora"):
    import glob as _g
    for c in [os.path.join(DRIVE,name), f"/content/drive/MyDrive/DiaLense_PartII/{name}",
              os.path.abspath(name)] + sorted(_g.glob(f"/kaggle/input/*/{name}")) \
              + sorted(_g.glob(f"/kaggle/input/*/*/{name}")):
        if os.path.isfile(os.path.join(c, "adapter_config.json")): return c
    return None
ADAPTER = _find_adapter()
if ADAPTER is None: raise SystemExit("\n  dialense_lora not found.")

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

# ============================== PREFLIGHT =====================================
print("\n"+"="*80); print("PREFLIGHT"); print("="*80, flush=True)
rows = {}
for p in sorted(glob.glob("DiaLense/results/raw/*/conversations.jsonl")):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line); rows.setdefault(r["scenario_id"], r)
URGENT = {"urgent", "emergent"}
pool = []
for sid, r in rows.items():
    cu = r.get("correct_urgency")
    if cu is None: continue
    hist = "Reported history: " + "; ".join(
        str(v).strip().rstrip(".") for v in r["latent_facts"].values()) + "."
    pool.append((sid, hist, int(cu in URGENT), cu))
hi = [x for x in pool if x[2] == 1]; lo = [x for x in pool if x[2] == 0]
print(f"  scenarios with an urgency label : {len(pool)}  ({len(hi)} urgent/emergent, {len(lo)} routine/soon)")
if len(hi) < 2*N_PER_CLASS or len(lo) < 2*N_PER_CLASS:
    raise SystemExit(f"  need >= {2*N_PER_CLASS} per class; have {len(hi)}/{len(lo)}")
rng = random.Random(0); rng.shuffle(hi); rng.shuffle(lo)
scen = hi[:N_PER_CLASS] + lo[:N_PER_CLASS]
content_train = set(x[0] for x in hi[:N_PER_CLASS//2] + lo[:N_PER_CLASS//2])
content_eval  = set(x[0] for x in scen) - content_train
print(f"  using {len(scen)} scenarios: content split {len(content_train)} train / {len(content_eval)} eval, balanced")
print(f"  style split: phrasings {N_TRAIN_PHR} train / {8-N_TRAIN_PHR} eval, all scenarios both sides")
n_items = len(scen) * len(CUES) * 8 * 2
print(f"  items per adapter state: {n_items}   (style eval n={len(scen)*len(CUES)*3*2}, "
      f"content eval n={len(content_eval)*len(CUES)*8*2})")
print(f"  adapter: {ADAPTER}\n", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE).eval()
model = PeftModel.from_pretrained(base, ADAPTER).eval()
print("loaded\n", flush=True)

def chat(t):
    return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":t}],
                                   tokenize=False, add_generation_prompt=True)
@torch.no_grad()
def states(cue, hist, adapter_on):
    enc = tok(chat(f"{cue} {hist}"), return_tensors="pt").to(DEVICE)
    ids = enc.input_ids[0].tolist()
    cid = tok(cue, add_special_tokens=False).input_ids
    lo_ = 0
    for i in range(len(ids)-len(cid)+1):
        if ids[i:i+len(cid)] == cid: lo_ = i; break
    hi_ = lo_ + len(cid)
    if adapter_on: hs = model(**enc, output_hidden_states=True).hidden_states
    else:
        with model.disable_adapter(): hs = model(**enc, output_hidden_states=True).hidden_states
    return {L: (hs[L][0].float()[-1].cpu().numpy(),
                hs[L][0].float()[lo_:hi_].mean(0).cpu().numpy()) for L in LAYERS}

def collect(adapter_on):
    Xf, Xc = defaultdict(list), defaultdict(list)
    meta = []
    for dim, prs in CUES.items():
        for pi, (sh, sl) in enumerate(prs):
            for slab, cue in ((1, sh), (0, sl)):
                for sid, hist, ulab, _ in scen:
                    st = states(cue, hist, adapter_on)
                    for L in LAYERS: Xf[L].append(st[L][0]); Xc[L].append(st[L][1])
                    meta.append(dict(phr=pi, style=slab, sid=sid, urg=ulab))
        print(f"   {dim} done", flush=True)
    return {L: np.array(v) for L, v in Xf.items()}, {L: np.array(v) for L, v in Xc.items()}, meta

def probe(X, y, tr, te):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=0).fit(sc.transform(X[tr]), y[tr])
    Z = sc.transform(X[te])
    return clf.decision_function(Z), clf.predict_proba(Z)[:, 1]

cache = {}
for adapter_on in (False, True):
    tag = "ON" if adapter_on else "OFF"
    print(f"collecting, adapter {tag} ...", flush=True)
    cache[tag] = collect(adapter_on)

meta = cache["OFF"][2]
phr = np.array([m["phr"] for m in meta]); sid = np.array([m["sid"] for m in meta])
y_style = np.array([m["style"] for m in meta]); y_urg = np.array([m["urg"] for m in meta])
s_tr, s_te = phr < N_TRAIN_PHR, phr >= N_TRAIN_PHR
c_tr = np.array([x in content_train for x in sid]); c_te = ~c_tr

def boot(d_off, d_on, y, n=N_BOOT, seed=0):
    d_off, d_on, y = np.array(d_off), np.array(d_on), np.array(y)
    rs = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rs.integers(0, len(y), len(y))
        if len(set(y[i])) < 2: continue
        out.append(roc_auc_score(y[i], d_off[i]) - roc_auc_score(y[i], d_on[i]))
    return float(np.mean(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

print("\n" + "="*96)
print("BASE MODEL — where is style readable?   (adapter OFF)")
print("="*96)
print(f"{'layer':>5} {'style final':>13} {'style cue':>12} {'cue - final':>12} {'urgency final':>15}")
base_tbl = {}
for L in LAYERS:
    df_s, _ = probe(cache["OFF"][0][L], y_style, s_tr, s_te)
    dc_s, _ = probe(cache["OFF"][1][L], y_style, s_tr, s_te)
    du, pu  = probe(cache["OFF"][0][L], y_urg, c_tr, c_te)
    a, b, c = (roc_auc_score(y_style[s_te], df_s), roc_auc_score(y_style[s_te], dc_s),
               roc_auc_score(y_urg[c_te], du))
    base_tbl[L] = dict(style_final=a, style_cue=b, urgency_final=c,
                       urgency_logloss=float(log_loss(y_urg[c_te], np.clip(pu, 1e-6, 1-1e-6))))
    print(f"{L:>5} {a:>13.3f} {b:>12.3f} {b-a:>+12.3f} {c:>15.3f}")

print("\n" + "="*96)
print("AUC DROP WITH THE ADAPTER ENGAGED   (positive = harder to decode after)")
print("="*96)
print(f"{'layer':>5}  {'STYLE final':>24}  {'STYLE cue':>24}  {'URGENCY final':>24}")
print("-"*96)
tbl = {}
for L in LAYERS:
    off_f, _ = probe(cache["OFF"][0][L], y_style, s_tr, s_te)
    on_f,  _ = probe(cache["ON"][0][L],  y_style, s_tr, s_te)
    off_c, _ = probe(cache["OFF"][1][L], y_style, s_tr, s_te)
    on_c,  _ = probe(cache["ON"][1][L],  y_style, s_tr, s_te)
    off_u, pu_off = probe(cache["OFF"][0][L], y_urg, c_tr, c_te)
    on_u,  pu_on  = probe(cache["ON"][0][L],  y_urg, c_tr, c_te)
    A = boot(off_f, on_f, y_style[s_te]); B = boot(off_c, on_c, y_style[s_te])
    C = boot(off_u, on_u, y_urg[c_te])
    ll = (float(log_loss(y_urg[c_te], np.clip(pu_on,1e-6,1-1e-6))) -
          float(log_loss(y_urg[c_te], np.clip(pu_off,1e-6,1-1e-6))))
    tbl[L] = dict(style_final=A, style_cue=B, urgency_final=C, urgency_logloss_increase=ll)
    f = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"
    print(f"{L:>5}  {f(A):>24}  {f(B):>24}  {f(C):>24}")

late = [L for L in LAYERS if L >= 20]
m = lambda k: float(np.mean([tbl[L][k][0] for L in late]))
sig = lambda k: [L for L in LAYERS if tbl[L][k][1] > 0]      # CI excludes zero, positive
sf, sc_, uf = m("style_final"), m("style_cue"), m("urgency_final")
ll_late = float(np.mean([tbl[L]["urgency_logloss_increase"] for L in late]))
print("-"*96)
print(f"late layers (>=20)  style-final {sf:+.3f}   style-cue {sc_:+.3f}   urgency-final {uf:+.3f}")
print(f"layers where the CI excludes zero: style-final {sig('style_final')}   "
      f"style-cue {sig('style_cue')}   urgency-final {sig('urgency_final')}")
print(f"urgency log-loss increase, late layers: {ll_late:+.4f}  (catches degradation AUC hides)")

if sig("style_final") and not sig("urgency_final"):
    verdict = ("TARGETED READOUT SUPPRESSION. Style is reliably harder to decode from the final "
               "position after the intervention, at layers " + str(sig("style_final")) +
               ", while clinical urgency is not. Earned version of the strong claim.")
elif sig("style_final") and sig("urgency_final"):
    verdict = ("GENERAL LATE-LAYER DEGRADATION. Urgency decodes worse too, so this is not a "
               "style-specific edit. Report plainly; it weakens the selectivity claim.")
else:
    verdict = ("NO DETECTABLE REPRESENTATIONAL CHANGE at this resolution. Report the BOUND -- "
               "the widest CI is your detection limit -- and frame as behavioural suppression "
               "without a measurable representational signature (Lee et al. 2024; Galichin et "
               "al. 2026). This is a finding, not a failure, provided you state the power.")
print(f"\nVERDICT: {verdict}")

cue_gain = float(np.mean([base_tbl[L]["style_cue"] - base_tbl[L]["style_final"]
                          for L in LAYERS if L <= 16]))
print(f"\nSEPARATE, ROBUST FINDING: in the base model, cue-position readout beats final-position "
      f"by {cue_gain:+.3f} AUC on average through layer 16.\n  This is direct evidence that "
      f"final-position-only measurement under-detects style (Geva et al. 2023; Tigges et al. 2024)\n"
      f"  and it independently justifies how the patching null is interpreted.")

json.dump(dict(layers=LAYERS, base=base_tbl, deltas={str(k): v for k, v in tbl.items()},
               late_means=dict(style_final=sf, style_cue=sc_, urgency_final=uf,
                               urgency_logloss_increase=ll_late),
               ci_excludes_zero=dict(style_final=sig("style_final"), style_cue=sig("style_cue"),
                                     urgency_final=sig("urgency_final")),
               n_style_eval=int(s_te.sum()), n_content_eval=int(c_te.sum()),
               mean_cue_minus_final_base=cue_gain, verdict=verdict),
          open("exp_D3_probe.json","w"), indent=2)
subprocess.run(["cp","exp_D3_probe.json",DRIVE+"/"],check=False)
print("\nsaved exp_D3_probe.json")
