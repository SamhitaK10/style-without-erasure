# ========================= PASTE INTO ONE COLAB CELL ===========================
# EXP_D2 — fix the content control, and put intervals on the AUC deltas.
# ~10 min on a T4.  Output: exp_D2_content_control.json
#
# WHY THIS IS NEEDED
#   EXP_D's content control was broken and I wrote it. The label was
#   `int(si < len(scen)//2)` -- "is this in the first half of the scenario list"
#   -- computed separately for the train split and the eval split. Those are
#   DISJOINT scenario sets, so the label means something different in each: the
#   probe learned to separate training scenarios 0-5 from 6-11 and was then asked
#   about eval scenarios 0-5 vs 6-11, which have no relationship to it. That is
#   why content AUC came out at 0.207-0.506, i.e. at or below chance. It was
#   measuring nothing, so "content-control drop -0.008" is not evidence of
#   anything.
#
#   This matters because the content control is the one thing separating
#   "the adapter targeted the style readout" from "the adapter degraded late-layer
#   final-position decoding in general". Without it, your headline D result has an
#   alternative explanation.
#
# THE FIX
#   A real content contrast, built the same way as the style contrast:
#   two FIXED clinical histories, every cue sentence paired with both, and the
#   probe asked which history. The label is identical in train and eval, and the
#   split is over PHRASINGS (5 train / 3 held out) rather than scenarios, so the
#   probe cannot memorise the cue text either.
#
#   Style is re-measured here under exactly the same protocol so the two are
#   directly comparable -- same layers, same probe, same split, same items.
#
# ALSO ADDED
#   Percentile bootstrap intervals on every AUC delta, resampling eval items.
#   EXP_D reported a mean drop of +0.062 with no interval, and that mean is
#   misleading anyway: the drop is ~0.00 at layer 2 and ~0.14 at layer 27. Report
#   the per-layer curve, not the mean.
#
# WHAT TO CONCLUDE
#   style drop large at late layers AND content drop ~0
#       -> targeted suppression of the style readout. That is the paper's claim,
#          and it is a strong one.
#   both drop at late layers
#       -> the adapter degraded late-layer final-position decoding generally.
#          Still publishable, much weaker, and you must say so.
# ==============================================================================
import os, sys, json, subprocess, random
from collections import defaultdict
import numpy as np

import torch as _t
DEVICE = "cuda" if _t.cuda.is_available() else "cpu"
DTYPE  = _t.float16 if DEVICE == "cuda" else _t.float32
if DEVICE == "cpu":
    _t.set_num_threads(max(1, os.cpu_count() or 2))
    print("  no GPU -> CPU, fp32. Slower; same AUCs.", flush=True)

print("installing deps...", flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","transformers","accelerate","peft",
                "scikit-learn"],check=False)
if not os.path.isdir("DiaLense"):
    subprocess.run(["git","clone","--depth","1","-q","https://github.com/SamhitaK10/DiaLense.git"],check=True)

# ---- peft / torchao guard (verified against peft 0.20.0) --------------------
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
from sklearn.metrics import roc_auc_score

MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "You are a physician taking a patient history. Ask one follow-up question."
LAYERS = [2, 4, 6, 8, 12, 16, 20, 24, 27]
N_TRAIN_PHR, N_BOOT = 5, 400

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
    cands = [os.path.join(DRIVE, name), f"/content/drive/MyDrive/DiaLense_PartII/{name}",
             os.path.abspath(name)] + sorted(_g.glob(f"/kaggle/input/*/{name}")) \
             + sorted(_g.glob(f"/kaggle/input/*/*/{name}"))
    for c in cands:
        if os.path.isfile(os.path.join(c, "adapter_config.json")): return c
    return None
ADAPTER = _find_adapter()
if ADAPTER is None: raise SystemExit("\n  dialense_lora not found.")
print(f"outputs -> {DRIVE}\nadapter -> {ADAPTER}", flush=True)

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

# ---- two FIXED clinical histories: the content contrast ----------------------
facts, seen = [], set()
for p in sorted(glob.glob("DiaLense/results/raw/*/conversations.jsonl")):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r["scenario_id"] in seen: continue
        seen.add(r["scenario_id"])
        facts.append("Reported history: " + "; ".join(
            str(v).strip().rstrip(".") for v in r["latent_facts"].values()) + ".")
rng = random.Random(0); idx = list(range(len(facts))); rng.shuffle(idx)
HIST_A, HIST_B = facts[idx[0]], facts[idx[1]]
train_phr = {d: v[:N_TRAIN_PHR] for d, v in CUES.items()}
eval_phr  = {d: v[N_TRAIN_PHR:] for d, v in CUES.items()}
print(f"\nCONTENT CONTRAST — two fixed histories, label consistent across splits")
print(f"  A: {HIST_A[:90]}...")
print(f"  B: {HIST_B[:90]}...")
print(f"  split is over PHRASINGS: {N_TRAIN_PHR} train / {8-N_TRAIN_PHR} held out per dimension\n", flush=True)

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
    cue_ids = tok(cue, add_special_tokens=False).input_ids
    lo = 0
    for i in range(len(ids)-len(cue_ids)+1):
        if ids[i:i+len(cue_ids)] == cue_ids: lo = i; break
    hi = lo + len(cue_ids)
    if adapter_on:
        hs = model(**enc, output_hidden_states=True).hidden_states
    else:
        with model.disable_adapter():
            hs = model(**enc, output_hidden_states=True).hidden_states
    return {L: (hs[L][0].float()[-1].cpu().numpy(),
                hs[L][0].float()[lo:hi].mean(0).cpu().numpy()) for L in LAYERS}

def collect(phr, adapter_on):
    Xf, Xc = defaultdict(list), defaultdict(list)
    y_style, y_content = [], []
    for dim, prs in phr.items():
        for sh, sl in prs:
            for slab, cue in ((1, sh), (0, sl)):
                for clab, hist in ((1, HIST_A), (0, HIST_B)):
                    st = states(cue, hist, adapter_on)
                    for L in LAYERS:
                        Xf[L].append(st[L][0]); Xc[L].append(st[L][1])
                    y_style.append(slab); y_content.append(clab)
    return ({L: np.array(v) for L, v in Xf.items()},
            {L: np.array(v) for L, v in Xc.items()},
            np.array(y_style), np.array(y_content))

def fit_scores(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=0).fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))

res = {}
for adapter_on in (False, True):
    tag = "ON" if adapter_on else "OFF"
    print(f"collecting, adapter {tag} ...", flush=True)
    Xtr_f, Xtr_c, ytr_s, ytr_k = collect(train_phr, adapter_on)
    Xte_f, Xte_c, yte_s, yte_k = collect(eval_phr,  adapter_on)
    row = {}
    for L in LAYERS:
        s_f = fit_scores(Xtr_f[L], ytr_s, Xte_f[L])
        s_c = fit_scores(Xtr_c[L], ytr_s, Xte_c[L])
        k_f = fit_scores(Xtr_f[L], ytr_k, Xte_f[L])
        row[L] = dict(style_final=s_f.tolist(), style_cue=s_c.tolist(), content_final=k_f.tolist())
        print(f"   L{L:<3} style(final) {roc_auc_score(yte_s, s_f):.3f}  "
              f"style(cue) {roc_auc_score(yte_s, s_c):.3f}  "
              f"content(final) {roc_auc_score(yte_k, k_f):.3f}", flush=True)
    res[tag] = dict(scores=row, y_style=yte_s.tolist(), y_content=yte_k.tolist())

# ---- bootstrap the AUC DELTA, resampling eval items -------------------------
def boot_delta(off, on, y, n=N_BOOT, seed=0):
    off, on, y = np.array(off), np.array(on), np.array(y)
    rs = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rs.integers(0, len(y), len(y))
        if len(set(y[i])) < 2: continue
        out.append(roc_auc_score(y[i], off[i]) - roc_auc_score(y[i], on[i]))
    return float(np.mean(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

ys, yk = np.array(res["OFF"]["y_style"]), np.array(res["OFF"]["y_content"])
print("\n" + "="*84)
print("AUC DROP WHEN THE ADAPTER IS ENGAGED   (positive = harder to decode after)")
print("="*84)
print(f"{'layer':>5}  {'STYLE final':>22}  {'STYLE cue':>22}  {'CONTENT final':>22}")
print("-"*84)
table = {}
for L in LAYERS:
    a = boot_delta(res["OFF"]["scores"][L]["style_final"], res["ON"]["scores"][L]["style_final"], ys)
    b = boot_delta(res["OFF"]["scores"][L]["style_cue"],   res["ON"]["scores"][L]["style_cue"],   ys)
    c = boot_delta(res["OFF"]["scores"][L]["content_final"],res["ON"]["scores"][L]["content_final"],yk)
    table[L] = dict(style_final=a, style_cue=b, content_final=c)
    f = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"
    print(f"{L:>5}  {f(a):>22}  {f(b):>22}  {f(c):>22}")

late = [L for L in LAYERS if L >= 20]
sf_late = float(np.mean([table[L]["style_final"][0] for L in late]))
cf_late = float(np.mean([table[L]["content_final"][0] for L in late]))
sc_late = float(np.mean([table[L]["style_cue"][0] for L in late]))
print("-"*84)
print(f"late layers (>=20)   style-final {sf_late:+.3f}   style-cue {sc_late:+.3f}   content-final {cf_late:+.3f}")

if sf_late > 0.05 and abs(cf_late) < 0.05 and abs(sc_late) < 0.05:
    verdict = ("TARGETED READOUT SUPPRESSION. The style representation survives at the cue "
               "tokens and clinical content still decodes at the final position; what the "
               "adapter removed is the projection of style onto the final position in late "
               "layers. That is a specific, defensible mechanistic claim and it matches the "
               "patching result, where the late-layer transition is where the model commits.")
elif sf_late > 0.05 and cf_late > 0.05:
    verdict = ("GENERAL LATE-LAYER DEGRADATION. Content decodes worse too, so this is not a "
               "style-specific edit. Report it plainly; it weakens the selectivity claim.")
else:
    verdict = ("NO RELIABLE DROP once intervals are attached. Report the bound, not the point "
               "estimate, and do not claim the representation changed.")
print(f"\nVERDICT: {verdict}")
json.dump(dict(layers=LAYERS, deltas={str(k): v for k, v in table.items()},
               late_layer_means=dict(style_final=sf_late, style_cue=sc_late, content_final=cf_late),
               histories=dict(A=HIST_A, B=HIST_B), verdict=verdict),
          open("exp_D2_content_control.json","w"), indent=2)
try: subprocess.run(["cp","exp_D2_content_control.json",DRIVE+"/"],check=False)
except Exception: pass
print("\nsaved exp_D2_content_control.json")
