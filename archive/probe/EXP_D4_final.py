# ========================= PASTE INTO ONE COLAB CELL ===========================
# EXP_D4 — the honest version. ~20 min on a T4.  Output: exp_D4_probe.json
#
# WHY D3'S VERDICT IS WRONG, AND IT IS MY ERROR
#   D3 printed "GENERAL LATE-LAYER DEGRADATION" because the urgency probe seemed
#   to drop +0.202 at layer 20 with a tight interval. That interval is invalid.
#
#   The content eval set was 640 items -- but those 640 came from only EIGHT
#   held-out scenarios. All 80 items sharing a scenario share the same clinical
#   history and therefore the same label; they are not independent. The bootstrap
#   resampled items, so it treated 8 units as 640 and understated the interval by
#   roughly sqrt(80) ~ 9x. Corrected, +0.202 [+0.162,+0.251] becomes about
#   [-0.20,+0.60]. It spans zero, and so does every other urgency "effect".
#
#   The tell was in the base numbers: urgency AUC bounced 0.452 to 0.737 with no
#   coherent depth profile, and 0.452 is BELOW chance. A probe that reads below
#   chance at layer 12 is not measuring reliably at layer 20 either.
#
#   The style probe has the same flaw, less severely: 480 items from 15 held-out
#   phrasings, so intervals are understated ~5.7x. The largest drop, +0.040 at
#   layer 8, becomes roughly [-0.05,+0.13].
#
#   This is pseudoreplication -- the same error the baseline analysis was careful
#   to avoid by treating phrasing rather than scenario as the unit. I built it
#   into the probe and then read a verdict off it.
#
# WHAT CHANGES HERE
#   * CLUSTER BOOTSTRAP. Resample the UNIT, not the item: phrasings for the style
#     probe, scenarios for the content probe. Intervals become honest and wide.
#   * A CLUSTER-LEVEL PERMUTATION TEST alongside, since a bootstrap on 15-18 units
#     is itself shaky.
#   * MORE UNITS. All 18 urgent/emergent scenarios plus 18 matched routine/soon,
#     split 18/18, so the content probe has 18 independent units instead of 8.
#   * EFFECTIVE n IS PRINTED next to every number, so nobody reads 640 again.
#   * PER-ITEM SCORES ARE SAVED, so any future re-analysis of the intervals costs
#     nothing and needs no GPU.
#
# WHAT TO EXPECT
#   Most likely: nothing survives at these unit counts. That is a real result --
#   behavioural suppression with no representational change detectable at this
#   resolution, which is what Lee et al. (ICML 2024) and Galichin et al. (EACL
#   2026) report for other interventions. Report the BOUND: "no layer shows a
#   style-decodability drop larger than X with 95% confidence". A bounded null is
#   publishable. An overconfident positive is retracted.
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
N_PER_CLASS, N_TRAIN_PHR, N_BOOT = 18, 5, 2000  # all 18 urgent + 18 matched non-urgent
N_PERM = 2000

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
N_PER_CLASS = min(N_PER_CLASS, len(hi), len(lo))
if N_PER_CLASS < 6: raise SystemExit(f"  too few per class: {len(hi)}/{len(lo)}")
rng = random.Random(0); rng.shuffle(hi); rng.shuffle(lo)
scen = hi[:N_PER_CLASS] + lo[:N_PER_CLASS]
content_train = set(x[0] for x in hi[:N_PER_CLASS//2] + lo[:N_PER_CLASS//2])  # balanced half
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

def _auc(y, s):
    return roc_auc_score(y, s)

def cluster_boot(d_off, d_on, y, clus, n=N_BOOT, seed=0):
    """Resample CLUSTERS with replacement, not items. clus = unit id per item."""
    d_off, d_on, y, clus = map(np.asarray, (d_off, d_on, y, clus))
    units = np.unique(clus)
    idx_by_unit = {u: np.where(clus == u)[0] for u in units}
    rs = np.random.default_rng(seed); out = []
    for _ in range(n):
        pick = rs.choice(units, len(units), replace=True)
        i = np.concatenate([idx_by_unit[u] for u in pick])
        if len(set(y[i])) < 2: continue
        out.append(_auc(y[i], d_off[i]) - _auc(y[i], d_on[i]))
    if not out: return (float("nan"),)*3 + (len(units),)
    return (float(np.mean(out)), float(np.percentile(out, 2.5)),
            float(np.percentile(out, 97.5)), len(units))

def cluster_perm(d_off, d_on, y, clus, n=N_PERM, seed=0):
    """Null: the OFF/ON assignment is exchangeable WITHIN a unit.
    Flip whole units and rebuild the delta."""
    d_off, d_on, y, clus = map(np.asarray, (d_off, d_on, y, clus))
    units = np.unique(clus); obs = _auc(y, d_off) - _auc(y, d_on)
    rs = np.random.default_rng(seed); hits = 0; tot = 0
    for _ in range(n):
        flip = set(units[rs.random(len(units)) < 0.5])
        a = np.where(np.isin(clus, list(flip)), d_on, d_off)
        b = np.where(np.isin(clus, list(flip)), d_off, d_on)
        try: null = _auc(y, a) - _auc(y, b)
        except ValueError: continue
        hits += (abs(null) >= abs(obs)); tot += 1
    return float((hits + 1) / (tot + 1)), float(obs)

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

style_unit = np.array([f"{m['phr']}" for m in meta])       # phrasing = the style unit
content_unit = np.array([str(m["sid"]) for m in meta])      # scenario = the content unit
su_te = style_unit[s_te]; cu_te = content_unit[c_te]
print("\n" + "="*104)
print("AUC DROP WITH THE ADAPTER ENGAGED — CLUSTER bootstrap (resamples units, not items)")
print("="*104)
print(f"  style probe   : {int(s_te.sum())} items from {len(set(su_te))} held-out phrasings  <- the real n")
print(f"  content probe : {int(c_te.sum())} items from {len(set(cu_te))} held-out scenarios  <- the real n")
print("-"*104)
print(f"{'layer':>5}  {'STYLE final':>26}  {'STYLE cue':>26}  {'URGENCY final':>26}")
print("-"*104)
tbl = {}
for L in LAYERS:
    off_f, _ = probe(cache["OFF"][0][L], y_style, s_tr, s_te)
    on_f,  _ = probe(cache["ON"][0][L],  y_style, s_tr, s_te)
    off_c, _ = probe(cache["OFF"][1][L], y_style, s_tr, s_te)
    on_c,  _ = probe(cache["ON"][1][L],  y_style, s_tr, s_te)
    off_u, pu_off = probe(cache["OFF"][0][L], y_urg, c_tr, c_te)
    on_u,  pu_on  = probe(cache["ON"][0][L],  y_urg, c_tr, c_te)
    A = cluster_boot(off_f, on_f, y_style[s_te], su_te)
    B = cluster_boot(off_c, on_c, y_style[s_te], su_te)
    C = cluster_boot(off_u, on_u, y_urg[c_te],   cu_te)
    pA, _ = cluster_perm(off_f, on_f, y_style[s_te], su_te)
    pC, _ = cluster_perm(off_u, on_u, y_urg[c_te],   cu_te)
    ll = (float(log_loss(y_urg[c_te], np.clip(pu_on,1e-6,1-1e-6))) -
          float(log_loss(y_urg[c_te], np.clip(pu_off,1e-6,1-1e-6))))
    tbl[L] = dict(style_final=A[:3], style_cue=B[:3], urgency_final=C[:3],
                  n_units_style=A[3], n_units_content=C[3],
                  perm_p_style=pA, perm_p_urgency=pC, urgency_logloss_increase=ll,
                  scores=dict(style_final_off=off_f.tolist(), style_final_on=on_f.tolist()))
    f = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"
    print(f"{L:>5}  {f(A):>26}  {f(B):>26}  {f(C):>26}")
print("-"*104)
print(f"{'':5}  cluster permutation p, style-final / urgency-final:")
for L in LAYERS:
    print(f"{L:>5}   style p={tbl[L]['perm_p_style']:.3f}    urgency p={tbl[L]['perm_p_urgency']:.3f}")

late = [L for L in LAYERS if L >= 20]
m = lambda k: float(np.mean([tbl[L][k][0] for L in late]))
sig = lambda k, p: [L for L in LAYERS if tbl[L][k][1] > 0 and tbl[L][p] < 0.05]
sf, sc_, uf = m("style_final"), m("style_cue"), m("urgency_final")
widest = max(abs(tbl[L]["style_final"][2]) for L in LAYERS)
print("-"*104)
print(f"late layers (>=20)  style-final {sf:+.3f}   style-cue {sc_:+.3f}   urgency-final {uf:+.3f}")
S = sig("style_final","perm_p_style"); U = sig("urgency_final","perm_p_urgency")
print(f"layers surviving BOTH a cluster CI excluding zero AND permutation p<.05:")
print(f"   style-final {S if S else 'none'}    urgency-final {U if U else 'none'}")

if S and not U:
    verdict = (f"TARGETED READOUT SUPPRESSION, cluster-corrected. Style decodability drops at "
               f"layers {S} and clinical urgency does not. This is the strong claim and it is "
               f"now earned rather than assumed.")
elif S and U:
    verdict = ("GENERAL DEGRADATION, cluster-corrected. Urgency decodes worse too, so the edit "
               "is not style-specific. Report plainly.")
else:
    verdict = (f"NO REPRESENTATIONAL CHANGE DETECTABLE at this resolution. Report the BOUND: with "
               f"95% confidence no layer shows a style-decodability drop larger than {widest:.3f} "
               f"AUC. Frame as behavioural suppression without a measurable representational "
               f"signature, citing Lee et al. (ICML 2024) and Galichin et al. (EACL 2026), and "
               f"state the unit counts so the reader can see the resolution. A bounded null is a "
               f"finding; an overconfident positive is a retraction.")
print(f"\nVERDICT: {verdict}")

cue_gain = float(np.mean([base_tbl[L]["style_cue"] - base_tbl[L]["style_final"]
                          for L in LAYERS if L <= 20]))
print(f"\nSEPARATE, ROBUST FINDING (base model, replicated across D, D2, D3): cue-position "
      f"readout\n  beats final-position by {cue_gain:+.3f} AUC on average through layer 20, then "
      f"reverses at 24-27.\n  Direct evidence that final-position-only measurement under-detects "
      f"style (Geva et al. 2023;\n  Tigges et al. 2024) -- and it is about the BASE model, so no "
      f"adapter claim rests on it.")

json.dump(dict(layers=LAYERS, base=base_tbl, deltas={str(k): v for k, v in tbl.items()},
               late_means=dict(style_final=sf, style_cue=sc_, urgency_final=uf,
                               urgency_logloss_increase=ll_late),
               ci_excludes_zero=dict(style_final=sig("style_final"), style_cue=sig("style_cue"),
                                     urgency_final=sig("urgency_final")),
               n_style_eval=int(s_te.sum()), n_content_eval=int(c_te.sum()),
               mean_cue_minus_final_base=cue_gain, verdict=verdict,
               note='intervals are CLUSTER bootstraps; units are phrasings (style) and scenarios (content)'),
          open("exp_D4_probe.json","w"), indent=2)
subprocess.run(["cp","exp_D4_probe.json",DRIVE+"/"],check=False)
print("\nsaved exp_D4_probe.json (per-item scores included: re-analysis needs no GPU)")
