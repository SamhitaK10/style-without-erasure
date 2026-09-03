# ========================= PASTE INTO ONE COLAB CELL ===========================
# EXPERIMENT D — did the adapter ERASE the style representation, or only SUPPRESS
# the readout?  Runtime ~10 min on a T4.  Output: exp_D_probe.json
#
# WHY THIS IS THE MOST IMPORTANT REMAINING EXPERIMENT
#   Your intervention changes the output distribution. It says nothing about
#   whether style information is still sitting in the residual stream, fully
#   available and merely unused. The literature on exactly this is deep:
#     Gonen & Goldberg 2019   debiased embeddings still cluster by gender
#     Lee et al. ICML 2024    DPO does not delete toxicity vectors; it steers around them
#     Jain et al. ICLR 2024   fine-tuning learns a removable "wrapper"
#     Galichin et al. EACL26  fine-tuning shifts the distribution over EXISTING features
#     Basani & Chhabra 2026   low-rank updates REDISTRIBUTE rather than eliminate
#   That last one names the LoRA critique with mechanistic support. A reviewer
#   will raise it. Right now you cannot answer.
#
# BOTH OUTCOMES ARE PUBLISHABLE. Only the unmeasured state is bad.
#   AUC unchanged -> behavioural suppression without representational erasure,
#       consistent with Lee et al. and Galichin et al. That is a finding, and it
#       gives the paper a third leg. Defensive citation: Ponkshe et al. ICLR 2026
#       show safety subspaces are not linearly separable from task-useful ones,
#       so non-erasure may be CORRECT rather than a defect.
#   AUC drops -> a stronger claim than you currently make. Say it.
#
# WHAT IT DOES
#   Trains a logistic probe to classify high- vs low-style from the residual
#   stream at several layers, with the adapter bypassed and engaged. Same items,
#   same layers, same split. Reports AUC and a permutation null.
#
# CONTROLS THAT MAKE IT INTERPRETABLE
#   * probe trained on TRAIN phrasings, evaluated on HELD-OUT phrasings, so it
#     cannot memorise the sentences
#   * label-shuffled permutation null per layer: an AUC that beats chance only
#     because the probe overfits will show up here
#   * a CONTENT probe (which scenario half) as a positive control: if the
#     adapter had damaged the representation globally, content AUC would fall too
#   * mean-pooled over CUE TOKEN POSITIONS as well as the final position, because
#     Geva et al. 2023 and Tigges et al. 2024 both predict final-position-only
#     readouts under-detect a distributed feature
# ==============================================================================
import os, sys, json, subprocess, random, gc
from collections import defaultdict

# Inference only -- forward passes with hidden states, plus sklearn probes.
# Runs on CPU. Budget ~1.5-3 h there against ~10 min on a T4: this collects
# roughly 1,900 forward passes with output_hidden_states=True, which is the
# expensive part. fp32 is used on CPU because fp16 matmul on CPU is slower, not
# faster; probe AUCs are unaffected by the dtype at this precision.
import torch as _t
DEVICE = "cuda" if _t.cuda.is_available() else "cpu"
DTYPE  = _t.float16 if DEVICE == "cuda" else _t.float32
if DEVICE == "cpu":
    _t.set_num_threads(max(1, os.cpu_count() or 2))
    print("  no GPU -> running on CPU in fp32. Slower; same AUCs.", flush=True)

print("installing deps...", flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","transformers","accelerate","peft",
                "scikit-learn","scipy"],check=False)
if not os.path.isdir("DiaLense"):
    subprocess.run(["git","clone","--depth","1","-q","https://github.com/SamhitaK10/DiaLense.git"],check=True)

# ---- peft / torchao guard  (verified against peft 0.20.0) --------------------
# is_torchao_available() is @lru_cache'd and RAISES rather than returning False
# on an old torchao, and it raises LAZILY inside get_peft_model — so a try/except
# around `import peft` never fires. Colab currently ships torchao 0.10.0 while
# peft wants >= 0.16.0. peft only needs torchao for int4/int8 torchao quantization,
# which we do not use, so we tell it torchao is absent. Four modules bind the
# symbol at import time (peft.import_utils, peft.utils.quantization_utils,
# peft.tuners.lora.torchao, peft.tuners.hira); patching all of them covers
# get_peft_model, forward, gradient checkpointing, save_pretrained,
# PeftModel.from_pretrained and disable_adapter.
import sys as _s
import peft, peft.import_utils as _iu
_no_torchao = lambda *a, **k: False
_iu.is_torchao_available = _no_torchao
for _n, _m in list(_s.modules.items()):
    if _n.startswith("peft") and hasattr(_m, "is_torchao_available"):
        _m.is_torchao_available = _no_torchao
from peft import PeftModel

import numpy as np, torch, glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM  = "You are a physician taking a patient history. Ask one follow-up question."
LAYERS  = [2, 4, 6, 8, 12, 16, 20, 24, 27]
N_SCEN  = 20
N_TRAIN_PHR = 5           # must match the training split exactly
N_PERM  = 200

# ---- output directory: works on Colab, Kaggle, or a plain machine -----------
def _outdir():
    try:
        from google.colab import drive; drive.mount("/content/drive")
        d = "/content/drive/MyDrive/DiaLense_PartII"; os.makedirs(d, exist_ok=True); return d
    except Exception: pass
    if os.path.isdir("/kaggle/working"):
        d = "/kaggle/working/DiaLense_PartII"; os.makedirs(d, exist_ok=True); return d
    d = os.path.abspath("DiaLense_PartII"); os.makedirs(d, exist_ok=True); return d
DRIVE = _outdir()
print(f"outputs -> {DRIVE}", flush=True)

def _find_adapter(name="dialense_lora"):
    """Look wherever the adapter plausibly lives. On Kaggle, upload it as a
    Dataset first (see KAGGLE.md) and it lands under /kaggle/input/."""
    import glob as _g
    cands = [os.path.join(DRIVE, name), f"/content/drive/MyDrive/DiaLense_PartII/{name}",
             os.path.abspath(name)] + sorted(_g.glob(f"/kaggle/input/*/{name}")) \
             + sorted(_g.glob(f"/kaggle/input/*/*/{name}"))
    for c in cands:
        if os.path.isfile(os.path.join(c, "adapter_config.json")): return c
    return None

ADAPTER = _find_adapter()
if ADAPTER is None:
    raise SystemExit(
        "\n  dialense_lora not found. Looked in the output dir, Google Drive, the\n"
        "  working directory, and /kaggle/input/*/.\n"
        "  On Kaggle: download dialense_lora from Drive, upload it as a Dataset,\n"
        "  and Add Input it to this notebook. See KAGGLE.md.")
print(f"adapter  -> {ADAPTER}", flush=True)

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

# ================================ PREFLIGHT ===================================
print("\n"+"="*76); print("PREFLIGHT"); print("="*76, flush=True)
facts, seen = [], set()
for p in sorted(glob.glob("DiaLense/results/raw/*/conversations.jsonl")):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r["scenario_id"] in seen: continue
        seen.add(r["scenario_id"])
        facts.append("Reported history: " + "; ".join(
            str(v).strip().rstrip(".") for v in r["latent_facts"].values()) + ".")
rng = random.Random(0)                       # SAME seed as the training script
idx = list(range(len(facts))); rng.shuffle(idx)
train_scen = [facts[i] for i in idx[:30]]
eval_scen  = [facts[i] for i in idx[30:30+N_SCEN]]
train_phr  = {d: v[:N_TRAIN_PHR] for d, v in CUES.items()}
eval_phr   = {d: v[N_TRAIN_PHR:] for d, v in CUES.items()}
print(f"  scenarios total {len(facts)} | probe-train {len(train_scen)} | probe-eval {len(eval_scen)}")
print(f"  phrasings: train {N_TRAIN_PHR}/dim, held out {8-N_TRAIN_PHR}/dim")
print(f"  adapter  : {ADAPTER}")
print(f"  layers   : {LAYERS}\n", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE).eval()
NL = base.config.num_hidden_layers
if max(LAYERS) >= NL + 1: raise SystemExit(f"  model has {NL} layers; LAYERS out of range")
model = PeftModel.from_pretrained(base, ADAPTER).eval()
_mem = f"{torch.cuda.memory_allocated()/2**30:.2f} GiB" if DEVICE == "cuda" else "cpu"
print(f"base + adapter loaded on {DEVICE} | {_mem}\n", flush=True)

def chat(t):
    return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":t}],
                                   tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def states(text, cue, adapter_on):
    """Residual stream at the final position AND mean-pooled over the cue tokens."""
    full = chat(f"{cue} {text}")
    enc = tok(full, return_tensors="pt").to(DEVICE)
    ids = enc.input_ids[0].tolist()
    cue_ids = tok(cue, add_special_tokens=False).input_ids
    # locate the cue span by subsequence search (offsets are unreliable after templating)
    lo = 0
    for i in range(len(ids)-len(cue_ids)+1):
        if ids[i:i+len(cue_ids)] == cue_ids: lo = i; break
    hi = lo + len(cue_ids)
    ctx = torch.no_grad()
    if adapter_on:
        hs = model(**enc, output_hidden_states=True).hidden_states
    else:
        with model.disable_adapter():
            hs = model(**enc, output_hidden_states=True).hidden_states
    out = {}
    for L in LAYERS:
        h = hs[L][0].float()
        out[L] = (h[-1].cpu().numpy(), h[lo:hi].mean(0).cpu().numpy())
    return out

def collect(phr, scen, adapter_on):
    X_fin, X_cue, y, ycontent = defaultdict(list), defaultdict(list), [], []
    for dim, prs in phr.items():
        for sh, sl in prs:
            for si, F in enumerate(scen):
                for lab, cue in ((1, sh), (0, sl)):
                    st = states(F, cue, adapter_on)
                    for L in LAYERS:
                        X_fin[L].append(st[L][0]); X_cue[L].append(st[L][1])
                    y.append(lab); ycontent.append(int(si < len(scen)//2))
    return ({L: np.array(v) for L, v in X_fin.items()},
            {L: np.array(v) for L, v in X_cue.items()},
            np.array(y), np.array(ycontent))

def probe_auc(Xtr, ytr, Xte, yte, seed=0):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, clf.decision_function(sc.transform(Xte))))

results = {}
for adapter_on in (False, True):
    tag = "adapter_ON" if adapter_on else "adapter_OFF"
    print(f"collecting activations, {tag} ...", flush=True)
    Xtr_f, Xtr_c, ytr, ctr = collect(train_phr, train_scen[:12], adapter_on)
    Xte_f, Xte_c, yte, cte = collect(eval_phr,  eval_scen[:12],  adapter_on)
    row = {}
    for L in LAYERS:
        a_f = probe_auc(Xtr_f[L], ytr, Xte_f[L], yte)
        a_c = probe_auc(Xtr_c[L], ytr, Xte_c[L], yte)
        a_content = probe_auc(Xtr_f[L], ctr, Xte_f[L], cte)
        rs = np.random.default_rng(L)
        null = [probe_auc(Xtr_f[L], rs.permutation(ytr), Xte_f[L], yte) for _ in range(8)]
        row[L] = dict(auc_final_pos=a_f, auc_cue_pos=a_c, auc_content_control=a_content,
                      auc_shuffled_null=float(np.mean(null)))
        print(f"   L{L:<3} style(final) {a_f:.3f}   style(cue) {a_c:.3f}   "
              f"content {a_content:.3f}   shuffled {np.mean(null):.3f}", flush=True)
    results[tag] = row

print("\n"+"="*76); print("ERASED, OR ONLY SUPPRESSED?"); print("="*76)
print(f"{'layer':>6} {'OFF final':>11} {'ON final':>10} {'delta':>8}   {'OFF cue':>9} {'ON cue':>8} {'delta':>8}")
print("-"*70)
drops_f, drops_c = [], []
for L in LAYERS:
    o, n = results["adapter_OFF"][L]["auc_final_pos"], results["adapter_ON"][L]["auc_final_pos"]
    oc, nc = results["adapter_OFF"][L]["auc_cue_pos"], results["adapter_ON"][L]["auc_cue_pos"]
    drops_f.append(o-n); drops_c.append(oc-nc)
    print(f"{L:>6} {o:>11.3f} {n:>10.3f} {o-n:>+8.3f}   {oc:>9.3f} {nc:>8.3f} {oc-nc:>+8.3f}")
mean_drop_f, mean_drop_c = float(np.mean(drops_f)), float(np.mean(drops_c))
content_delta = float(np.mean([results["adapter_OFF"][L]["auc_content_control"] -
                               results["adapter_ON"][L]["auc_content_control"] for L in LAYERS]))
print("-"*70)
print(f"mean AUC drop, final position : {mean_drop_f:+.3f}")
print(f"mean AUC drop, cue positions  : {mean_drop_c:+.3f}")
print(f"content-control AUC drop      : {content_delta:+.3f}   (should be ~0)")

if mean_drop_f < 0.05 and mean_drop_c < 0.05:
    verdict = ("SUPPRESSION, NOT ERASURE. Style stays linearly decodable after the "
               "intervention. Report it: behaviour changed, representation did not. "
               "Cite Lee et al. 2024 and Galichin et al. 2026 as the same pattern, and "
               "Ponkshe et al. 2026 for why non-erasure may be correct rather than a defect.")
elif mean_drop_c > 0.10:
    verdict = ("PARTIAL ERASURE. Style is measurably harder to decode after the "
               "intervention, especially at the cue positions. This is a stronger claim "
               "than the paper currently makes — state it, with the permutation null.")
else:
    verdict = ("MIXED: the final position moved but the cue positions did not, or vice "
               "versa. Report both readouts; do not average them into one number.")
print(f"\nVERDICT: {verdict}")
if content_delta > 0.10:
    print("\n  WARNING: the content control also dropped. The adapter may have degraded "
          "the representation globally, which would invalidate a targeted-erasure reading.")

json.dump(dict(layers=LAYERS, results=results,
               mean_auc_drop_final=mean_drop_f, mean_auc_drop_cue=mean_drop_c,
               content_control_drop=content_delta, verdict=verdict),
          open("exp_D_probe.json","w"), indent=2)
print("\nsaved exp_D_probe.json")
