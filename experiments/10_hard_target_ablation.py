# ========================= PASTE INTO ONE COLAB CELL ===========================
# EXPERIMENT E — the objective comparison, made quantitative.
# Runtime ~35 min on a T4 (retrains the hard-target arm).
# Output: exp_E_objective_comparison.json  -> Table 3 and Figure 3 of the paper.
#
# WHY
#   Right now the hard-target failure is a story: "loss hit 1e-4, the distribution
#   collapsed." The number that makes it evidence is ENTROPY of the student's
#   output distribution. Forward KL is mass-covering, so the teacher's entropy is
#   a floor for the soft-target student; cross-entropy against a one-hot target
#   has no floor at all. That is the whole mechanism, and it is currently unmeasured.
#
#   Supporting literature, all pointing at this measurement:
#     Guo et al. ICML 2017      NLL keeps minimising long after accuracy saturates
#     Müller et al. NeurIPS 19  one-hot targets destroy relational structure
#     Huang et al. ICML 2025    instruction tuning degrades calibration; label
#                               smoothing scales badly with vocabulary size, which
#                               is the argument for a real teacher distribution
#     Cui et al. 2025           entropy collapse as a named failure mode
#
# WHAT IT MEASURES, for base / teacher / soft-target student / hard-target student
#   * mean output entropy in bits over the reference positions
#   * top-1 probability (the blunt version of the same thing)
#   * held-out style reduction
#   * neutral-prompt drift, against the 0.147-bit positive control as the yardstick
#   * count of near-deterministic positions (p_max > 0.99) — the memorisation tell
#
# IT RETRAINS THE HARD-TARGET ARM. If you still have that checkpoint, set
# HARD_ADAPTER and the script will load instead of retraining.
#
# CHECKPOINTED. The hard-target loop saves to <HARD_ADAPTER>_ckpt every 40 steps
# and at each epoch end, with progress in hard_progress.json on Drive. If the
# runtime dies you lose at most ~40 steps: re-run the cell and it fast-forwards
# past the completed examples. Caveat, and it belongs in the appendix if you use
# a resumed run: optimizer state is NOT restored, so a resumed run is not
# bit-identical to an uninterrupted one. For an ablation whose whole point is
# "cross-entropy on one-hot targets drives confidence to 1", that is immaterial —
# but say it rather than let a reviewer wonder.
# ==============================================================================
import os, sys, json, subprocess, random, gc
import numpy as np

try:
    import torch as _t
    if not _t.cuda.is_available():
        raise SystemExit("\n  NO GPU. Runtime > Change runtime type > T4 GPU > Save, then rerun.")
except ImportError: pass

print("installing deps...", flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","transformers","accelerate","peft"],check=False)
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
from peft import LoraConfig, get_peft_model, PeftModel

import torch, glob
import torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM  = "You are a physician taking a patient history. Ask one follow-up question."
TARGET_LEN, N_TRAIN_SCEN, N_EVAL_SCEN, N_TRAIN_PHR = 24, 30, 20, 5
EPOCHS, LR, ACCUM = 2, 3e-5, 8

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

SOFT_ADAPTER = _find_adapter("dialense_lora")
if SOFT_ADAPTER is None:
    raise SystemExit(
        "\n  dialense_lora not found. Looked in the output dir, Google Drive, the\n"
        "  working directory, and /kaggle/input/*/.\n"
        "  On Kaggle: upload it as a Dataset and Add Input it. See KAGGLE.md.")
HARD_ADAPTER = os.path.join(DRIVE, "dialense_lora_hard")   # written here
print(f"soft adapter -> {SOFT_ADAPTER}\nhard adapter -> {HARD_ADAPTER}", flush=True)

CUES = {  # first 5 per dim are TRAIN, last 3 are HELD OUT — must match training
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

facts, seen = [], set()
for p in sorted(glob.glob("DiaLense/results/raw/*/conversations.jsonl")):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r["scenario_id"] in seen: continue
        seen.add(r["scenario_id"])
        facts.append("Reported history: " + "; ".join(
            str(v).strip().rstrip(".") for v in r["latent_facts"].values()) + ".")
rng = random.Random(0)
idx = list(range(len(facts))); rng.shuffle(idx)
train_scen = [facts[i] for i in idx[:N_TRAIN_SCEN]]
eval_scen  = [facts[i] for i in idx[N_TRAIN_SCEN:N_TRAIN_SCEN+N_EVAL_SCEN]]
train_phr  = {d: v[:N_TRAIN_PHR] for d, v in CUES.items()}
eval_phr   = {d: v[N_TRAIN_PHR:] for d, v in CUES.items()}
print(f"train {len(train_scen)} scen x {N_TRAIN_PHR} phr | held out {len(eval_scen)} scen x {8-N_TRAIN_PHR} phr\n", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
def chat(t):
    return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":t}],
                                   tokenize=False, add_generation_prompt=True)

def load_base():
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    m.config.use_cache = False
    return m

@torch.no_grad()
def dist_over(m, text, cont):
    pid = tok(chat(text), return_tensors="pt").input_ids.cuda()
    ids = torch.cat([pid, cont.unsqueeze(0)], 1)
    return torch.softmax(m(ids).logits[0, pid.shape[-1]-1:-1].float(), -1)

def entropy_bits(P):
    return float(-(P * P.clamp_min(1e-12).log2()).sum(-1).mean())
def top1(P):  return float(P.max(-1).values.mean())
def frac_deterministic(P, thr=0.99): return float((P.max(-1).values > thr).float().mean())
def js(P, Q):
    M = 0.5*(P+Q)
    kl = lambda A,B: (A*(torch.log2(A.clamp_min(1e-12))-torch.log2(B.clamp_min(1e-12)))).sum(-1)
    return float((0.5*kl(P,M)+0.5*kl(Q,M)).mean())

print("computing reference continuations and teacher targets...", flush=True)
base = load_base()
@torch.no_grad()
def neutral_answer(F):
    enc = tok(chat(F), return_tensors="pt").to("cuda")
    out = base.generate(**enc, max_new_tokens=TARGET_LEN, do_sample=False, pad_token_id=tok.eos_token_id)
    return out[0, enc.input_ids.shape[-1]:]
train_targets = [neutral_answer(F) for F in train_scen]
eval_targets  = [neutral_answer(F) for F in eval_scen]
teachers = [dist_over(base, F, t).half().cpu() for F, t in zip(train_scen, train_targets)]

POSITIVE_CONTROL = float(np.mean([
    js(dist_over(base, eval_scen[i], eval_targets[i]),
       dist_over(base, eval_scen[(i+1) % len(eval_scen)], eval_targets[i]))
    for i in range(10)]))
print(f"  positive control (full content swap): {POSITIVE_CONTROL:.5f} bits  <- the yardstick\n", flush=True)

def measure(m, label):
    ent, t1, det = [], [], []
    for F, t in zip(eval_scen, eval_targets):
        P = dist_over(m, F, t); ent.append(entropy_bits(P)); t1.append(top1(P)); det.append(frac_deterministic(P))
    gaps = {}
    for dim, prs in eval_phr.items():
        v = [js(dist_over(m, f"{sh} {F}", t), dist_over(m, f"{sl} {F}", t))
             for sh, sl in prs for F, t in zip(eval_scen, eval_targets)]
        gaps[dim] = float(np.mean(v))
    return dict(label=label, entropy_bits=float(np.mean(ent)), top1_prob=float(np.mean(t1)),
                frac_positions_p_gt_0p99=float(np.mean(det)),
                style_gap=gaps, style_gap_mean=float(np.mean(list(gaps.values()))))

print("measuring BASE ...", flush=True)
M_base = measure(base, "base")
base_neutral = [dist_over(base, F, t) for F, t in zip(eval_scen[:8], eval_targets[:8])]
teacher_entropy = float(np.mean([entropy_bits(t.float()) for t in teachers]))
del base; gc.collect(); torch.cuda.empty_cache()

def drift_vs_base(m):
    return float(np.mean([js(dist_over(m, F, t), b)
                          for F, t, b in zip(eval_scen[:8], eval_targets[:8], base_neutral)]))

print("measuring SOFT-TARGET student ...", flush=True)
b2 = load_base(); soft = PeftModel.from_pretrained(b2, SOFT_ADAPTER).eval()
M_soft = measure(soft, "soft_target_KL"); M_soft["neutral_drift"] = drift_vs_base(soft)
del soft, b2; gc.collect(); torch.cuda.empty_cache()

# ---------------- hard-target arm: load, else retrain -------------------------
CKPT      = HARD_ADAPTER + "_ckpt"
PROGRESS  = os.path.join(DRIVE, "hard_progress.json")
def _prog():
    try: return json.load(open(PROGRESS))
    except Exception: return {"done": False, "examples_seen": 0}

if not os.path.isdir(HARD_ADAPTER):
    pr = _prog()
    resume_from = pr["examples_seen"] if (not pr["done"] and os.path.isdir(CKPT)) else 0
    if resume_from:
        print(f"\nresuming hard-target training from example {resume_from} "
              f"(optimizer state is NOT restored — see note below)", flush=True)
    else:
        print("\nhard-target checkpoint absent — training it (this is the ablation)...", flush=True)
    m = load_base()
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    if resume_from:
        m = PeftModel.from_pretrained(m, CKPT, is_trainable=True)
    else:
        m = get_peft_model(m, lora)
    m.enable_input_require_grads(); m.gradient_checkpointing_enable()
    for p_ in m.parameters():
        if p_.requires_grad: p_.data = p_.data.float()
    def hard_ce(text, cont):
        """Cross-entropy against the reference TOKEN STRING. Minimised at unit
        confidence, so optimisation has no stopping point short of memorisation."""
        pid = tok(chat(text), return_tensors="pt").input_ids.cuda()
        ids = torch.cat([pid, cont.unsqueeze(0)], 1)
        logits = m(ids).logits[0, pid.shape[-1]-1:-1].float()
        return Fn.cross_entropy(logits, cont)
    ex = [(sh, sl, F, t) for dim, prs in train_phr.items() for sh, sl in prs
          for F, t in zip(train_scen, train_targets)]
    random.Random(0).shuffle(ex)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)
    scaler = torch.amp.GradScaler("cuda"); m.train(); step = 0
    seen_total, run = resume_from, []
    TOTAL = EPOCHS * len(ex)
    for ep in range(EPOCHS):
        for i, (sh, sl, F, t) in enumerate(ex):
            pos = ep*len(ex) + i
            if pos < resume_from: continue              # fast-forward past done work
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = 0.5*(hard_ce(f"{sh} {F}", t) + hard_ce(f"{sl} {F}", t))
            scaler.scale(loss/ACCUM).backward(); run.append(float(loss.detach()))
            seen_total = pos + 1
            if (pos+1) % ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); step += 1
                if step % 20 == 0:
                    print(f"   ep{ep+1} {seen_total}/{TOTAL} step{step} "
                          f"loss {np.mean(run[-ACCUM*20:]):.6f}", flush=True)
                if step % 40 == 0:                      # CHECKPOINT — survive a disconnect
                    m.save_pretrained(CKPT)
                    json.dump({"done": False, "examples_seen": seen_total,
                               "recent_loss": float(np.mean(run[-40:]))}, open(PROGRESS, "w"))
                    print(f"   [checkpoint at {seen_total}/{TOTAL}]", flush=True)
        print(f"  epoch {ep+1} done | mean loss {np.mean(run) if run else float('nan'):.6f}", flush=True)
        m.save_pretrained(CKPT)
        json.dump({"done": False, "examples_seen": seen_total,
                   "recent_loss": float(np.mean(run[-40:])) if run else None}, open(PROGRESS, "w"))
    final_hard_loss = float(np.mean(run[-40:])) if run else _prog().get("recent_loss")
    m.eval(); m.save_pretrained(HARD_ADAPTER)
    json.dump({"done": True, "examples_seen": seen_total,
               "recent_loss": final_hard_loss}, open(PROGRESS, "w"))
    print(f"  saved {HARD_ADAPTER}   final loss {final_hard_loss:.2e}", flush=True)
    hard = m
else:
    print(f"\nloading existing hard-target adapter from {HARD_ADAPTER}", flush=True)
    final_hard_loss = None
    b3 = load_base(); hard = PeftModel.from_pretrained(b3, HARD_ADAPTER).eval()

print("measuring HARD-TARGET student ...", flush=True)
M_hard = measure(hard, "hard_target_CE"); M_hard["neutral_drift"] = drift_vs_base(hard)
M_hard["final_train_loss"] = final_hard_loss

# ================================ REPORT ======================================
def red(m): return 100*(m["style_gap_mean"] - M_base["style_gap_mean"])/M_base["style_gap_mean"]
print("\n"+"="*78); print("TABLE 3 — OBJECTIVE COMPARISON"); print("="*78)
print(f"{'':22s} {'entropy':>9} {'top-1 p':>9} {'p>.99':>7} {'style gap':>11} {'reduction':>10} {'drift':>9}")
print("-"*78)
print(f"{'base model':22s} {M_base['entropy_bits']:>9.3f} {M_base['top1_prob']:>9.3f} "
      f"{M_base['frac_positions_p_gt_0p99']:>7.3f} {M_base['style_gap_mean']:>11.5f} {'—':>10} {'—':>9}")
print(f"{'teacher (blind base)':22s} {teacher_entropy:>9.3f} {'—':>9} {'—':>7} {'—':>11} {'—':>10} {'—':>9}")
print(f"{'soft-target KL':22s} {M_soft['entropy_bits']:>9.3f} {M_soft['top1_prob']:>9.3f} "
      f"{M_soft['frac_positions_p_gt_0p99']:>7.3f} {M_soft['style_gap_mean']:>11.5f} "
      f"{red(M_soft):>9.1f}% {M_soft['neutral_drift']:>9.4f}")
print(f"{'hard-target CE':22s} {M_hard['entropy_bits']:>9.3f} {M_hard['top1_prob']:>9.3f} "
      f"{M_hard['frac_positions_p_gt_0p99']:>7.3f} {M_hard['style_gap_mean']:>11.5f} "
      f"{red(M_hard):>9.1f}% {M_hard['neutral_drift']:>9.4f}")
print("-"*78)
print(f"positive control (full content swap) = {POSITIVE_CONTROL:.5f} bits")
print(f"hard-target drift as a multiple of it: {M_hard['neutral_drift']/POSITIVE_CONTROL:.2f}x")
print(f"soft-target drift as a multiple of it: {M_soft['neutral_drift']/POSITIVE_CONTROL:.2f}x")

print("""
WHAT THIS BUYS YOU
  The soft-target student's entropy should sit at or above the TEACHER's, because
  forward KL is mass-covering: the teacher's entropy is a floor. The hard-target
  student's should sit far below the base model's, with a high fraction of
  near-deterministic positions. That pair of numbers is the mechanism, and it
  replaces the anecdote about training loss.
  Figure 3(b) is: neutral drift for both arms, with the positive control drawn as
  a horizontal line. If hard-target drift exceeds it, you have a fairness metric
  improving while the model degrades, in one frame.
""")
json.dump(dict(positive_control=POSITIVE_CONTROL, teacher_entropy_bits=teacher_entropy,
               base=M_base, soft_target=M_soft, hard_target=M_hard,
               reduction_soft_pct=red(M_soft), reduction_hard_pct=red(M_hard)),
          open("exp_E_objective_comparison.json","w"), indent=2)
print("saved exp_E_objective_comparison.json")
