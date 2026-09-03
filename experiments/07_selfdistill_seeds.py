# ========================= PASTE INTO ONE COLAB CELL ===========================
# EXPERIMENT A — seed variance on the headline number.
# Runtime ~30 min PER SEED on a T4. Run it twice (SEED=1, then SEED=2).
# Output: exp_A_seed<N>.json, then run the AGGREGATE cell at the bottom.
#
# WHY
#   You currently report "91.6%" from one run. ARR's checklist item C3 explicitly
#   permits a single run if you are transparent about it, and reviewer heuristic
#   H13 bars "run more experiments" as a stated reason to reject — so this is not
#   a desk-reject. But R5 (missing significance assessment) is a listed legitimate
#   weakness, and a one-decimal percentage from n=1 invites it. Dodge et al. (2020)
#   is the citation a hostile reviewer reaches for: fine-tuning outcomes vary
#   substantially across seeds, and best-of-N reporting inflates results.
#   Either run this twice, or stop writing 91.6%.
#
# WHAT VARIES WITH THE SEED, AND WHAT MUST NOT
#   varies : LoRA init, data order, dropout mask
#   fixed  : the held-out split (scenarios AND phrasings), the teacher targets,
#            the reference continuations, every hyperparameter
#   The split is deliberately seeded with random.Random(0) regardless of SEED.
#   If the split moved with the seed you would be measuring split variance, not
#   initialisation variance, and the three numbers would not be comparable.
# ==============================================================================
SEED = 1        # <<<<<< CHANGE TO 2 FOR THE SECOND RUN, 3 FOR A THIRD

import os, sys, json, subprocess, random, gc
import numpy as np

try:
    import torch as _t
    if not _t.cuda.is_available():
        raise SystemExit("\n  NO GPU. Runtime > Change runtime type > T4 GPU > Save, then rerun.")
except ImportError: pass

print(f"installing deps... (SEED={SEED})", flush=True)
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
from peft import LoraConfig, get_peft_model

import torch, glob
import torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "You are a physician taking a patient history. Ask one follow-up question."
TARGET_LEN, N_TRAIN_SCEN, N_EVAL_SCEN, N_TRAIN_PHR = 24, 30, 20, 5
EPOCHS, LR, ACCUM, DRIFT_STOP = 2, 3e-5, 8, 0.05

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

def keep(name):
    """Copy a result next to the other outputs; no-op if it is already there."""
    src = os.path.abspath(name); dst = os.path.join(DRIVE, os.path.basename(name))
    if src == os.path.abspath(dst): return
    subprocess.run(["cp","-r",src,DRIVE+"/"],check=False)

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

facts, doctor_turns, seen = [], [], set()
for p in sorted(glob.glob("DiaLense/results/raw/*/conversations.jsonl")):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r["scenario_id"] in seen: continue
        seen.add(r["scenario_id"])
        facts.append("Reported history: " + "; ".join(
            str(v).strip().rstrip(".") for v in r["latent_facts"].values()) + ".")
        for t in r["dialogue_turns"]:
            if t["speaker"] == "Doctor" and len(t["text"].split()) > 6:
                doctor_turns.append(t["text"]); break

# SPLIT IS FIXED ACROSS SEEDS — this is the whole point
split_rng = random.Random(0)
idx = list(range(len(facts))); split_rng.shuffle(idx)
train_scen = [facts[i] for i in idx[:N_TRAIN_SCEN]]
eval_scen  = [facts[i] for i in idx[N_TRAIN_SCEN:N_TRAIN_SCEN+N_EVAL_SCEN]]
train_phr  = {d: v[:N_TRAIN_PHR] for d, v in CUES.items()}
eval_phr   = {d: v[N_TRAIN_PHR:] for d, v in CUES.items()}
print(f"split (FIXED): train {len(train_scen)} scen x {N_TRAIN_PHR} phr | "
      f"held out {len(eval_scen)} scen x {8-N_TRAIN_PHR} phr", flush=True)

# SEED VARIES ONLY THE RUN
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED); run_rng = random.Random(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
model.config.use_cache = False
def chat(t):
    return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":t}],
                                   tokenize=False, add_generation_prompt=True)
@torch.no_grad()
def neutral_answer(F):
    enc = tok(chat(F), return_tensors="pt").to("cuda")
    out = model.generate(**enc, max_new_tokens=TARGET_LEN, do_sample=False, pad_token_id=tok.eos_token_id)
    return out[0, enc.input_ids.shape[-1]:]
@torch.no_grad()
def dist_over(text, cont):
    pid = tok(chat(text), return_tensors="pt").input_ids.cuda()
    ids = torch.cat([pid, cont.unsqueeze(0)], 1)
    return torch.softmax(model(ids).logits[0, pid.shape[-1]-1:-1].float(), -1)
def js(P, Q):
    M = 0.5*(P+Q)
    kl = lambda A,B: (A*(torch.log2(A.clamp_min(1e-12))-torch.log2(B.clamp_min(1e-12)))).sum(-1)
    return float((0.5*kl(P,M)+0.5*kl(Q,M)).mean())
@torch.no_grad()
def doctor_perplexity(n=40):
    tot, cnt = 0.0, 0
    for txt in doctor_turns[:n]:
        ids = tok(txt, return_tensors="pt").input_ids.cuda()
        if ids.shape[-1] < 4: continue
        tot += float(Fn.cross_entropy(model(ids).logits[0, :-1].float(), ids[0, 1:])); cnt += 1
    return float(np.exp(tot/max(cnt,1)))
def style_gap_heldout():
    return {dim: float(np.mean([js(dist_over(f"{sh} {F}", t), dist_over(f"{sl} {F}", t))
                                for sh, sl in prs for F, t in zip(eval_scen, eval_targets)]))
            for dim, prs in eval_phr.items()}

train_targets = [neutral_answer(F) for F in train_scen]
eval_targets  = [neutral_answer(F) for F in eval_scen]
teachers = [dist_over(F, t).half().cpu() for F, t in zip(train_scen, train_targets)]
teacher_entropy = float(np.mean([float(-(t.float()*t.float().clamp_min(1e-9).log2()).sum(-1).mean())
                                 for t in teachers]))
before_gap = style_gap_heldout(); before_ppl = doctor_perplexity()
before_neutral = [dist_over(F, t) for F, t in zip(eval_scen[:8], eval_targets[:8])]
print(f"BEFORE  mean style gap {np.mean(list(before_gap.values())):.5f} | ppl {before_ppl:.2f} "
      f"| teacher entropy {teacher_entropy:.3f} bits\n", flush=True)

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lora)
model.enable_input_require_grads(); model.gradient_checkpointing_enable()
for p_ in model.parameters():
    if p_.requires_grad: p_.data = p_.data.float()
print(f"LoRA trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n", flush=True)

def distil_toward(text, cont, teacher):
    pid = tok(chat(text), return_tensors="pt").input_ids.cuda()
    ids = torch.cat([pid, cont.unsqueeze(0)], 1)
    logp = torch.log_softmax(model(ids).logits[0, pid.shape[-1]-1:-1].float(), -1)
    T = teacher.cuda().float()
    return (T * (T.clamp_min(1e-9).log() - logp)).sum(-1).mean()

examples = [(sh, sl, F, t, th) for dim, prs in train_phr.items() for sh, sl in prs
            for F, t, th in zip(train_scen, train_targets, teachers)]
run_rng.shuffle(examples)                        # data order varies with SEED
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
scaler = torch.amp.GradScaler("cuda"); model.train(); step = 0; stop = False
for ep in range(EPOCHS):
    if stop: break
    running = []
    for i, (sh, sl, F, t, th) in enumerate(examples):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = 0.5*(distil_toward(f"{sh} {F}", t, th) + distil_toward(f"{sl} {F}", t, th))
        scaler.scale(loss/ACCUM).backward(); running.append(float(loss))
        if (i+1) % ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); step += 1
            if step % 40 == 0:
                model.save_pretrained(f"dialense_lora_seed{SEED}"); keep(f"dialense_lora_seed{SEED}")
            if step % 20 == 0:
                model.eval()
                with torch.no_grad():
                    d = float(np.mean([js(dist_over(F, t), b) for F, t, b in
                                       zip(eval_scen[:4], eval_targets[:4], before_neutral[:4])]))
                model.train()
                print(f"   ep{ep+1} step{step} loss {np.mean(running[-ACCUM*20:]):.4f} drift {d:.4f}", flush=True)
                if d > DRIFT_STOP:
                    print(f"   STOPPING: drift {d:.4f} > {DRIFT_STOP}"); stop = True; break
    print(f"  epoch {ep+1} mean loss {np.mean(running):.4f}", flush=True)

model.eval(); model.config.use_cache = False
model.save_pretrained(f"dialense_lora_seed{SEED}"); keep(f"dialense_lora_seed{SEED}")
after_gap = style_gap_heldout(); after_ppl = doctor_perplexity()
after_neutral = [dist_over(F, t) for F, t in zip(eval_scen[:8], eval_targets[:8])]
drift = float(np.mean([js(a, b) for a, b in zip(before_neutral, after_neutral)]))

mb, ma = np.mean(list(before_gap.values())), np.mean(list(after_gap.values()))
print("\n"+"="*70); print(f"SEED {SEED} RESULT  (held-out phrasings x held-out scenarios)"); print("="*70)
print(f"{'dimension':28s} {'before':>10} {'after':>10} {'change':>10}")
for d in sorted(before_gap):
    print(f"{d:28s} {before_gap[d]:>10.5f} {after_gap[d]:>10.5f} "
          f"{100*(after_gap[d]-before_gap[d])/before_gap[d]:>9.1f}%")
print("-"*60)
print(f"{'MEAN':28s} {mb:>10.5f} {ma:>10.5f} {100*(ma-mb)/mb:>9.1f}%")
print(f"\nperplexity {before_ppl:.2f} -> {after_ppl:.2f} | neutral drift {drift:.5f} bits")

out = dict(seed=SEED, before=before_gap, after=after_gap,
           reduction_pct=float(100*(ma-mb)/mb),
           per_dim_reduction_pct={d: float(100*(after_gap[d]-before_gap[d])/before_gap[d])
                                  for d in before_gap},
           before_ppl=before_ppl, after_ppl=after_ppl, neutral_drift=drift,
           teacher_entropy_bits=teacher_entropy,
           config=dict(lr=LR, epochs=EPOCHS, r=16, split_seed=0, run_seed=SEED))
json.dump(out, open(f"exp_A_seed{SEED}.json","w"), indent=2); keep(f"exp_A_seed{SEED}.json")
print(f"\nsaved exp_A_seed{SEED}.json"); print(json.dumps(out, indent=2))

# ============================ AGGREGATE CELL ==================================
# Run this ONLY after all seeds are done. Paste as its own cell.
AGG = r'''
import json, glob, numpy as np
files = sorted(sum([glob.glob(p) for p in ["exp_A_seed*.json", "DiaLense_PartII/exp_A_seed*.json",
        "/content/drive/MyDrive/DiaLense_PartII/exp_A_seed*.json",
        "/kaggle/working/DiaLense_PartII/exp_A_seed*.json", "/kaggle/input/*/exp_A_seed*.json"]], []))
runs, seen = [], set()
for f in files:
    r = json.load(open(f))
    if r["seed"] in seen: continue
    seen.add(r["seed"]); runs.append(r)
runs.sort(key=lambda r: r["seed"])
red = np.array([r["reduction_pct"] for r in runs])
print(f"seeds: {[r['seed'] for r in runs]}")
for r in runs: print(f"  seed {r['seed']}: {r['reduction_pct']:+.1f}%  drift {r['neutral_drift']:.5f}")
print(f"\nMEAN {red.mean():+.1f}%   RANGE {red.min():+.1f}% .. {red.max():+.1f}%   SPREAD {red.max()-red.min():.1f} pts")
print("\nREPORT IT AS:  'mean {:.1f}%, range {:.1f}-{:.1f} across {} seeds'".format(
      abs(red.mean()), abs(red.max()), abs(red.min()), len(runs)))
print("Do NOT pool seeds into a confidence interval at n=3 — the range is the honest summary.")
dims = sorted(runs[0]["per_dim_reduction_pct"])
print(f"\n{'dimension':28s} " + " ".join(f"seed{r['seed']:>6}" for r in runs) + f"{'range':>9}")
for d in dims:
    v = [r["per_dim_reduction_pct"][d] for r in runs]
    print(f"{d:28s} " + " ".join(f"{x:>10.1f}" for x in v) + f"{max(v)-min(v):>9.1f}")
spread = red.max()-red.min()
print("\nVERDICT:", "STABLE — report mean +/- range and the criticism is dead."
      if spread < 8 else
      "UNSTABLE — restate the claim as a range, not a point. That is still publishable, "
      "and hiding it is not.")
'''
print("\n\n# ---- paste this as a separate cell once all seeds are done ----")
print(AGG)
