# ========================= PASTE INTO ONE COLAB CELL ===========================
# RECONCILE THE POSITIVE CONTROL.  ~4 min. Output: positive_control.json
#
# THE PROBLEM
#   Two of your scripts compute "replace the entire clinical history" differently,
#   and they disagree: v6 got 0.147 bits, EXP_E got 0.235.
#
#   v6      JS( [cue + placebo + facts_i] , [cue + placebo + facts_j] )
#           the style/placebo PREFIX IS PRESENT in both prompts, exactly as it is
#           in the style measurement, and only the facts change
#   EXP_E   JS( [facts_i] , [facts_j] )
#           bare facts, no prefix at all -- a different prompt distribution
#   v6 also drew from the first 10 scenarios of the full list (which includes
#   TRAINING scenarios); EXP_E used held-out eval scenarios.
#
#   Two things differ at once, so neither number can be corrected into the other.
#
# WHICH CONVENTION IS RIGHT
#   v6's. The whole point of the denominator is to answer "how big is the style
#   effect compared to changing the medicine, measured the same way". D_style is
#   computed WITH the prefix present, so the positive control must be too.
#   EXP_E's bare-facts version measures a different contrast and reads larger,
#   which makes every percentage-of-control look smaller than it should.
#
# WHAT THIS DOES
#   Computes both conventions on ONE scenario set (held-out), so you can quote a
#   single canonical number everywhere and footnote the other. It also reports
#   the number on training scenarios so you can see how much of the 0.147/0.235
#   gap was convention and how much was scenario choice.
#
# WHAT TO DO WITH THE OUTPUT
#   Use PREFIXED / HELD-OUT as the canonical positive control in the paper.
#   Recompute every "% of positive control" and every drift multiple against it.
#   State the definition in Methods in one sentence -- a denominator nobody can
#   reconstruct is worse than no denominator.
# ==============================================================================
import os, sys, json, subprocess, random
import numpy as np

# Pure inference: ~40 generations and ~80 forward passes total. Runs on CPU in
# roughly 15-30 min, on a T4 in ~4. Model is fp32 either way, so the numbers are
# identical -- this cell defines a denominator, so it must not vary by device.
import torch as _t
DEVICE = "cuda" if _t.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    _t.set_num_threads(max(1, os.cpu_count() or 2))
    print("  no GPU -> running on CPU (slower, identical numbers)", flush=True)

subprocess.run([sys.executable,"-m","pip","install","-q","transformers","accelerate"],check=False)
if not os.path.isdir("DiaLense"):
    subprocess.run(["git","clone","--depth","1","-q","https://github.com/SamhitaK10/DiaLense.git"],check=True)

import torch, glob
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "You are a physician taking a patient history. Ask one follow-up question."
CONT_LEN, N_TRAIN_SCEN, N_EVAL_SCEN, N_PAIRS = 24, 30, 20, 10

# the exact prefix v6 used for its positive control: phrasing 0 of fluency,
# plus its matched placebo sentence
CUE     = "The patient speaks fluent, grammatical English as a first language."
PLACEBO = "The patient arrived by bus on a cloudy morning."

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
first10    = facts[:10]                      # what v6 actually used

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval().to(DEVICE)
print(f"model on {DEVICE}", flush=True)

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
def js(P, Q):
    M = 0.5*(P+Q)
    kl = lambda A,B: (A*(torch.log2(A.clamp_min(1e-12))-torch.log2(B.clamp_min(1e-12)))).sum(-1)
    return float((0.5*kl(P,M)+0.5*kl(Q,M)).mean())

def positive_control(scen, prefixed):
    """Swap the entire clinical history, holding the reference continuation fixed."""
    n = min(N_PAIRS, len(scen))
    refs = [reference(F) for F in scen[:n]]
    mk = (lambda F: f"{CUE} {PLACEBO} {F}") if prefixed else (lambda F: F)
    return float(np.mean([js(scored(mk(scen[i]), refs[i]),
                             scored(mk(scen[(i+1) % n]), refs[i])) for i in range(n)]))

print("computing four variants...\n", flush=True)
res = {
 "prefixed_heldout":  positive_control(eval_scen,  True),
 "bare_heldout":      positive_control(eval_scen,  False),
 "prefixed_train":    positive_control(train_scen, True),
 "prefixed_first10":  positive_control(first10,    True),
}
print("="*70); print("POSITIVE CONTROL — full clinical-history swap, bits"); print("="*70)
print(f"  PREFIXED, held-out scenarios   {res['prefixed_heldout']:.5f}   <-- USE THIS ONE")
print(f"  bare facts, held-out scenarios {res['bare_heldout']:.5f}   (EXP_E's convention)")
print(f"  prefixed, training scenarios   {res['prefixed_train']:.5f}")
print(f"  prefixed, first 10 scenarios   {res['prefixed_first10']:.5f}   (what v6 reported as 0.147)")
print("-"*70)
print(f"  convention effect (prefixed vs bare, same scenarios) : "
      f"{res['bare_heldout'] - res['prefixed_heldout']:+.5f} bits")
print(f"  scenario effect (held-out vs first10, both prefixed) : "
      f"{res['prefixed_heldout'] - res['prefixed_first10']:+.5f} bits")

CANON = res["prefixed_heldout"]
print(f"\nCANONICAL POSITIVE CONTROL = {CANON:.5f} bits")
print("\nRESCALE THESE NUMBERS BEFORE THEY GO IN THE PAPER:")
for label, val in [("style effect, communication_style", 0.00888),
                   ("style effect, health_literacy",     0.00604),
                   ("style effect, confidence",          0.00452),
                   ("hard-target neutral drift",         0.302),
                   ("soft-target neutral drift",         0.0026)]:
    print(f"   {label:36s} {val:.5f} bits = {100*val/CANON:6.2f}% of control"
          f"   ({val/CANON:.2f}x)")
print("""
METHODS SENTENCE TO USE
  "The positive control replaces the entire clinical history while holding the
   style and placebo sentences and the reference continuation fixed, averaged
   over held-out scenarios; it is X bits. All divergences are additionally
   reported as a percentage of it, because divergence in bits has no natural
   scale and a null result cannot otherwise be distinguished from an
   insensitive instrument."

NOTE ON THE HARD-TARGET CLAIM
  Whichever denominator you pick, hard-target drift (0.302 bits) still exceeds a
  full content swap. The claim survives; only the multiple changes. Say the
  multiple, and say which denominator produced it.
""")
json.dump(dict(variants=res, canonical=CANON,
               definition="JS over a fixed 24-token reference continuation between "
                          "two prompts sharing an identical style+placebo prefix and "
                          "differing only in the clinical history; mean over 10 "
                          "held-out scenario pairs"),
          open("positive_control.json","w"), indent=2)
print("saved positive_control.json")
