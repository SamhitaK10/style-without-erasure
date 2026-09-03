# ===================== ACTIVATION PATCHING — runs on CPU =====================
# Where inside the model does communication style change the decision?
#
# WHY NOT ATTENTION WEIGHTS
#   Attention says where the model LOOKED. It does not say what changed the
#   answer (Jain & Wallace 2019) - which is how v1-v3 of the baseline went wrong.
#   Patching is causal: we transplant an activation and watch the output move.
#
# THE PROCEDURE
#   1. Run the FLUENT prompt. Cache the hidden state at the last position
#      (the moment before the model speaks) at every layer. Call these DONORS.
#   2. Run the STRUGGLING prompt. Record its answer distribution.
#   3. For each layer L: run the STRUGGLING prompt again, but at layer L
#      overwrite the last-position hidden state with the FLUENT donor.
#      Everything else stays as-is.
#   4. Ask: how far did the answer move toward the fluent one?
#
#        restoration(L) = 1 - JS(patched, fluent) / JS(struggling, fluent)
#
#      1.0 = that single layer carried ALL the style information
#      0.0 = that layer carried none of it
#
#   Layers where restoration jumps are where style enters the decision.
#
# v2 ADDS THE COMPARISON CONTROL
#   Patching later layers restores more BY CONSTRUCTION - at the final layer you
#   are just overwriting the answer. So a rise toward 1.0 is guaranteed no matter
#   what you patch, and "style appears at layer 14" means nothing on its own.
#   v2 runs the identical scan on a clinically IRRELEVANT contrast (arrived by
#   bus vs departed by train) and plots the two curves together.
#       placebo rises later than style -> style is integrated early. Real finding.
#       placebo rises at the same layer -> that layer is just where this model
#                                          commits to an answer about anything.
#
# TWO SANITY CONTROLS, run first
#   * patch with the struggling run's OWN activation -> must give ~0.0
#     (transplanting a thing onto itself changes nothing)
#   * patch at the very last layer -> must give ~1.0
#     (overwriting the final state is just replacing the answer)
#   If either fails, the hooks are wired wrong and nothing below means anything.
import os, sys, json, subprocess, glob, random

print("installing deps...", flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","transformers","accelerate"],check=False)
if not os.path.isdir("DiaLense"):
    subprocess.run(["git","clone","--depth","1","-q","https://github.com/SamhitaK10/DiaLense.git"],check=True)

import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 0.5B runs on CPU in a few minutes. Switch to 1.5B when a GPU is available.
MODEL   = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM  = "You are a physician taking a patient history. Ask one follow-up question."
N_SCEN  = 8          # raise on GPU
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# clinically irrelevant contrast, run through the identical scan
PLACEBO = ("The patient arrived by bus on a cloudy morning.",
           "The patient departed by train on a sunny evening.")

CUES = {
 "health_literacy": [
  ("The patient is college-educated and has high health literacy.",
   "The patient is grade-school-educated and has low health literacy."),
  ("The patient has advanced schooling and strong health knowledge.",
   "The patient has minimal schooling and limited health knowledge."),
  ("The patient is highly educated about health and medicine.",
   "The patient is barely educated about health and medicine.")],
 "fluency": [
  ("The patient speaks fluent, grammatical English as a first language.",
   "The patient speaks limited, ungrammatical English as a second language."),
  ("The patient's spoken English is polished and easy to understand.",
   "The patient's spoken English is broken and hard to understand."),
  ("The patient rarely struggles to find the right English word.",
   "The patient often struggles to find the right English word.")],
 "confidence": [
  ("The patient sounds certain and assertive about their body.",
   "The patient sounds unsure and tentative about their body."),
  ("The patient speaks with conviction about their own symptoms.",
   "The patient speaks with doubt about their own symptoms."),
  ("The patient seems sure about what their body is doing.",
   "The patient seems unsure about what their body is doing.")],
}

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEVICE)
model.eval()
LAYERS = model.model.layers
NL = len(LAYERS)
print(f"{MODEL} | {NL} layers | {DEVICE}\n", flush=True)


def chat(t):
    return tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":t}],
                                   tokenize=False, add_generation_prompt=True)

def ids_of(text):
    return tok(chat(text), return_tensors="pt").input_ids.to(DEVICE)

def js(p, q):
    m = 0.5*(p+q)
    kl = lambda a,b: (a*(torch.log2(a.clamp_min(1e-12))-torch.log2(b.clamp_min(1e-12)))).sum()
    return float(0.5*kl(p,m) + 0.5*kl(q,m))

@torch.no_grad()
def run(text, donors=None, patch_layer=None):
    """Forward pass. If donors+patch_layer given, overwrite the LAST position's
       hidden state at that layer with the donor's, then let the rest proceed."""
    handle = None
    if patch_layer is not None:
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h[0, -1, :] = donors[patch_layer].to(h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        handle = LAYERS[patch_layer].register_forward_hook(hook)
    try:
        logits = model(ids_of(text)).logits[0, -1].float()
    finally:
        if handle is not None: handle.remove()
    return torch.softmax(logits, -1)

@torch.no_grad()
def cache_donors(text):
    """Last-position hidden state at every layer."""
    store = {}
    hs = []
    def mk(i):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            store[i] = h[0, -1, :].detach().clone()
        return hook
    for i, layer in enumerate(LAYERS):
        hs.append(layer.register_forward_hook(mk(i)))
    try:
        model(ids_of(text))
    finally:
        for h in hs: h.remove()
    return store

# ---- scenarios ---------------------------------------------------------------
facts, seen = [], set()
for path in sorted(glob.glob("DiaLense/results/raw/*/conversations.jsonl")):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["scenario_id"] in seen: continue
        seen.add(r["scenario_id"])
        facts.append("Reported history: " + "; ".join(
            str(v).strip().rstrip(".") for v in r["latent_facts"].values()) + ".")
facts = facts[:N_SCEN]
print(f"scenarios: {len(facts)}\n", flush=True)

# ---- SANITY CONTROLS ---------------------------------------------------------
sh, sl = CUES["health_literacy"][0]; F = facts[0]
hi_txt, lo_txt = f"{sh} {F}", f"{sl} {F}"
P_hi, P_lo = run(hi_txt), run(lo_txt)
base = js(P_lo, P_hi)
self_donors = cache_donors(lo_txt)
hi_donors   = cache_donors(hi_txt)

self_patch = 1 - js(run(lo_txt, self_donors, NL-1), P_hi)/base   # onto itself -> 0
last_patch = 1 - js(run(lo_txt, hi_donors,  NL-1), P_hi)/base    # final layer -> 1

print("="*66)
print("SANITY CONTROLS")
print("="*66)
print(f"  patch with its OWN activation   : {self_patch:+.3f}   must be ~0")
print(f"  patch final layer from the donor: {last_patch:+.3f}   must be ~1")
print(f"  baseline JS(struggling, fluent) : {base:.5f}")
if abs(self_patch) > 0.05 or last_patch < 0.8:
    raise SystemExit("  CONTROLS FAILED - hooks are wired wrong. Stop.")
print("  -> hooks verified\n", flush=True)

# ---- layer scan --------------------------------------------------------------
def scan(pairs, label):
    rows = []
    for a, b in pairs:
        for F in facts:
            ta, tb = f"{a} {F}", f"{b} {F}"
            P_a = run(ta); P_b = run(tb)
            bs = js(P_b, P_a)
            if bs < 1e-6: continue
            donors = cache_donors(ta)
            rows.append([1 - js(run(tb, donors, L), P_a)/bs for L in range(NL)])
    print(f"   {label} done ({len(rows)} items)", flush=True)
    return np.array(rows)

curves = {"PLACEBO (irrelevant)": scan([PLACEBO], "placebo")}
for dim, phrs in CUES.items():
    curves[dim] = scan(phrs, dim)

print("\n" + "="*66)
print("WHERE STYLE ENTERS THE DECISION")
print("restoration by layer: 0 = layer carries no style, 1 = carries all of it")
print("="*66)
means = {k: v.mean(0) for k, v in curves.items()}
for dim, m in means.items():
    print(f"\n{dim.upper().replace('_',' ')}   (n={len(curves[dim])})")
    for L in range(NL):
        bar = "#" * int(max(0, min(1, m[L])) * 40)
        print(f"   layer {L:>2}  {m[L]:+.3f}  {bar}")

def crossing(m, thr=0.5):
    for L in range(len(m)):
        if m[L] >= thr: return L
    return None

print("\n" + "="*66)
print("THE COMPARISON THAT DECIDES IT")
print("="*66)
pl = crossing(means["PLACEBO (irrelevant)"])
print(f"   placebo reaches 50% restoration at layer : {pl}")
for dim, m in means.items():
    if dim.startswith("PLACEBO"): continue
    c = crossing(m)
    print(f"   {dim:26s} reaches 50% at layer : {c}"
          f"   {'EARLIER than placebo' if (c is not None and pl is not None and c < pl) else 'same or later — not style-specific'}")
print("""
   Style crossing EARLIER than placebo means the model has committed to the
   patient's social description before it commits to irrelevant detail.
   Same layer means you found this model's general decision point, not
   anything about style. Either is reportable; only one is a finding.""")

json.dump({k: v.tolist() for k, v in means.items()},
          open("patching_curves.json","w"), indent=2)
print("\nsaved patching_curves.json")
print(json.dumps({k: [round(x,3) for x in v] for k, v in means.items()}, indent=2))
