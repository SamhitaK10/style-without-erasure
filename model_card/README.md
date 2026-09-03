---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
  - lora
  - peft
  - self-distillation
  - knowledge-distillation
  - interpretability
  - fairness
  - research
language:
  - en
---

# Style Without Erasure LoRA for Qwen2.5-1.5B-Instruct

A LoRA adapter trained by self-distillation against a **style-blind teacher**:
the frozen base model reading a *cue-free* prompt supervises the adapted model
reading the *cue-present* version of the same prompt. It reduces how much the
model's next-token distribution moves when a speaker is described differently,
while the propositional content of the request is held byte-identical.

This is a **research artifact** accompanying *Style Without Erasure: Measuring
and Removing Speaker-Style Sensitivity in Language Models*. It is not a product,
and §"Out of scope" below is not boilerplate — please read it.

- Code, paper and full provenance: https://github.com/SamhitaK10/style-without-erasure
- Base model: [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache-2.0)

## Checkpoint provenance

This release uses a recovered retained backup of the `dialense_lora` checkpoint.
Its configuration matches the paper run exactly: Qwen2.5-1.5B-Instruct, r=16,
α=32, all seven Qwen attention/MLP projection targets, 392 tensors and
18,464,768 adapter parameters. The released files should match these hashes:

```text
adapter_config.json       sha256 884371482b02bb30481ffd2c6406cff933976b0e2a9e42505a6b683f06a21159
adapter_model.safetensors sha256 d48899aab39a1d18060825b2384cafcf889b4110020ba246824c572e11d12d8c
```

No checksum was recorded at the moment of the original evaluation, so exact
byte-for-byte identity with the evaluated checkpoint is not independently
provable. The architecture and configuration match the reported run exactly.

---

## What it does

The base model answers the same clinical question differently depending on how
the person asking is described. Measured against a placebo contrast matched on
exact token length and input-embedding distance, that sensitivity runs 3.6×–7.4×
the matched control across five speaker-style dimensions — fluency, health
literacy, confidence, emotional expressiveness, communication style.

This adapter is trained to produce, in the presence of a style cue, the
distribution the base model would have produced without one. The teacher is the
model itself, so no external labels and no judgment about how a system *ought*
to adapt to a speaker enter the objective.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, "REPLACE_WITH_YOUR_HF_REPO_ID")
model.eval()

msgs = [
    {"role": "system", "content": "You are a physician taking a patient history. Ask one follow-up question."},
    {"role": "user", "content": "The patient speaks limited, ungrammatical English as a second language. "
                                "Reported history: chest pressure for 3 days; radiates to the left arm; "
                                "father had a heart attack at 55."},
]
ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=48)[0][ids.shape[-1]:], skip_special_tokens=True))
```

The system prompt above is the one used throughout training and evaluation.
Behaviour under other system prompts was not measured.

`model.disable_adapter()` restores base behaviour — which is also the point made
under "Suppression, not erasure" below.

## Training

| | |
|---|---|
| Method | Self-distillation; forward KL to the frozen base model's full output distribution on the cue-free prompt |
| Adapter | LoRA, r=16, α=32, dropout 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | 18,464,768 (1.18% of the model) |
| Optimiser | AdamW, lr 3e-5, 2 epochs, gradient accumulation 8, gradient-norm clip 1.0 |
| Precision | fp16 base, fp32 adapter, gradient checkpointing |
| Supervision target | 24-token reference continuation, greedily decoded from the neutral prompt |
| Training data | 30 clinical scenarios × 5 phrasings per dimension; 20 scenarios and 3 phrasings held out |
| Drift guard | Training aborts if divergence on style-neutral prompts exceeds 0.05 bits (never fired) |

Forward KL is mass-covering, so the teacher's entropy acts as a floor on the
student's. That is the mechanism that separates this from an objective which
merely makes the model confidently insensitive to everything — see the ablation
below.

## Evaluation

Held-out phrasings crossed with held-out scenarios; the model saw neither.
Divergence is Jensen–Shannon in bits over a fixed 24-token reference
continuation.

| Metric | Value | Backing artifact |
|---|---|---|
| Style sensitivity, mean reduction | **91.6%** (original run) | `results/finetune_v2_ACCEPTED.json` |
| Same, across 3 seeds | 91.5% mean, range 90.9–91.8% | *raw output not preserved* |
| Perplexity on genuine clinician text | 12.2345 → **12.0237** (improves) | `results/finetune_v2_ACCEPTED.json` |
| Drift on style-neutral prompts | **0.002632 bits** = 0.01× a full content-swap control | `results/finetune_v2_ACCEPTED.json` |
| Response to a full clinical-content swap | **+2.7%** (unchanged) | `results/selectivity_PASSED.json` |
| Style influence relative to the clinical effect | 5.22% → **0.43%** | `results/selectivity_PASSED.json` |
| Output entropy | 2.022 vs base 2.024 | *raw output not preserved* |
| Mean top-1 probability | 0.607 vs base 0.604 | *raw output not preserved* |
| Positions with p > .99 | 12.4% vs base 12.7% | *raw output not preserved* |

Rows marked *raw output not preserved* are reported in the paper but their raw
experiment output was lost; the repository documents this openly in
[`results/MISSING.md`](https://github.com/SamhitaK10/style-without-erasure/blob/main/results/MISSING.md).
They are listed here for completeness, not as verified claims.

**Ablation worth knowing about.** The same objective with hard one-hot targets,
every other hyperparameter fixed, collapses the output distribution: entropy
0.278 bits, 68.8% of positions above p = .99, and the sensitivity metric flips
sign between two learning rates while the damage reproduces. A model can satisfy
a "reduced style sensitivity" metric by becoming insensitive to everything, and
perplexity does not catch it. This adapter is the soft-target arm.

## Out of scope

**Not for clinical use.** No patient, clinician, or care setting was involved at
any point. The clinical framing is a testbed chosen because it supplies matched
scenarios with verified content identity — it is not a claim about medicine. No
evaluation of clinical accuracy, safety, or patient outcomes was performed.

**The removal is not style-specific.** After training, the matched *placebo*
contrast also falls 75.4%. The model did not learn to disregard communication
style specifically; it learned to disregard non-content framing in general,
while retaining sensitivity to clinical content. Do not describe this adapter as
"debiased".

**Suppression, not erasure.** A linear probe still recovers the style dimension
from the residual stream after training, with no layer showing a reduction that
survives both a cluster-corrected bootstrap interval and a cluster permutation
test. The design can exclude drops larger than 0.107 AUC at 15 held-out units;
it cannot show the representation is unchanged. Behaviourally the adapter does
what was asked, but the information is still there, the effect is one
`disable_adapter()` away from reverting, and robustness to further fine-tuning
was not tested.

**One model, one scale.** Everything above is `Qwen2.5-1.5B-Instruct`. Nothing
establishes that the method transfers to other families or sizes. The one
attempt at a second scale — activation patching on the 0.5B model — did not
replicate and is reported as a failed replication.

**Stated style, not enacted style.** The cues are third-person descriptors of
how a speaker communicates, not speech by people who actually communicate that
way. A model may respond differently to enacted dysfluency than to a description
of it.

**Next-token measurement.** Divergence is measured over a fixed 24-token
continuation. No human evaluation of generation quality was performed.

## Training data

The training prompts combine author-written style descriptors with clinical
content blocks derived from the 50 scenarios in
[DiaLense](https://github.com/SamhitaK10/DiaLense), whose clinical content is
grounded in the **MTS-Dialog** corpus.

MTS-Dialog is published under **CC BY 4.0** and its attribution requirement
passes to work derived from it:

> Ben Abacha A, Yim W, Fan Y, Lin T. *An Empirical Study of Clinical Note
> Generation from Doctor-Patient Encounters.* Proceedings of EACL 2023,
> pp. 2291–2302. https://aclanthology.org/2023.eacl-main.168

Scenario content follows a source note but is restructured into a fixed ten-fact
schema with identifying specifics generalised; the scenarios are not verbatim
copies. Forty of the fifty are traceable to specific MTS-Dialog records; ten
predate the provenance table and have no recorded source. The upstream corpus
states no de-identification procedure, so **no de-identification claim is made
here**. Full chain:
[`data/README.md`](https://github.com/SamhitaK10/style-without-erasure/blob/main/data/README.md).

No corpus text is redistributed in this repository — only adapter weights.

## Licence

The adapter weights are released under **Apache-2.0**, matching the base model
they are a derivative of. Using them requires the base model, which carries its
own Apache-2.0 terms held by the Qwen Team.

The attribution obligation on the MTS-Dialog-derived training data (CC BY 4.0)
is satisfied by the citation above and travels with any redistribution.

## Citation

```bibtex
@misc{kondareddy_style_without_erasure,
  title  = {Style Without Erasure: Measuring and Removing Speaker-Style
            Sensitivity in Language Models},
  author = {Kondareddy, Samhita},
  note   = {Manuscript},
  url    = {https://github.com/SamhitaK10/style-without-erasure}
}
```
