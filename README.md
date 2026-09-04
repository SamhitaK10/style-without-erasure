# Style Without Erasure

**Measuring and Removing Speaker-Style Sensitivity in Language Models**

Samhita Kondareddy · [Read the paper](paper/Style_Without_Erasure_Samhita_Kondareddy_final.pdf) · [Model adapter](https://huggingface.co/samhitak10/qwen2.5-1.5b-style-without-erasure-lora)

A language model asked the same clinical question gives a different answer
depending on how the person asking is described, even when every fact in the
request is identical. This repository contains the measurement protocol, the
intervention that removes the behavioural effect, and the analysis showing that
the style information is still linearly decodable afterwards at the resolution
the design can measure.

All experiments use a single model, `Qwen/Qwen2.5-1.5B-Instruct`. Nothing here
establishes that these results hold for other models or scales.

---

## Overview

Any two different prompts produce some divergence in a model's output
distribution. "We changed the style sentence and the output moved" is therefore
not evidence of a style effect; it may be evidence only that the prompt changed.
The measurement here is built around three controls that make the difference
legible:

- a **matched placebo contrast** — a clinically irrelevant sentence swap matched
  to the style swap on the exact token length of *both* sides and on cosine
  distance in the model's input-embedding space, so the change in prompt length
  cancels in the difference;
- a **positive control** — replacing the entire clinical history, measured on the
  same prompt form and scenario set, which turns an uninterpretable divergence
  into a percentage of a full content swap;
- **position counterbalancing** — every comparison is run with the style sentence
  first and with the placebo first, and averaged.

The intervention is **self-distillation against a style-blind teacher**. The
frozen base model reads the cue-free prompt and produces a full output
distribution; a LoRA-adapted student reads the *cue-present* prompt and is
trained to match it. It is context distillation with the context moved to the
student's side. Because forward KL is mass-covering, the teacher's entropy acts
as a floor on the student's, which is the mechanism that separates this from an
objective that simply makes the model confidently insensitive to everything.

The final question is whether removing the behaviour removed the information. It
did not, as far as this design can tell: a linear probe still recovers the style
dimension from the residual stream, and **no layer shows a reduction that
survives both a cluster-corrected bootstrap interval excluding zero and a cluster
permutation test**. That is reported as a bound on what the design could have
detected, not as proof that the representation is unchanged.

### Research questions

1. Does swapping a speaker-style descriptor move the next-token distribution
   more than a length- and embedding-matched irrelevant swap, and by how much
   relative to replacing the clinical content entirely?
2. Does that effect survive different ways of writing the style cue, and
   different ways of constructing the placebo?
3. Can the behavioural effect be removed without collapsing the output
   distribution or making the model insensitive to clinical content?
4. After the behavioural effect is removed, is the style dimension still
   linearly decodable from internal activations?
5. Where in the network does the style-versus-placebo difference arise under
   final-position activation patching?

---

## Key Findings

Divergence is Jensen–Shannon in bits over a fixed 24-token greedily-decoded
reference continuation. Every number below comes from `analysis/master.json`,
which the figures and tables are generated from.

| | Result |
|---|---|
| Style vs matched placebo | **3.6× – 7.4×** across five dimensions; 8/8 phrasings positive per dimension; one-sided Wilcoxon *p* = .0039 (the floor at *n* = 8) |
| As a share of a full content swap | **3.1% – 6.0%** |
| Robustness to cue wording | 5/5 dimensions positive in all three arms; exact sign test *p* = .0312 each (the floor at *n* = 5) |
| Robustness to placebo construction | Direction robust; magnitude is not — three of five dimensions fall by about a third under a flat placebo pool |
| Self-distillation, held-out reduction | **91.5% mean**, range 90.9 – 91.8% across three seeds |
| Perplexity on genuine clinician text | 12.23 → 12.01 – 12.08 (improves slightly) |
| Drift on style-neutral prompts | 0.0023 – 0.0026 bits = **0.01×** the matched positive control |
| Output distribution after the intervention | entropy 2.022 vs base 2.024; top-1 0.607 vs 0.604; positions with *p* > .99: 12.4% vs 12.7% |
| Selectivity | Response to a full clinical-content swap **+2.7%** (unchanged); style influence relative to the clinical effect falls 5.22% → 0.43% |
| Hard-target ablation (identical hyperparameters, one-hot targets) | entropy collapses to **0.278 bits**, top-1 rises to **0.936**, 68.8% of positions sit above *p* = .99, and the sensitivity metric flips sign between two learning rates (−24.2% vs +19.2%) while the damage reproduces (0.302 / 0.3025 bits) |
| Linear probe, base model | up to **0.884 AUC**; cue-token positions read out **+0.064 AUC** above the final position through layer 20 |
| Linear probe, after the intervention | no layer's reduction survives both tests; **bound: 0.107 AUC** at 15 held-out units |
| Activation patching, style minus placebo | small early-layer effects peaking at **+0.065 (L4, fluency)** and **+0.074 (L5, confidence)**; a large negative mid-network region reaching **−0.355 (L17)** |

Two findings are negative and are reported as such: the placebo contrast also
falls 75.4% after the intervention, so the removal is not style-specific; and
the conversation-level transfer experiment was uninformative rather than
negative (Appendix I).

---

## Method

**Style contrast.** Five dimensions — fluency, health literacy, confidence,
emotional expressiveness, communication style — each with eight paired
third-person descriptor sentences differing only in the level of that dimension.
The clinical content block is byte-identical on both sides; identity is
hash-verified. All 40 pairs are in `src/style_erasure/cues.py` and Appendix A.

**Matched placebo contrast.** For each style pair, a clinically irrelevant
sentence pair with the same token count on each side and the closest achievable
cosine distance in input-embedding space. Two builders are compared —
family-constrained (both placebo sentences from one stem family) and flat pool —
because the measured magnitude depends on the choice.

**Positive control.** Replacing the entire clinical history, measured on the
same prompt form and scenario set as the numerator.

**Soft-target self-distillation.** Teacher: frozen base model on the cue-free
prompt. Student: LoRA adapter (r = 16, α = 32, dropout 0.05, on all attention
and feed-forward projections — 18,464,768 trainable parameters, 1.18% of the
model) on the cue-present prompt. Loss: forward KL to the teacher's full
distribution.

**Distribution-preservation tests.** An intervention scored by a sensitivity
metric can satisfy that metric by becoming insensitive to everything, and
perplexity does not catch it. Three numbers do: output entropy, mean top-1
probability, and the fraction of positions above *p* = .99. These are reported
for every trained variant, alongside a selectivity check (does the model still
respond to a full clinical-content swap?) and drift on style-neutral prompts.

**Probing.** Logistic-regression probes on residual-stream activations at the
final prompt position and at the cue-token positions, at nine layers. The unit
of analysis is the **(dimension, phrasing) pair** — 15 held-out units — not the
item; intervals are cluster bootstraps and tests are cluster permutations. Four
earlier probe runs failed on exactly this point and are kept in
[`archive/probe/`](archive/probe/) with the specific error documented in each.

**Activation patching.** Layerwise restoration at the final prompt position,
reported as the difference between the style curve and the matched placebo
curve, with bootstrap intervals, Benjamini–Hochberg correction across layers,
and a pre-specified ±0.03 magnitude floor. This measures the causal contribution
of the patched state at that position; it is not a claim about where information
is stored.

---

## Repository Structure

| Path | Contents |
|---|---|
| `paper/` | The manuscript PDF and its Markdown source |
| `model_card/` | The Hugging Face model card for the trained adapter. The weights themselves are not in this repository |
| `src/style_erasure/` | Shared library. Holds `cues.py` — the 40 cue phrasings, extracted verbatim. The experiment scripts still carry their own inline copies; finishing the extraction is item 7 in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) |
| `experiments/` | One numbered script per experiment; the number is the run order |
| `analysis/` | `master.json` plus the figure, table and paper builders. No value is typed into plotting code — everything reads `master.json` |
| `figures/` | The 11 paper figures as PDF |
| `data/` | Provenance and licence documentation. The scenario file is generated, not committed |
| `tools/` | `fetch_scenarios.py` (pinned scenario extraction), `verify_results.py` (checks every reported number against its artifact), `publish_adapter.py` (validate and publish the LoRA adapter), `check_results.py` (which outputs exist) |
| `docs/` | [Experiment provenance](docs/EXPERIMENT_PROVENANCE.md), [result verification](docs/RESULT_VERIFICATION.md), [third-party attribution](docs/THIRD_PARTY.md), [reproducibility status](docs/REPRODUCIBILITY.md), run order, results ledger, design decisions |
| `archive/` | Superseded experiment versions, each kept because it documents a specific error |

---

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/SamhitaK10/style-without-erasure.git
cd style-without-erasure
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Exact package versions from the original runs were not preserved — Colab
supplied the stack and no lockfile was saved. `requirements.txt` states
compatibility floors and explains what evidence each one rests on. Results
produced under different versions may differ in the last decimal places.

**Hardware.** Every GPU experiment was run on a single NVIDIA T4. The two
training scripts require at least 5 GiB of free device memory and abort
otherwise. `experiments/13_patching_0.5b.py` runs on CPU. The analysis scripts
need no GPU.

---

## Data

Fifty clinical content blocks, one per scenario, of the form
`Reported history: <fact>; <fact>; …`. The same block appears on both sides of
every style comparison, so propositional content is identical by construction.

```bash
python tools/fetch_scenarios.py     # → data/scenarios.jsonl (50 rows)
```

The scenarios come from [DiaLense](https://github.com/SamhitaK10/DiaLense), and
their clinical content is grounded in **MTS-Dialog** (Ben Abacha et al., EACL
2023), which is published under **CC BY 4.0**. The fetch is pinned to a specific
DiaLense commit. Forty of the fifty scenarios are traceable to specific
MTS-Dialog records through DiaLense's provenance table; ten (S1–S10) predate
that table and have no recorded source record.

Nothing from either source is committed here. Full chain, licence terms,
attribution requirements and the S1–S10 limitation:
[`data/README.md`](data/README.md) and [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).

---

## Models

| | |
|---|---|
| Identifier | `Qwen/Qwen2.5-1.5B-Instruct` |
| Publisher | Qwen Team, Alibaba Cloud |
| URL | <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct> |
| Licence | Apache-2.0 |
| Architecture | 28 layers, hidden size 1536, 12 attention heads, 2 KV heads, vocabulary 151,936 |
| Revision | Not recorded — the scripts call `from_pretrained` without a `revision=` argument |

### Trained adapter

The LoRA weights are **not stored in this Git repository**. `.gitignore` excludes
adapter weights deliberately; the public model release belongs on Hugging Face.

A retained Drive backup of the original `dialense_lora` directory was recovered
after the initial audit had incorrectly treated the checkpoint as lost. The
recovered checkpoint was validated against `results/finetune_v2_ACCEPTED.json`:

- base model: `Qwen/Qwen2.5-1.5B-Instruct`
- LoRA rank: 16
- LoRA alpha: 32
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- tensors: 392
- trainable parameters: 18,464,768
- weight size: 73,911,112 bytes

The recovered files hash to:

```text
adapter_config.json       sha256 884371482b02bb30481ffd2c6406cff933976b0e2a9e42505a6b683f06a21159
adapter_model.safetensors sha256 d48899aab39a1d18060825b2384cafcf889b4110020ba246824c572e11d12d8c
```

The historical `_adapter_lost` field inside the accepted result artifact is left
unchanged because it records what was believed at the time. It is superseded by
[`docs/ADAPTER_RECOVERY.md`](docs/ADAPTER_RECOVERY.md). No training-time hash was
recorded, so exact byte-for-byte identity with the evaluated checkpoint cannot be
independently proven; its architecture and configuration match the reported run
exactly.

Validate the recovered directory before publishing:

```bash
python tools/publish_adapter.py --adapter ./dialense_lora --dry-run
python tools/publish_adapter.py --adapter ./dialense_lora \
    --repo-id SamhitaK10/qwen2.5-1.5b-style-without-erasure-lora
```

`publish_adapter.py` refuses to upload an adapter whose `adapter_config.json` or
parameter count does not match the reported configuration. The model card it
uploads is [`model_card/README.md`](model_card/README.md).

`python tools/publish_adapter.py --selftest` independently re-derives the
18,464,768-parameter count from the Qwen2.5-1.5B architecture.

`Qwen/Qwen2.5-0.5B-Instruct` (24 layers) is used by one experiment only, whose
result the paper reports as a failed replication. Neither checkpoint is
redistributed. Details: [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).

---

## Running Experiments

Every script in `experiments/` was written as a single Colab cell. There is no
`argparse`, no `__main__` guard, and hyperparameters are module-level constants;
`07_selfdistill_seeds.py` requires editing `SEED` in the source between runs.
The commands below are therefore given as `python <script>` plus the constant to
edit, which is an accurate description of the current interface rather than an
aspirational one. Replacing it is item 1 in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

Order matters and is explained in [`docs/RUN_ORDER.md`](docs/RUN_ORDER.md).
Times are for a single T4. Restart the runtime between GPU scripts.

```bash
# environment and data
pip install -r requirements.txt
python tools/fetch_scenarios.py

# 1. baseline style-vs-placebo sensitivity                      ~25 min
python experiments/01_baseline.py

# 2. positive control, four conventions                          ~5 min
python experiments/03_positive_control.py

# 3. cue operationalisation, three arms                      ~25-35 min
python experiments/04_cue_operationalisation.py

# 4. placebo-construction robustness                            ~10 min
python experiments/05_placebo_robustness.py

# 5. soft-target self-distillation (writes the LoRA adapter)    ~30 min
python experiments/06_selfdistill.py

# 6. multi-seed self-distillation — edit SEED to 1, then 2, then 3
python experiments/07_selfdistill_seeds.py                    # ~30 min each

# 7. selectivity (needs the adapter from step 5)                ~10 min
python experiments/08_selectivity.py

# 8. hard-target ablation + distribution diagnostics            ~35 min
python experiments/10_hard_target_ablation.py

# 9. style and content probing (needs the adapter)              ~20 min
python experiments/11_probe.py

# 10. activation patching, 1.5B (reconstruction)               ~1-2 h
python experiments/12_patching_1.5b.py --selftest    # statistics only, no GPU
python experiments/12_patching_1.5b.py               # the full run

# 11. activation patching, 0.5B — CPU                          ~10 min
python experiments/13_patching_0.5b.py

# which outputs exist, and whether they match the paper
python tools/check_results.py
python tools/verify_results.py     # writes docs/RESULT_VERIFICATION.md
```

Steps 7 and 9 load a LoRA adapter by path. The recovered main-paper adapter can be used directly for these steps after downloading it from the Hugging Face release. If you are reproducing training from scratch instead, run step 5 or one of the seeded step 6 runs first.

**Step 10's script is a reconstruction.** The script that produced
`results/localisation_1.5B.json` (paper §6, Appendix H) was never saved.
`experiments/12_patching_1.5b.py` reconstructs it from the artifact's own design
fields, the paper's method section, and the surviving 0.5 B sibling; its header
names the evidence behind every recovered parameter and flags the five that were
not recoverable. Its statistics layer is verified independently of any model:

```bash
python experiments/12_patching_1.5b.py --selftest   # no GPU, no model, no network
```

It has **not** been run against the 1.5 B model, so §6 is documented and
executable but not yet reproduced. Running step 10 and then
`python tools/verify_results.py` settles it.

---

## Reproducing Figures

All 11 figures and 6 tables are generated from `analysis/master.json`. No value
is typed into plotting code.

```bash
python analysis/make_figures.py        # → figures/*.pdf and figures/*.png
python analysis/make_latex_tables.py   # → analysis/tables/table{1..6}.tex
python analysis/build_paper.py         # → paper/acl_paper.tex
cd paper && xelatex acl_paper.tex && xelatex acl_paper.tex
```

`build_paper.py` needs `pandoc` and `xelatex` with the TeX Gyre fonts. The full
chain was verified to run clean from this layout and produce the committed
27-page PDF. `figures/*.png` and `paper/acl_paper.{tex,pdf}` are generated and
gitignored; the PDF figures and `paper/style_without_erasure.pdf` are committed.

---

## Results

Which results have a committed artifact behind them, and which do not:

| Committed and matching the paper | File |
|---|---|
| Baseline ratios, per-dimension divergences, positive control | `results/baseline_v6.json` (a faithful transcription of console output — the original file was lost to a runtime reset) |
| Soft-target self-distillation, original run | `results/finetune_v2_ACCEPTED.json` |
| Hard-target first attempt (rejected) | `results/finetune_attempt1_REJECTED.json` |
| Selectivity | `results/selectivity_PASSED.json` |
| Activation patching, 1.5 B | `results/localisation_1.5B.json` (script reconstructed — see above) |
| Activation patching, 0.5 B (failed replication) | `results/patching_curves_0.5B.json` |
| Conversation-level transfer | `results/partI_linkage.json` |

Six experiment outputs were never written to disk, so roughly a dozen reported
numbers — most of Table 4, and all of Tables 2, 3, 5 and 6 — currently have no
artifact behind them. They are listed with their re-run order in
it is putting a file behind numbers that already exist.

Every reported number is checked against its artifact by
`python tools/verify_results.py`, which writes
[`docs/RESULT_VERIFICATION.md`](docs/RESULT_VERIFICATION.md). As committed:
**38 EXACT, 4 within rounding tolerance, 0 mismatches**, 20 rows with no
artifact, and 1 not yet run. Those 21 rows become real comparisons the moment
the corresponding JSON lands in `results/raw/`.

Full script-to-result-to-figure mapping, including seeds and configurations:
[`docs/EXPERIMENT_PROVENANCE.md`](docs/EXPERIMENT_PROVENANCE.md).

---

## Limitations

**One model, one scale.** Every reported result comes from
`Qwen/Qwen2.5-1.5B-Instruct`. Nothing here shows the baseline magnitude, the
intervention's efficacy, or the representational null holds for other model
families or sizes. The one attempt at a second scale — activation patching on
the 0.5 B model — did **not** replicate the 1.5 B pattern, and is reported as a
failed replication rather than quietly dropped.

**The removal is not style-specific.** After the intervention the matched
placebo contrast also falls 75.4%. The model did not learn to disregard
communication style specifically; it learned to disregard non-content framing in
general, while retaining sensitivity to clinical content. This is a genuine
limit on the claim.

**The representational result is a bound, not a demonstration of preservation.**
"No layer shows a reduction surviving both tests" at 15 held-out units means the
design can exclude drops larger than 0.107 AUC. It does not show the
representation is unchanged. The content probe is separately underpowered: the
corpus has 50 scenarios, 18 of them urgent, so a scenario-level control cannot
exceed 18 units.

**Stated style, not enacted style.** The cues are third-person descriptors of
how a speaker communicates, not speech by speakers who actually communicate that
way. This buys byte-identical content and matched controls at the cost of
construct validity. A model may respond differently to enacted dysfluency than
to a description of it.

**Measured magnitude depends on how the placebo is built.** Direction is robust
— every dimension stays positive under both builders — but three of five
dimensions fall by about a third under a flat placebo pool. The
family-constrained builder is the primary one and is specified so the choice is
reproducible.

**Next-token measurement, not generation quality.** Divergence is measured over
a fixed 24-token reference continuation. Nothing here shows a reader would judge
two generations differently, and no human evaluation was performed.

**No conversation-level transfer claim.** Across 20 held-out scenario pairs the
baseline friction gap was ≈ −0.025 against absolute levels near 1.55 —
indistinguishable from zero and opposite in sign to the disparity the detectors
were built to find. There was no gap available to narrow. The experiment is
reported as uninformative, not as a negative result.

**Activation patching bounds a causal contribution; it does not localise
storage.** The large negative mid-network region is most naturally read as
evidence that style information is distributed across the cue-token positions,
so overwriting a single position understates it. That is a statement about the
method's reach as much as about the model.

**Clinical setting is a testbed, not a claim.** Nothing in the measurement or
the intervention is specific to medicine. No claim is made about patient
outcomes, care quality, or clinical safety.

---

## Citation

```bibtex
@misc{kondareddy_style_without_erasure,
  title  = {Style Without Erasure: Measuring and Removing Speaker-Style
            Sensitivity in Language Models},
  author = {Kondareddy, Samhita},
  note   = {Manuscript}
}
```

If you use the clinical scenarios, you must also cite the corpus they derive
from — see [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md) for the required
MTS-Dialog citation and attribution.

---

## Licenses and Third-Party Resources

| Component | Licence |
|---|---|
| Code in this repository (`src/`, `experiments/`, `analysis/`, `tools/`) | **MIT** — [`LICENSE`](LICENSE) |
| Result files and figures | **MIT** |
| Clinical scenarios and anything derived from them | **CC BY 4.0**, inherited from MTS-Dialog — [`LICENSE-DATA`](LICENSE-DATA) |
| `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-0.5B-Instruct` | Apache-2.0, held by the Qwen Team. Not redistributed |
| MTS-Dialog corpus | CC BY 4.0, held by its authors. Not redistributed |
| The paper (`paper/`) | Author's copyright; no separate open licence granted |

The MIT licence covers this repository's own code only. It does not relicense
Qwen, MTS-Dialog, or DiaLense. Attribution requirements and full terms:
[`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).


