# Experiment provenance

Every reported result traced to the script, inputs, model, configuration, seed
and output file that produced it. Compiled by reading the scripts and result
files in this repository; nothing here is inferred from memory or reconstructed
from the paper.

**Shared across every experiment unless stated otherwise**

| | |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (28 layers, hidden size 1536) |
| System prompt | `You are a physician taking a patient history. Ask one follow-up question.` |
| Inputs | 50 DiaLense scenario fact blocks, from `results/raw/*/conversations.jsonl`, deduplicated by `scenario_id` (see `data/README.md`) |
| Split seed | `random.Random(0)` over the shuffled scenario index — **fixed in every script**, so the 30 train / 20 held-out split is identical across all runs and all seeds |
| Phrasing split | first 5 of 8 phrasings per dimension train, last 3 held out |
| Metric | Jensen–Shannon divergence in bits over a 24-token greedily-decoded reference continuation (`CONT_LEN`/`TARGET_LEN = 24`) |
| Bootstrap seed | `np.random.default_rng(0)`, 4,000 resamples (baseline and robustness experiments) |

---

## Provenance table

| ID | Experiment | Script | Inputs | Model | Config | Seed | Result file | Used in |
|----|-----------|--------|--------|-------|--------|------|-------------|---------|
| E1 | Baseline style-vs-placebo sensitivity | `experiments/01_baseline.py` | 20 scenarios, 8 phrasings × 5 dimensions | Qwen2.5-1.5B-Instruct | `N_SCENARIOS=20`, `CONT_LEN=24`, position-counterbalanced | `random.Random(0)` (placebo selection); `default_rng(0)`, 4000 resamples | `results/baseline_v6.json` — reconstructed from console output, see note A | §5.1, Table 1, Fig. 2 |
| E2 | Placebo construction | no standalone script — builder functions inside E1, E4, E5 | — | — | family-constrained (default) and flat-pool builders; exact token-length match on both sides, minimum cosine distance in input-embedding space | `random.Random(0)` | none (intermediate) | §3.3, §4.4, App. B |
| E3 | Positive control, four conventions | `experiments/03_positive_control.py` | 30 train + 20 held-out scenarios, 10 pairs | Qwen2.5-1.5B-Instruct | `CONT_LEN=24`, `N_TRAIN_SCEN=30`, `N_EVAL_SCEN=20`, `N_PAIRS=10`; prefix = fluency phrasing 0 + its placebo | `random.Random(0)` | **not saved**, see note B | §3.4, App. D |
| E4 | Cue operationalisation (3 arms) | `experiments/04_cue_operationalisation.py` | 20 scenarios; descriptors, mechanical 2nd→3rd person rewrite, verbatim corpus `style_instruction` | Qwen2.5-1.5B-Instruct | `N_SCENARIOS=20`, `CONT_LEN=24`, `LEN_TOL=0`, frame `"The patient is described as follows. "` | `random.Random(0)`; per-assembly `random.Random(seed)`; `default_rng(0)`, 4000 resamples | **not saved**, see note B | §5.2, Table 2, Fig. 3, App. C, App. G |
| E5 | Placebo-construction robustness + independent reimplementation | `experiments/05_placebo_robustness.py` | 20 scenarios, 8 phrasings × 5 dimensions, run twice (family vs flat pool) | Qwen2.5-1.5B-Instruct | `N_SCENARIOS=20`, `CONT_LEN=24` | `random.Random(0)`; `default_rng(0)`, 4000 resamples | **not saved**, see note B | §5.3, Table 3, Fig. 4 |
| E6 | Soft-target self-distillation (original run) | `experiments/06_selfdistill.py` | 30 train scenarios × 5 phrasings/dim | Qwen2.5-1.5B-Instruct + LoRA r=16, α=32, dropout 0.05, all q/k/v/o/gate/up/down projections (18,464,768 params, 1.18%) | `EPOCHS=2`, `LR=3e-5`, `ACCUM=8`, `DRIFT_STOP=0.05`, `TARGET_LEN=24`, fp16 base / fp32 adapter | split `random.Random(0)`; **run seed not set — this script does not call `torch.manual_seed`**, see note C | `results/finetune_v2_ACCEPTED.json` | §5.4, Table 4 (original row), Fig. 6 |
| E7 | Multi-seed self-distillation | `experiments/07_selfdistill_seeds.py` | same as E6 | same as E6 | same as E6 | split `random.Random(0)` fixed; run seed set by the `SEED` constant to **1, 2, 3** via `torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed`, `random.Random` | **not saved**, see note B | §5.4, Table 4, Fig. 5 |
| E8 | Selectivity (style / clinical / placebo contrasts) | `experiments/08_selectivity.py` | 20 held-out scenarios | Qwen2.5-1.5B-Instruct + the E6 adapter (`ADAPTER="dialense_lora"`) | measurement only; adapter toggled via `PeftModel.disable_adapter()` | deterministic — greedy decoding, no sampling, no resampling | `results/selectivity_PASSED.json` | §5.5 |
| E9 | Neutral-prompt drift | no standalone script — computed inside E6, E7, E10 | prompts with no style sentence | Qwen2.5-1.5B-Instruct ± adapter | `DRIFT_STOP=0.05` abort threshold | inherits the parent run | inside `results/finetune_v2_ACCEPTED.json` (soft) and `results/finetune_attempt1_REJECTED.json` (hard) | §5.5, §5.7, Fig. 8 |
| E10 | Hard-target ablation + distribution diagnostics | `experiments/10_hard_target_ablation.py` | same split as E6 | Qwen2.5-1.5B-Instruct; base, teacher, soft-target student, hard-target student | `EPOCHS=2`, `ACCUM=8`, LoRA r=16/α=32/dropout 0.05; two hard-target runs at `LR=1e-4` and `LR=3e-5`; checkpoints every 40 steps | `random.Random(0)` for split and example order; run seed not set | **not saved**, see note B | §5.7, Table 5, Figs. 7–9 |
| E11 | Style probe | `experiments/11_probe.py` | 18 urgent + 18 matched routine scenarios; 5 train / 3 held-out phrasings per dimension | Qwen2.5-1.5B-Instruct ± LoRA adapter | layers `[2,4,6,8,12,16,20,24,27]`; logistic regression at final-prompt and cue-token positions; unit = (dimension, phrasing) → 15 held-out units | `random.Random(0)` for scenario order; cluster bootstrap `default_rng(0)`, `N_BOOT=2000`; cluster permutation `default_rng(0)`, `N_PERM=2000` | **not saved**, see note B | §5.6, Table 6, Fig. 10, App. J |
| E12 | Content (urgency) probe | `experiments/11_probe.py` (same run) | same; corpus urgency annotation | same | activations averaged within scenario; `C=0.05`; unit = scenario → 18 held-out units | same as E11 | **not saved**, see note B | §5.6, §8, App. J |
| E13 | Activation patching, 1.5 B | `experiments/12_patching_1.5b.py` — **reconstruction**, see note D | 20 held-out scenarios × 5 phrasings × 3 dimensions | Qwen2.5-1.5B-Instruct | final-prompt-position restoration; style-minus-placebo difference; bootstrap intervals; Benjamini–Hochberg across 28 layers; pre-specified ±0.03 magnitude floor | split `random.Random(0)`; bootstrap `default_rng(0)`, 4000 resamples (the last two are reconstruction choices) | `results/localisation_1.5B.json` (original); `results/raw/localisation_1.5B_rerun.json` (rerun, not yet produced) | §6, App. H |
| E14 | Activation patching, 0.5 B (reported as a failed replication) | `experiments/13_patching_0.5b.py` | 8 scenarios × 3 phrasings = 24 items per dimension | **`Qwen/Qwen2.5-0.5B-Instruct`** (24 layers) — CPU | single fixed placebo pair; sanity controls: patch-with-own-activation and patch-final-layer | deterministic scan; no sampling | `results/patching_curves_0.5B.json` | §6 |
| E15 | Conversation-level transfer | script lives in the DiaLense repository, not here | 20 held-out scenario pairs, 4 conversations each, 11 turns | patient: unmodified base model; clinician: base vs adapted | pre-existing unmodified friction detectors | not recorded in the result file | `results/partI_linkage.json` | App. I |
| E16 | Figures and tables | `analysis/make_figures.py`, `analysis/make_latex_tables.py`, `analysis/build_paper.py` | `analysis/master.json` only | — | no value is typed into plotting code | deterministic | `figures/*.pdf`, `analysis/tables/*.tex`, `paper/acl_paper.tex` | all figures and tables |

---

## Notes

**A. `baseline_v6.json` is a transcription.** The file declares its own status in
its `_meta` block: `"reconstructed_from": "console output; original file lost to
a Colab runtime reset"`. Its values match the paper exactly, but it is a
faithful transcription of printed output rather than a file the program wrote.
Re-running E1 replaces it with a genuine artifact.

**B. Six result files were never written to disk.** The runs happened and their
numbers are in the paper; no artifact survives. See
[`../results/MISSING.md`](../results/MISSING.md) for the file list, the claims
each one backs, and the re-run order. Re-running them is not new science — it
is putting a file behind numbers that currently have none.

**C. Seeds that were not recorded.** `experiments/06_selfdistill.py` (E6) and
`experiments/10_hard_target_ablation.py` (E10) fix the data split with
`random.Random(0)` but never call `torch.manual_seed`, so LoRA initialisation,
dropout masks and data order in those runs were governed by the framework
default RNG state and **were not recorded in the experiment artifacts**. This is
recoverable going forward but not retroactively: an exact bitwise re-run of the
original E6 and E10 is not possible. E7 exists precisely because of this — it
re-runs E6 under three explicitly seeded conditions (1, 2, 3) and reports mean
and range, which is the number the paper leans on. `results/selectivity_PASSED.json`
records `"Two independent runs agreed to 3 decimal places on every training loss
and every reported number"`, which bounds how much this matters in practice.

**D. E13's script is a reconstruction, not the original.** The script that
produced `results/localisation_1.5B.json` was never saved. It is not in this
repository, not in any surviving session transcript, and not on any machine
reachable from the environment this audit ran in.

`experiments/12_patching_1.5b.py` reconstructs it. Its header carries a
parameter-by-parameter table naming the evidence behind every recovered value —
the artifact's own `_measure`/`_power`/`_reporting_rule` fields, the paper's §6.2
and Appendix H, the surviving 0.5 B sibling (from which the hook mechanics and
the restoration formula are taken verbatim), and the split convention shared by
every other script here. Five details were **not** recoverable and are flagged
both in the header and in the output's `_config`: which five phrasings, which
three placebo pairs, the bootstrap resample count, the bootstrap unit, and the
floating-point precision. Each is a command-line flag.

The reconstruction's statistics layer is verified independently of any model:
`python experiments/12_patching_1.5b.py --selftest` checks Benjamini–Hochberg
against the published Benjamini & Hochberg (1995) Table 1 example, then confirms
on synthetic curves that a planted positive bump, a planted negative region, a
significant-but-sub-floor effect and pure noise are each handled as the
reporting rule requires. That test passes.

**What has not been done: the reconstruction has not been run against the 1.5 B
model**, so it has not been shown to reproduce the committed artifact. §6 is
therefore *documented and independently executable*, not *reproduced*. Running
`python experiments/12_patching_1.5b.py` followed by
`python tools/verify_results.py` settles it.

**E. The LoRA adapter was recovered from a retained Drive backup.** The
historical `_adapter_lost` field in `results/finetune_v2_ACCEPTED.json` reflected
the state of the Colab runtime, not the later-discovered external backup. The
recovered `dialense_lora` checkpoint matches the reported base model, r=16,
α=32, seven Qwen projection targets, 392 tensors and 18,464,768 parameters. Its
SHA-256 hashes are recorded in `docs/ADAPTER_RECOVERY.md`. Because no hash was
recorded at training time, exact byte-level identity with the evaluated run is
not independently provable. E8 and E11/E12 can use this recovered adapter by
path. Adapter weights remain gitignored and should be distributed on Hugging Face.

---

## What "seed" means in this project, and why the split seed is the important one

Two random processes matter and they are deliberately separated.

The **data split** — which 30 scenarios train, which 20 are held out, which 5
phrasings per dimension train and which 3 are held out — is fixed at
`random.Random(0)` in every script, independent of any run seed. If the split
moved with the seed, the seeded runs would be measuring split variance rather
than initialisation variance and the three numbers would not be comparable.

The **run seed** — LoRA initialisation, data order, dropout masks — varies. E7
sets it explicitly to 1, 2 and 3 and reports mean 91.49% with range
90.94–91.82%. E6 and E10 did not set it (note C).

Bootstrap and permutation seeds are fixed at `0` throughout, so the reported
intervals and *p*-values are deterministic given the same measurements.
