# What is missing from `results/`, and what depends on it

This file exists so a reader does not have to discover any of it themselves.

Every figure and table in the paper is generated from `analysis/master.json`.
`master.json` was **assembled by hand** from console output and from the seven
JSON files committed in this directory. Six experiment outputs it draws on were
never saved to disk, and one committed result's script had to be reconstructed.

---

## 1. Committed, and matching the paper

| File | Backs | Verified |
|---|---|---|
| `baseline_v6.json` | Table 1, Figure 2, positive control 0.1475 bits | ✅ all five dimensions exact — but see the caveat below |
| `finetune_v2_ACCEPTED.json` | Table 4, "original (unseeded)" row | ✅ exact, after the §3 corrections |
| `finetune_attempt1_REJECTED.json` | §5.7 hard-target narrative; drift 0.302 bits | ✅ |
| `selectivity_PASSED.json` | §5.5 (+2.7% / −75.4% / 5.22%→0.43%) | ✅ exact |
| `localisation_1.5B.json` | §6 and Appendix H, all patching numbers | ✅ output matches the paper exactly — but see §2 |
| `patching_curves_0.5B.json` | §6, the failed 0.5 B replication | ✅ |
| `partI_linkage.json` | Appendix I, the conversation-level null | ✅ |

**Caveat on `baseline_v6.json`.** The file declares its own status:

```json
"reconstructed_from": "console output; original file lost to a Colab runtime reset"
```

It is a faithful transcription, not a program output. Re-running
`experiments/01_baseline.py` would replace it with a genuine artefact.

---

## 2. Committed output whose script was reconstructed

**`localisation_1.5B.json`** — the activation-patching section of the paper
(§6, Appendix H) rests on this file. The script that produced it was never
saved and could not be found anywhere: not in this repository, not in any
surviving session transcript, not on disk.

`experiments/12_patching_1.5b.py` reconstructs it from the artifact's own
design fields, the paper's method description, and the surviving 0.5 B sibling
script. Its statistics layer is verified against the published Benjamini &
Hochberg (1995) example and against synthetic curves with planted effects
(`--selftest`, passes).

**It has not been run against the 1.5 B model.** Until it is, §6 is documented
and executable but not reproduced. One command settles it:

```bash
python experiments/12_patching_1.5b.py      # ~1-2 h on a T4
python tools/verify_results.py              # writes docs/RESULT_VERIFICATION.md
```

Five parameters were not recoverable and are reconstruction choices, each
exposed as a flag: which five phrasings, which three placebo pairs, bootstrap
resample count, bootstrap unit, floating-point precision. If the rerun's peaks
land near +0.065 (L4, fluency) and +0.074 (L5, confidence) with the same
positive layers 3–8, the reconstruction is faithful. If they do not, the
reconstruction choices are the first place to look — not the paper.

---

## 3. Transcription discrepancies — found and corrected

Three numbers in the paper disagreed with the committed source file. The source
file was correct in each case, confirmed by recomputing from its own
`before`/`after` fields. `analysis/master.json` and `paper/paper.md` have been
corrected to match the artifact, and the figures and tables regenerated.

| Quantity | Was | Now | Source |
|---|---|---|---|
| Health literacy reduction, original run | 87.9% | **88.077%** | recomputed −88.077% from `finetune_v2_ACCEPTED.json` |
| Confidence reduction, original run | 94.3% | **94.370%** | recomputed −94.370% |
| Perplexity after, original run | 12.04 | **12.0237** | `quality_guard.doctor_perplexity_after` |
| Drift, original run (Appendix F table) | 0.00260 | **0.002632** | `quality_guard.neutral_answer_drift_bits` |

No other reported value moved. `analysis/master.json` now agrees with
`results/finetune_v2_ACCEPTED.json` on every per-dimension value and on
perplexity; this is checked programmatically.

---

## 4. Outputs that were never saved

These experiments ran, their numbers are in the paper, and no artefact
survives. Re-running them is not new science — it is putting a file behind
numbers that currently have none.

| Missing file | Script | Paper claims with no artefact | GPU time |
|---|---|---|---|
| `exp_E_objective_comparison.json` | `experiments/10_hard_target_ablation.py` | Entropy 2.024 / 2.022 / 0.278; top-1 0.604 / 0.607 / 0.936; *p*>.99 fraction 12.7 / 12.4 / 68.8%; teacher entropy 1.985; −24.2% vs +19.2%; drift 0.3025; final train loss 3.22e-4 — **Table 5 and Figures 7–9 in full** | ~35 min |
| `exp_A_seed1.json`, `exp_A_seed2.json`, `exp_A_seed3.json` | `experiments/07_selfdistill_seeds.py` | 91.5% mean, 90.9–91.8% range, all seeded per-dimension values, seeded perplexities and drifts — **most of Table 4 and Figure 5** | ~30 min each |
| `exp_B_raw_instructions.json` | `experiments/04_cue_operationalisation.py` | Three arms 5/5 positive, *p* = .0312, medians 2.5× / 2.6× / 3.7×; the contamination audit (75.3 / 32.2 / 21.6%); Spearman ρ = +0.50 — **Table 2, Figure 3, Appendix G** | ~25–35 min |
| `exp_B2_placebo_sensitivity.json` | `experiments/05_placebo_robustness.py` | Family vs flat table; the independent reimplementation matching to mean \|diff\| 0.02 — **Table 3, Figure 4** | ~10 min |
| `positive_control.json` | `experiments/03_positive_control.py` | The four conventions 0.199 / 0.234 / 0.212 / 0.153 — **Appendix D** | ~5 min |
| `exp_D5_probe.json` | `experiments/11_probe.py` | Base AUC up to 0.884; the 0.107 AUC bound; cue-vs-final +0.064 — **Table 6, Figure 10, Appendix J** | ~20 min |

Suggested order: `10` → `07` ×3 → `04` → `05` → `03` → `11`. That is roughly
3 hours of T4 time and it closes every gap in this section.

`experiments/11_probe.py` additionally needs a LoRA adapter. A retained Drive
backup of `dialense_lora` has since been recovered and validated against the
reported configuration, so retraining is not required merely to supply an
adapter. The historical `_adapter_lost` field in `finetune_v2_ACCEPTED.json` is
stale and is preserved only as historical metadata. See
`docs/ADAPTER_RECOVERY.md`. Adapter weights are gitignored and should be obtained
from the separate Hugging Face release once published.

---

## 5. Device memory — claim restated, instrumentation added

The paper previously reported an approximate **5 GiB peak training memory**
footprint. That figure was a design annotation in the training script
(`MAX_MEM_NOTE`), backed by a preflight check that aborts if less than 5 GiB of
device memory is free — not a measured peak. `torch.cuda.reset_peak_memory_stats()`
was called but never read back.

Resolved two ways: §4.1 and Appendix E now describe it as a **requirement** the
script enforces, and state that peak allocation was not recorded;
`experiments/06_selfdistill.py` now reads `torch.cuda.max_memory_allocated()`
back and writes it to its result file as `peak_memory_gib`, so the next run
produces a real measurement.
