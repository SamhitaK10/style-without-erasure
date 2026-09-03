# Archive

Superseded code, kept deliberately. Nothing here is runnable as-is and nothing
here backs a number in the paper. Each file is retained because its header
documents a specific error, and four of them document the *same* error four
different ways.

**Do not run these.** Use `experiments/` instead.

---

## `probe/` — four failed probe runs

The paper's §5.6 claim is a **bounded null**: no layer shows a reduction in
style decodability that survives both a cluster-corrected bootstrap interval
excluding zero and a cluster permutation test. Reaching a claim that careful
took five attempts, and the four that failed are here because each one failed
in a way that is easy to repeat.

| File | What went wrong |
|---|---|
| `EXP_D_probe.py` | The content-control label was `int(si < len(scen)//2)` — "is this in the first half of the scenario list" — computed **separately** on the train and eval splits. Those are disjoint scenario sets, so the label meant something different in each. Content AUC came out at or below chance (0.207–0.506): the control was measuring nothing, so the reported style drop had no alternative explanation ruled out. |
| `EXP_D2_content_control.py` | Fixed the label, but **changed the item set at the same time** — 12 varying scenarios (~360 eval items) became 2 fixed histories (~60). Its null is an underpowered null, not a refutation. Worse, the two histories chosen were maximally different (a neonate with congestion vs a type 2 diabetic), so the control sat at ceiling: AUC 1.000 → 1.000, zero-width intervals. A control with no headroom cannot detect degradation. |
| `EXP_D3_final.py` | Bootstrapped **items** on clustered data. The 640-item content eval set came from only 8 held-out scenarios; all 80 items sharing a scenario share a label and are not independent. Intervals were understated by roughly √80 ≈ 9×. Corrected, +0.202 [+0.162, +0.251] becomes about [−0.20, +0.60]. This is pseudoreplication — the same error the baseline analysis was careful to avoid by treating phrasing rather than scenario as the unit. |
| `EXP_D4_final.py` | Fixed the bootstrap, then set the cluster key to the **phrasing index alone**. Phrasing index 5 exists in all five dimensions, so five distinct cue pairs collapsed into one cluster: 3 units instead of 15. With 3 units the permutation test has 2³ = 8 sign patterns and its smallest reachable *p* is about 0.125 — which is why every style *p* came back 0.228, 0.498, 0.749 or 1.000. Those are the only values that test can emit. It was not a null; it was a test with no power. |

The fix in `experiments/11_probe.py` is `unit = (dimension, phrasing)` → 15
held-out units, plus averaging each cue's activations within a scenario for the
content probe and regularising harder (C = 0.05).

`SALVAGE_D4.py` and `SAVE_D5.py` are one-off rescues written after a
`json.dump` crashed on an in-memory namespace. They depend on undefined globals
and cannot run standalone. `SALVAGE_D4.py` sets `ll_late = float("nan")`,
i.e. writes a placeholder into a result field — a reason not to trust anything
it produced.

---

## `early/` — pre-paper iterations

From the earlier framing of this work, before it became a standalone paper.

| File | Note |
|---|---|
| `dialense_v5.py` | Predecessor of the baseline. Required both **style** sides to tokenize to equal length, which is the wrong constraint — only the *placebo pair* needs to match the style pair's lengths. That one error dropped 13 of 25 phrasings and deleted the fluency dimension entirely. Fixed in `experiments/01_baseline.py`. |
| `dialense_final.py`, `colab_v2.py`, `colab_v3.py`, `colab_full.py` | Earlier measurement iterations. |
| `dialense_finetune.py` | The **first intervention attempt**: cross-entropy against hard token targets. Training loss fell to 1e-4; the output distribution collapsed; neutral-prompt drift hit 0.302 bits. Kept because it is a documented negative result and the origin of §5.7. Its saved output is `results/finetune_attempt1_REJECTED.json`. |
| `dialense_patching.py` | First patching attempt. `Qwen/Qwen2.5-0.5B-Instruct`, no placebo control. |
| `step1_attention.py` | Attention-weight analysis. Abandoned: attention says where the model *looked*, not what changed the answer. |
| `step2_scale.py` | The attention test at scale, with bootstrap intervals and Benjamini–Hochberg correction. **The only script in this whole project with a real CLI** (`argparse`, `--data`, `--selftest`, `--limit`) and a self-test that verifies the statistics invent no signal. Not used by the current paper, but it is the model for what `experiments/` should look like — see `docs/REPRODUCIBILITY.md` items 1 and 8. It consumes `pairs.jsonl`, which is not committed here. |
