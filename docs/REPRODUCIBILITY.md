# Reproducibility status

A living record of what a stranger can and cannot reproduce from this
repository. Updated 3 September 2026. Score below is from the release audit.

---

## Honest summary

The analysis path is clean: all 11 figures and 6 tables regenerate
deterministically from `analysis/master.json`, and no number is typed into
plotting code. The experiment path is not: the scripts were written as
single-cell Colab pastes, `master.json` was assembled by hand, six experiment
outputs were never saved, and one reported analysis has no script.

Nothing below is a rerun of *new* science. It is all about putting artefacts and
interfaces behind results that already exist.

---

## Release audit score

| Dimension | Score | Why |
|---|---|---|
| Repository organization | 2 / 2 | Fixed by this layout |
| Installation | 2 / 2 | `requirements.txt` lists only packages the code imports, with compatibility floors and a written account of which versions were and were not recoverable |
| Data provenance | 2 / 2 | `data/README.md` and `docs/THIRD_PARTY.md` trace the full chain, licences and attribution; the fetch is pinned to a specific DiaLense commit; the S1–S10 gap is stated precisely |
| Experiment commands | 0 / 2 | No CLI anywhere. See item 1 |
| Saved configs | 0 / 2 | No config files; hyperparameters are module constants |
| Random seeds | 1 / 2 | Correct design (split fixed at `random.Random(0)`, run seed varies), now documented per experiment in `EXPERIMENT_PROVENANCE.md` — but the run seed is changed by editing source, and two runs did not set one at all |
| Result traceability | 1 / 2 | Seven outputs committed and matching, checked automatically by `tools/verify_results.py` (38 EXACT, 4 within tolerance, 0 mismatches); six raw outputs still missing; §6's script is a reconstruction not yet executed |
| Figures | 2 / 2 | Deterministic from `master.json` |
| Documentation | 2 / 2 | This directory, `EXPERIMENT_PROVENANCE.md`, `THIRD_PARTY.md` and the README |
| Licensing | 2 / 2 | MIT code / CC BY 4.0 data, attribution carried |

**15 / 20**, up from 7 / 20 before the reorganisation and 13 / 20 before the
provenance and environment pass. The remaining 5 points are items 1–3 below.

---

## Open items, in priority order

### 1. Give every experiment a CLI — worth 4 points (the largest remaining gap)

Every script in `experiments/` is a Colab-cell paste: no `argparse`, no
`__main__` guard, hyperparameters as module-level constants, output paths
hardcoded to Google Drive or Kaggle. `07_selfdistill_seeds.py` requires editing
`SEED = 1` in the source between runs — a manual step disguised as a constant.

What to add to each script:

- `argparse` with at least `--seed`, `--out-dir`, `--model`, `--n-scenarios`
- an `if __name__ == "__main__":` guard
- the resolved config dumped to JSON next to every result file
- `--out-dir` defaulting to `./results`, replacing the
  `/content/drive/MyDrive/...` and `/kaggle/{input,working}` paths

That single change moves *experiment commands*, *saved configs*, and *random
seeds*, and makes the README's command table literally true rather than
descriptive.

### 2. Re-run the six unsaved experiments — worth 1 point

See [`../results/MISSING.md`](../results/MISSING.md) §4 for the file list,
the claims each backs, and the suggested order. About 3 hours of T4 time.

### 3. Run the reconstructed 1.5 B activation-patching script — worth part of 1 point

`experiments/12_patching_1.5b.py` now exists. The original was never saved and
could not be found anywhere, so this is a reconstruction; its header names the
evidence behind every recovered parameter and the five that were not
recoverable. Its statistics layer passes `--selftest` against the published
Benjamini & Hochberg (1995) example and against synthetic curves with planted
effects, independently of any model.

It has not been run against the 1.5 B model. One command settles whether §6
reproduces:

```bash
python experiments/12_patching_1.5b.py && python tools/verify_results.py
```

### 4. Environment — done, with a stated limit

`requirements.txt` now lists exactly the packages the code imports, with
compatibility floors and Python 3.10+. Exact versions from the original runs are
**not recoverable**: Colab supplied the stack and no lockfile was saved. One
earlier audit note was wrong and is corrected here — the `peft==0.14.0` line in
`experiments/06_selfdistill.py` is a *recovery path* that installs 0.14.0 and
exits if the torchao guard fails, not the version the reported run used. There
is no version contradiction between the scripts.

### 5. Pin the DiaLense reference — done

`tools/fetch_scenarios.py` is pinned to DiaLense commit
`a1adecdd31fa6905583f7beb79e58eb4b062bc06`, verified to yield exactly 50
scenarios, and now errors rather than warns on a wrong count. The pin does not
claim to be the commit the original experiments ran against — DiaLense has a
single squashed commit, so its history cannot establish that.

The six experiment scripts still clone the default branch inline. Until they
call this tool instead, run it first if you need a pinned extraction.

### 6. Add provenance to `master.json`

Give each block a `_source` field naming the file or run it came from, and a
`_transcribed: true` flag where no file exists. Then add a checker that fails
if a `_source` file is absent. Right now `master.json` is the paper's single
source of truth and nothing records where its numbers came from.

### 7. Finish the shared-library extraction

`src/style_erasure/cues.py` holds the 40 cue pairs, extracted verbatim. **The
experiment scripts still carry their own inline copies — six in total** — and
the placebo builder exists in three copies. Until they import from `src/`, the
extraction is documentation rather than deduplication, and editing one copy
without the others is a live correctness risk in a paper whose result is
explicitly sensitive to how the placebo is built.

Modules to finish: `placebo.py` (family-constrained and flat builders),
`divergence.py` (JS divergence over the 24-token reference), `scenarios.py`
(loader for `data/scenarios.jsonl`), `stats.py` (cluster bootstrap, cluster
permutation, exact sign test, Wilcoxon).

### 8. Self-test the statistics

`archive/early/step2_scale.py` — the only script in the project with a real CLI
— has a `--selftest` that verifies its bootstrap and BH path invents no signal.
`stats.py` deserves the same. The paper's methodological contribution is about
units of analysis, and four probe runs in `archive/probe/` failed on exactly
that; a test proving the cluster bootstrap does not understate intervals would
be a genuine asset rather than boilerplate.

---

## Known discrepancies between paper and committed results

None outstanding. Three were found in Table 4's original-run row, all favouring
the committed file; `analysis/master.json` and `paper/paper.md` were corrected to
match the artifact and the figures regenerated. See
[`../results/MISSING.md`](../results/MISSING.md) §3.

---

## What is verified

- All 11 figures and all 6 tables rebuild from `analysis/master.json`.
- `tools/fetch_scenarios.py` extracts exactly 50 scenarios, matching the
  paper's 30 train / 20 held-out split.
- `src/style_erasure/cues.py` imports and yields 5 dimensions × 8 pairs,
  matching Appendix A.
- No secrets, credentials, tokens, personal email addresses or local user
  paths anywhere in the tree.
- `tools/fetch_scenarios.py` fetches the pinned DiaLense commit from the public
  remote with no credentials and extracts 50 scenarios.
- `CITATION.cff` validates against schema 1.2.0 (`cffconvert --validate`).
- `analysis/master.json` agrees with `results/finetune_v2_ACCEPTED.json` on
  every per-dimension value and on perplexity.
- `tools/verify_results.py` reports 0 mismatches across 63 checked rows and
  writes `docs/RESULT_VERIFICATION.md`.
- `experiments/12_patching_1.5b.py --selftest` passes: Benjamini-Hochberg
  matches the published 1995 Table 1 example to 4dp, and the reporting rule
  behaves correctly on planted positive, planted negative, sub-floor and null
  curves.
- Every `python .../*.py` command in the README references a file that exists;
  the seven that need no GPU were executed.
