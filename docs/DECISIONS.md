# Settled decisions — carry these into the writeup

Updated 2 Sep 2026, after `FIX_positive_control`, `EXP_B`, `EXP_B2`.

## 1. The positive control is computed per scenario set, not once globally

Measured, prefixed (style + placebo prefix present, only the history swapped):

| scenario set | value | use it for |
|---|---|---|
| first 20 (baseline set) | **0.1475 bits** | every baseline number |
| held-out 20 | **0.1992 bits** | intervention, drift, anything measured on held-out items |
| training 30 | 0.2120 bits | — |
| bare facts, no prefix | 0.2341 bits | **do not use** — different contrast |

I earlier said to standardise on 0.199 everywhere. That was wrong. Numerator and
denominator must come from the same items, and the baseline was measured on the
first 20 scenarios. Two denominators, each matched to its measurement, is the
correct treatment; one denominator applied across sets is not.

**Methods sentence:** "The positive control replaces the entire clinical history
while holding the style and placebo sentences and the reference continuation
fixed. It is computed on the same scenario set as the measurement it scales:
0.147 bits on the baseline set and 0.199 bits on the held-out set. All
divergences are additionally reported as a percentage of it, because divergence
in bits has no natural scale and a null result cannot otherwise be distinguished
from an insensitive instrument."

**Also report:** the control varies 0.147–0.212 across scenario sets — 23% of its
own value. Percentages inherit that; do not quote them to three significant
figures.

**Consequences:** published baseline percentages (3.1–6.0%) are correct as they
stand. Hard-target neutral drift is **1.52×** a full content swap (0.302 / 0.199),
not the 2.05× implied by the old 0.147 denominator.

## 2. The baseline replicates exactly under an independent reimplementation

`EXP_B2` was written from scratch and reproduces v6 to two decimal places:
mean |FAMILY − v6| = **0.02** across all five dimensions, with the positive
control at 0.14745 against v6's 0.147. All five 8/8 phrasings positive, p = .0039.

This is a real internal replication and worth one sentence in the paper. It also
means the v6 pipeline is deterministic and the published numbers are sound.

## 3. Ratio magnitude is somewhat placebo-construction dependent

Two builders, same style sentences, same scenarios, 8 phrasings:

| dimension | v6 / FAMILY | FLAT | change |
|---|---|---|---|
| fluency | 3.6× | 2.5× | −31% |
| health_literacy | 7.4× | 4.9× | −34% |
| confidence | 5.0× | 3.3× | −34% |
| emotional_expressiveness | 4.7× | 4.8× | +2% |
| communication_style | 4.9× | 4.7× | −4% |
| **median** | **4.9×** | **4.7×** | −4% |

FAMILY draws both placebo sentences from the same stem family; FLAT draws from
one pool of 16 stems, so a pair can span families and sits further apart
semantically — mean placebo divergence rises 1.30×.

**Direction and significance are robust**: every dimension stays positive under
both builders, 7–8 of 8 phrasings, all p ≤ .0117. **Magnitude is not fully
robust**: three of five ratios fall by about a third.

The script's printed verdict said "ROBUST" because it thresholded the mean
absolute difference at 1.5 and FLAT came in at 1.11. That threshold was mine and
it is too generous. The honest reading is: robust in sign, sensitive in magnitude.

**What to do:** report FAMILY as primary — it is the protocol the baseline used —
and FLAT as a robustness row. State the family constraint explicitly in Methods;
it is load-bearing and no reader would guess it. Pre-empting Selvam et al. (2023)
this way is cheaper than being caught by it.

## 4. EXP_B's DESC arm was underpowered by design, not a failed replication

DESC in `EXP_B` used one phrasing per dimension, matched to the raw arm's n = 1,
and gave ratios of 1.0–6.2×. At 8 phrasings the same pipeline gives 2.5–4.9×.
So the collapse was single-phrasing noise, confirmed. Say so; do not present the
n = 1 DESC numbers as a baseline.

## 5. The style effect survives every operationalisation

All three arms 5/5 dimensions positive, sign test p = .0312 (the floor at n = 5).
Median ratios: DESC 2.5×, CONV 2.6×, RAW 3.7×.

**Do not report RAW's percentage-of-control as a larger effect.** Raw instructions
are 73–136 tokens against DESC's 11–15 — a ~10× longer manipulated span, and
sensitivity scales with token-change rate (Yang et al. 2026). The ratio is the
comparable quantity across arms; the percentage is not.

## 6. RAW effect size tracks content contamination — report this

| dimension | clinical words on one side only | RAW % of control |
|---|---|---|
| communication_style | yes | 75.3% |
| fluency | yes | 32.2% |
| health_literacy | yes | 21.6% |
| emotional_expressiveness | none | 12.4% |
| confidence | none | 9.9% |

The three contaminated dimensions are the three largest effects; the two clean
ones are the two smallest. `communication_style`'s low instruction embeds a long
narrative example, which a length-matched irrelevant paragraph does not absorb —
75% of a full content swap is not a style effect.

Report the audit table and this correlation. It is the honest caveat on the RAW
arm and it costs nothing to state.

## 7. Cue operationalisation changes the dimension ordering

Spearman ρ between DESC and RAW dimension ratios: **+0.50, p = .39, n = 5**.
Which dimension looks largest depends on how the cue is written — Tonneau et al.
(2026) appearing in our own data. Report it; do not claim a stable ordering of
dimensions by sensitivity.


---

## 8. Table 3 is in, and the hard-target story changed for the better

`EXP_E`, held-out phrasings x held-out scenarios:

| model | entropy (bits) | top-1 p | frac p>.99 | style gap | change | neutral drift |
|---|---|---|---|---|---|---|
| base | 2.024 | 0.604 | 0.127 | 0.00942 | — | — |
| teacher (blind base) | 1.985 | — | — | — | — | — |
| soft-target KL | **2.022** | 0.607 | 0.124 | 0.00080 | **−91.6%** | 0.0026 |
| hard-target CE | **0.278** | 0.936 | **0.688** | 0.01123 | **+19.2%** | 0.3025 |

Final hard-target training loss 3.22e-04.

**The soft-target prediction held exactly.** Entropy 2.022 against the base's
2.024 and the teacher's 1.985 — forward KL is mass-covering, so the teacher's
entropy is a floor, and the student sits just above it. Top-1 probability and the
near-deterministic fraction are also unchanged (0.607 vs 0.604; 0.124 vs 0.127).
The intervention removed 91.6% of the style gap while leaving the output
distribution distributionally indistinguishable from the base model. That is the
mechanism, stated in numbers rather than asserted.

**The hard-target collapse, quantified.** Entropy falls 7.3x, from 2.024 to 0.278.
Top-1 probability rises to 0.936. The fraction of positions above p = 0.99 goes
from 12.7% to **68.8%**. This is what memorisation looks like measured rather
than described.

### The claim is now stronger than "it produces a false positive"

The first hard-target run (lr 1e-4) reported the style gap **falling 24.2%**.
This run (lr 3e-5) reports it **rising 19.2%**. Opposite signs.

Meanwhile the damage reproduces almost exactly: neutral drift 0.302 bits in the
first run, 0.3025 here.

So: **the degradation is reproducible to three decimal places and the fairness
metric does not reproduce even in sign.** Once the output distribution collapses,
JS divergence between two near-one-hot distributions is dominated by whether they
happen to peak on the same token, so the sensitivity number stops measuring
sensitivity at all. Do not claim hard targets produce an apparent improvement —
claim that they destroy the instrument, and that the resulting number is
uninterpretable in either direction. Report both runs as the evidence.

### The drift denominator is the bare control, not the prefixed one

Drift is measured on prompts containing **no style line**, so its matched
positive control is the bare-facts swap: 0.2353 bits. Hard-target drift is
therefore **1.29x a full content swap**, not the 1.52x I said earlier. The
principle in section 1 extends: match the denominator to the prompt *form* as
well as the scenario set. Soft-target drift is 0.01x.
