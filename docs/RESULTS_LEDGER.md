# What we actually have — 2 Sep 2026

Status of every experiment. **Solid** = replicated or controlled well enough to
write down. **Single run** = the number is probably right but is one observation.
**Unresolved** = do not write anything from it yet.

---

## SOLID — write these

### 1. The style effect exists, and the baseline replicates exactly
Swapping a speaker-background sentence moves the next-token distribution more
than a length- and embedding-matched irrelevant sentence, on all five dimensions,
8/8 phrasings each, p = .0039.

| dimension | ratio | D_style | D_placebo |
|---|---|---|---|
| health_literacy | 7.4× | 0.00604 | 0.00082 |
| confidence | 5.0× | 0.00452 | 0.00090 |
| communication_style | 4.9× | 0.00888 | 0.00181 |
| emotional_expressiveness | 4.7× | 0.00543 | 0.00115 |
| fluency | 3.6× | 0.00513 | 0.00144 |

`EXP_B2` reimplemented this from scratch and reproduced it to **mean |difference|
= 0.02** across all five dimensions, with its own positive control at 0.14745
against v6's 0.147. That is a genuine internal replication and it is worth a
sentence in the paper.

Magnitude: 3.1–6.0% of a full clinical-history swap.

### 2. The effect survives every way of writing the cue
`EXP_B`, three operationalisations, unit = dimension, n = 5, exact sign test:

| arm | dimensions positive | p | median ratio |
|---|---|---|---|
| your descriptors | 5/5 | .0312 | 2.5× |
| mechanical 2nd→3rd person | 5/5 | .0312 | 2.6× |
| verbatim corpus instruction | 5/5 | .0312 | 3.7× |

p = .0312 is the floor at n = 5. **Do not** compare percentages across arms — raw
instructions are 73–136 tokens against 11–15 for the descriptors, so the ratio is
the comparable quantity and the percentage is not.

### 3. Ratios are robust in sign, sensitive in magnitude
Same style sentences, same scenarios, only the placebo builder differs:

| dimension | family-constrained | flat pool |
|---|---|---|
| fluency | 3.6× | 2.5× |
| health_literacy | 7.4× | 4.9× |
| confidence | 5.0× | 3.3× |
| emotional_expressiveness | 4.7× | 4.8× |
| communication_style | 4.9× | 4.7× |

Every dimension stays positive under both (7–8 of 8 phrasings, p ≤ .0117), but
three of five shrink by about a third. State the family constraint in Methods.

### 4. The RAW arm's size tracks content contamination
Three of five corpus instructions contain clinical words on one side only, and
they are the three largest effects — 75.3%, 32.2%, 21.6% of a content swap. The
two clean dimensions are the two smallest, 12.4% and 9.9%. Report the audit table
and this correlation; it is the honest caveat on that arm.

### 5. Cue operationalisation reorders the dimensions
Spearman ρ between descriptor and raw dimension ratios: **+0.50, p = .39, n = 5.**
Do not claim a stable ordering of dimensions by sensitivity.

### 6. The positive control, defined and measured
Prefixed (style + placebo present, only the history swapped):

| scenario set | bits |
|---|---|
| baseline set (first 20) | **0.1475** |
| held-out 20 | **0.1992** |
| training 30 | 0.2120 |
| bare facts, no prefix — wrong convention | 0.2341 |

Use the one matching the measurement's own scenario set. It varies 23% across
sets, so percentages carry that uncertainty.

### 7. Style is more readable at the cue tokens than at the final position
Base model, replicated across three separate probe runs: cue-position AUC exceeds
final-position AUC by roughly **+0.07** on average through layer 20, reversing at
layers 24–27.

This is about the **base model**, so no adapter claim depends on it. It is direct
evidence in your own model that final-position-only measurement under-detects a
distributed feature (Geva et al. 2023; Tigges et al. 2024) — which is what
licenses your reading of the patching null instead of leaving it an excuse.

---

### SOLID (added after EXP_E): the objective comparison

| model | entropy | top-1 | p>.99 | style gap | change | drift |
|---|---|---|---|---|---|---|
| base | 2.024 | 0.604 | 0.127 | 0.00942 | — | — |
| teacher | 1.985 | — | — | — | — | — |
| soft-target KL | 2.022 | 0.607 | 0.124 | 0.00080 | −91.6% | 0.0026 |
| hard-target CE | 0.278 | 0.936 | 0.688 | 0.01123 | +19.2% | 0.3025 |

Soft-target entropy sits between the teacher's floor and the base model — the
mass-covering prediction, confirmed. Hard-target entropy collapses 7.3x with 68.8%
of positions above p = 0.99.

Across two hard-target runs at different learning rates the style gap moved
**−24.2% then +19.2%** — opposite signs — while neutral drift was **0.302 then
0.3025**. The damage reproduces; the fairness metric does not reproduce in sign.
Claim: hard targets destroy the instrument, not that they fake an improvement.

Hard-target drift = 1.29x a full content swap, against the bare-prompt control
(0.2353), which is the form-matched denominator for a measurement taken on
prompts with no style line.

## SINGLE RUN — probably right, not yet defensible as stated

### 8. Self-distillation reduces the measure by ~92%
Held-out phrasings × held-out scenarios, one seed. Perplexity on real clinician
turns 12.23 → 12.04. Neutral-prompt drift 0.0026 bits.
**Needs `EXP_A` (2 more seeds) before "91.6%" is a number you can write.**

### 9. Selectivity
Clinical sensitivity unchanged (+2.7%). Placebo sensitivity fell 75.4%. Style
influence relative to clinical effect 5.22% → 0.43%, a 12.2× reduction.
The 75.4% is a real limit on the claim: the intervention disregards non-clinical
framing generally, not style specifically. One run.

### 10. MOVED TO SOLID — see the ledger entry below

### 11. Activation patching (1.5B)
Style-specific restoration exceeds the matched contrast at layers 3–8: fluency
+0.065 (L4), confidence +0.074 (L5), health literacy +0.027 (below the ±0.03
floor). Layers 10–19 go the other way, to −0.355. Layers 20–26 reach significance
for everything at trivial magnitudes and are not interpreted.
The 0.5B layer-14 result did **not** replicate. Report the failed replication.

---

## RESOLVED AND CLOSED

### 12. RESOLVED — erasure vs suppression: a bounded null

`EXP_D5`, cluster bootstrap over 15 held-out (dimension, phrasing) units for
style and 18 held-out scenarios for urgency, plus a cluster permutation test.

**No layer survives both a CI excluding zero and a permutation test at p < .05.**

> A linear probe recovers the style dimension from the base model's residual
> stream at up to 0.88 AUC. After the intervention, no layer shows a reduction in
> style decodability larger than **0.107 AUC** (95% cluster bootstrap, 15 units).
> The intervention changes behaviour without a representational signature
> detectable at this resolution.

That is a finding, not a failure: it matches Lee et al. (ICML 2024) on DPO and
toxicity, and Galichin et al. (EACL 2026) on instruction tuning. Report the bound
and the unit count so the reader can see the resolution.

The urgency control is honest but weak — this corpus has 50 scenarios and only 18
urgent, so a scenario-level control can never exceed 18 units. Report it as an
underpowered check. The load-bearing selectivity evidence is behavioural
(clinical sensitivity unchanged at +2.7%), and you already have it.

Four earlier versions of this experiment were discarded for analysis errors: a
content label computed on disjoint scenario sets, a control at ceiling, an
item-level bootstrap over clustered data, and a cluster key that collapsed five
dimensions into one unit. Only D5 is valid.

### 13. Conversation-level transfer: uninformative
The baseline friction gap was −0.025 against absolute levels near 1.55 —
indistinguishable from zero and pointing the wrong way. There was no gap to
narrow. CI an order of magnitude wider than the effect. Appendix, one paragraph.

---

## NOT RUN YET

| # | what | gets you |
|---|---|---|
| `EXP_A` ×2 | seed variance | makes #8 quotable |

About 1 hour of T4 time — two seed runs.

---

## The paper you can currently write

A real, controlled, replicated measurement of speaker-style sensitivity in a
language model's output distribution, with an interpretable scale; evidence that
it survives three ways of operationalising the cue; an intervention that removes
most of it without measurable loss of clinical responsiveness; a failure mode
showing the obvious alternative objective produces a false positive on the same
metric; and a mechanistic section that bounds rather than localises, with an
independently replicated reason why final-position measurement under-detects.

What you cannot yet write: a stable headline percentage, an entropy mechanism for
the ablation, and anything about conversations or patient outcomes. The representational
question is answered as a bound, above.
