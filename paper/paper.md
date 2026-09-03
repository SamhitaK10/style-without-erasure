# Speaker-Style Sensitivity in Language Models: Controlled Measurement, Distribution-Preserving Removal, and What Survives It

**[AUTHOR NAME]**  ·  **[AFFILIATION]**  ·  **[EMAIL]**

*Preprint. Main text ~9,800 words. Code, adapters, prompts, and all raw outputs: [REPOSITORY URL].*

---

## Abstract

Language models change what they say when a speaker is described differently, even when the propositional content of the request is held fixed. Measuring that sensitivity is harder than it appears: divergence between two model outputs has no natural scale, and any two different strings produce some divergence, so an uncontrolled measurement cannot distinguish a style effect from a surface-form effect. We introduce a measurement protocol in which each style contrast is paired with a placebo contrast matched on the exact token length of both sides and on cosine distance in the model's input-embedding space, and in which all divergences are reported against a positive control that replaces the entire propositional content of the prompt. On a corpus of 50 clinical history-taking scenarios and five speaker-style dimensions, style contrasts move a 1.5-billion-parameter instruction-tuned model 3.6× to 7.4× more than matched placebo contrasts (8 of 8 phrasings positive per dimension, one-sided Wilcoxon *p* = .0039), amounting to 3.1–6.0% of a full content swap; an independent reimplementation reproduced these ratios to a mean absolute difference of 0.02. The effect is positive in all five dimensions under three separate operationalisations of the style cue (sign test *p* = .0312 each, the floor at *n* = 5), though its magnitude depends on how the placebo pool is constructed and on how the cue is written.

We then remove the behavioural effect by self-distillation against a *style-blind teacher*: the frozen base model's own full output distribution on the cue-free prompt supervises a LoRA-adapted student on the cue-present prompt. Across three seeded runs this reduces held-out style sensitivity by a mean of 91.5% (range 90.9–91.8), with perplexity on genuine clinician text improving slightly (12.23 → 12.01–12.08) and drift on style-neutral prompts of 0.0023–0.0026 bits, 0.01× the matched positive control. The student's output distribution is statistically indistinguishable from the base model's: entropy 2.022 bits against 2.024, mean top-1 probability 0.607 against 0.604, and 12.4% of positions above *p* = .99 against 12.7%.

Two negative results constrain what this means. First, a linear probe recovers the style dimension from the base model's residual stream at up to 0.884 AUC, and after the intervention **no layer shows a reduction in style decodability that survives both a cluster-corrected bootstrap and a cluster permutation test**; with 95% confidence no layer's drop exceeds 0.107 AUC across 15 held-out units. Behavioural influence falls by an order of magnitude while linear decodability is preserved at the resolution we can measure. Second, the same objective implemented with hard token targets collapses the output distribution — entropy 0.278 bits, 68.8% of positions above *p* = .99 — and produces a style-gap change of −24.2% in one run and +19.2% in another while neutral-prompt drift reproduces at 0.302 and 0.3025 bits. The damage is reproducible to three decimal places; the fairness metric does not reproduce in sign. We argue this is the more general finding: hard-target fine-tuning does not fake an improvement so much as destroy the instrument, after which the sensitivity number is uninterpretable in either direction. Our principal limitation is scope: one model at one size, five predefined style dimensions expressed as written descriptors rather than enacted speech, and a measurement taken over next-token distributions rather than full generated dialogue.

---

## 1  Introduction

A conversational language model that answers the same question differently depending on how the asker writes is exhibiting a property that is easy to name and difficult to measure. The property matters. Systems built on these models are being deployed to take clinical histories, triage requests, and mediate access to services, and their value rests on eliciting the same information from every user. A system that asks fewer questions, or commits earlier to an answer, when the user writes less fluently reproduces a documented failure of human professional communication — at scale, and without the corrective of a person who can notice it happening.

The measurement problem is the bottleneck. Suppose we change one sentence describing a speaker and observe that the model's output distribution moves. Three explanations are available and only one is interesting. The output may have moved because the *style* changed; because the two sentences are simply *different strings* of different lengths and lexical content; or because the model is generally brittle to any perturbation of its prompt, a property documented at large magnitude by the prompt-sensitivity literature. Without a control that isolates the second and third explanations, a reported style effect is unfalsifiable. And even with such a control, a divergence expressed in bits carries no scale: 0.005 bits is neither large nor small until it is compared with something.

This paper is organised around taking that problem seriously and then acting on the measurement it yields.

**Content must be separated from style, and style from surface form.** Our prompts are built so that the propositional content — the clinical history — is byte-identical across the two sides of every comparison, verified by hash. What differs is a single sentence describing how the speaker communicates. Against each such style contrast we place a *placebo contrast*: a pair of clinically irrelevant sentences matched to the style pair on the exact token length of each side and on cosine distance between the two sides in the model's input-embedding space. If the model's output moves for the style pair and equally for the placebo pair, then "these are different strings" explains everything we have seen. The reported effect is the difference. Concurrent work has since shown that this class of control is not optional: re-analysed under magnitude-matched baselines, most published counterfactual-prompting effects in a comparable medical setting do not survive.

**Effects need a denominator.** We express every divergence as a percentage of a positive control that replaces the entire clinical history while holding the style and placebo sentences fixed. This converts an uninterpretable quantity into a statement of the form "changing how the speaker talks moves the model *n*% as much as changing what is medically wrong with them." It also makes a null result meaningful: without such a scale, a measurement of zero cannot be distinguished from an insensitive instrument.

**An intervention judged only by a falling sensitivity metric is not judged at all.** Any procedure evaluated by a sensitivity measure can satisfy that measure by reducing the model's sensitivity to everything — including things it should remain sensitive to. Conventional quality guards do not catch this. We show a concrete case in which perplexity remains within tolerance while the model's style-blind behaviour is the quantity that has moved, and in which the sensitivity metric itself becomes non-reproducible in sign. We therefore treat a *selectivity* check — does the model still respond to what it should respond to? — and a *distributional* check — is the output distribution still the shape it was? — as part of the evaluation rather than as supplementary material.

**Preserving the output distribution is a first-class requirement.** A model whose predictive distribution has collapsed onto single tokens is not a fairer model; it is a broken one that happens to score well. We make this measurable by reporting entropy, mean top-1 probability, and the fraction of positions above *p* = .99 for every trained variant, alongside the fairness metric it was trained to improve.

### Contributions

1. **A controlled measurement framework for speaker-style sensitivity** (§3, §4), combining a magnitude-matched placebo contrast, an interpretable positive-control denominator, position counterbalancing, and phrasing rather than scenario as the unit of analysis. We report the effect at 3.6–7.4× the matched control across five style dimensions and 3.1–6.0% of a full content swap (§5.1, Table 1, Figure 2).

2. **Evidence that the effect replicates across dimensions, phrasings, cue operationalisations, and an independent reimplementation** (§5.1–5.3), together with an honest account of what does *not* replicate: the magnitude of the ratio depends on placebo construction, and the rank ordering of dimensions by sensitivity does not survive a change in how the cue is written (Spearman ρ = +0.50, *p* = .39, *n* = 5).

3. **A self-distillation intervention against a style-blind teacher** — the frozen base model's own distribution on the cue-free prompt — reducing held-out style sensitivity by a mean of 91.5% (range 90.9–91.8 across three seeds) at 1.18% of parameters, with perplexity and neutral behaviour intact (§5.4–5.5, Table 4, Figures 5–6). We position this explicitly as *context distillation run backwards* and as concurrent with a same-skeleton method published in the prompt-injection literature (§2.4).

4. **Evidence that style remains linearly decodable after the intervention**, with no layer surviving a cluster-corrected significance test and a stated resolution bound of 0.107 AUC (§5.6, §6, Table 6, Figures 10–11). We report this as a bound rather than as proof of preservation.

5. **Evidence that hard-target supervised fine-tuning collapses the output distribution and renders the sensitivity metric non-reproducible in sign** (§5.7, Table 5, Figures 7–9). We argue the correct description is instrument destruction, not a false positive, and that this generalises to any intervention scored by a sensitivity metric.

---

## 2  Related Work

### 2.1  Prompt sensitivity and the null-effect problem

Language models are known to be strongly sensitive to prompt properties that carry no semantic content. Systematic search over semantically equivalent formats produces accuracy spreads of tens of points [Sclar et al., 2024], and minimal edits — an added space, a greeting — change a substantial fraction of predictions [Salinas and Morstatter, 2024]. Benchmark rankings are unstable across human-written paraphrases of the same instruction [Mizrahi et al., 2024], which motivates treating phrasing rather than item as the unit of analysis, as we do throughout. Prompt-sensitivity indices define the relevant baseline directly: the relative change in response likelihood under an intent-preserving substitution, a quantity uncorrelated with task performance [Chatterjee et al., 2024].

This literature is the strongest objection to any style-sensitivity claim, and we treat it as such rather than as background. Our response is structural: the placebo arm *is* an intent-preserving substitution matched in magnitude to the style substitution, so the reported effect is what remains after generic prompt sensitivity is subtracted. The most direct statement of this requirement is recent and independent of us [Yang et al., 2026], showing that gender-swapped medical prompts flip predictions at essentially the rate that plain paraphrase does, and that only a handful of previously published demographic-sensitivity results in that setting survive a magnitude-matched baseline. That work also arrives at per-sample Jensen–Shannon divergence as the appropriate metric, converging with our choice. We differ in the perturbation studied — sociolinguistic style rather than demographic substitution — and we note in §8 that our placebo is a matched *irrelevant sentence* rather than a token-matched *paraphrase of the style sentence*, so we meet part but not all of that standard.

### 2.2  Style, dialect, and interactional bias in conversational systems

That models respond differently to speakers who write differently is established. Matched-guise probing with parallel dialect texts shows models assigning worse outcomes on dialect alone, with the effect surviving alignment training that suppresses the corresponding overt judgments [Hofmann et al., 2024]. Studies across many English varieties, with native-speaker annotation, find non-standard varieties receiving more stereotyping and worse comprehension [Fleisig et al., 2024]; the pattern replicates in other languages with parallel dialect corpora [Bui et al., 2025]. Reasoning benchmarks constructed as parallel standard/dialect pairs show accuracy degradation exceeding what matched typo injection produces [Lin et al., 2025] — an early and important use of a null perturbation arm in this literature, and a reason our placebo idea is not novel in kind.

A parallel line studies the user rather than third parties: bias in how a system treats the person it is speaking to. The framing paper for this area [Abeliuk et al., 2026] enumerates user-signal types and identifies writing style and dialect as underexplored relative to names and explicit self-identification [see also Eloundou et al., 2025]. In the clinical setting, perturbing patient messages along tonal and syntactic axes while holding content fixed shifts treatment recommendations and produces reduced-care errors [Gourabathina et al., 2025a], and the sensitivity exceeds that of human clinicians reading the same material [Gourabathina et al., 2025b]; separately, varying communication framing across a large vignette set changes triage urgency by tens of percentage points [Omar et al., 2026].

Two recent results counsel caution and we adopt both. Operationalising the same demographic group four different ways — names, dialect translation, real conversation history, explicit descriptors — yields disparities that reverse direction depending on the cue [Tonneau et al., 2026], so a measured effect is partly a property of the operationalisation. And in some settings conversation *topic* predicts model behaviour far more strongly than user demographics do [Neplenbroek et al., 2026]. Our §5.2 is a direct response to the first: we test three cue operationalisations and report that the dimension ordering does not transfer between them.

Our difference from all of this work is the readout. These studies reduce to a downstream label — accuracy, refusal, triage tier, rater score. We measure the divergence of the output distribution itself under a matched control, which is sensitive at magnitudes where a discrete outcome has not yet flipped, and which does not saturate.

### 2.3  Persona, style, and speaker attributes inside the network

Interpretability work has established that persona-like and stylistic attributes are represented in ways amenable to linear methods. Trait directions extracted from natural-language descriptions can monitor and steer character traits [Chen et al., 2025]; a dominant persona axis recovered by PCA over role vectors is consistent across model families [Lu et al., 2026]. Style vectors computed as mean activation differences steer generated register [Konen et al., 2024]. Sentiment is linearly represented and causally efficacious [Tigges et al., 2024], with the important caveat — directly relevant to us — that it is aggregated at intermediate summary tokens rather than only at the final position. Sociodemographic attributes of a user emerge as linearly decodable subspaces from indirect cues [Bouchaud and Ramaciotti, 2025], and models infer user demographics from stereotypical content even without disclosure [Neplenbroek et al., 2025].

The localisation picture is genuinely contested. Some work reports extremely concentrated control — three attention heads governing persona expression [Izawa et al., 2026], a single direction mediating refusal [Arditi et al., 2024]. Other work reports the opposite: steering a default persona required capping activations across 8–16 simultaneous layers, with single-layer interventions insufficient [Lu et al., 2026]; dialect steering required four causally selected layers [Wu et al., 2026]. Our mechanistic section (§6) sits in this dispute deliberately and modestly: we report a bounded null under one protocol at one position, and we note that the positive results are at head, neuron, or feature resolution, or use multi-position generation-time steering, and are therefore not contradicted by what we measure.

### 2.4  Distillation, soft targets, and invariance training

The soft-target argument originates with knowledge distillation [Hinton et al., 2015]: the relative probabilities a teacher assigns to non-target classes carry structure that a one-hot label destroys. Self-distillation into an identically parameterised student behaves as a regulariser rather than as compression [Furlanello et al., 2018], and repeated self-distillation progressively sparsifies the solution [Mobahi et al., 2020] — a theoretical shadow of the over-suppression we observe in §5.4.

The closest machinery is context distillation [Askell et al., 2021; Snell et al., 2022], which minimises the KL divergence between a frozen pretrained model conditioned on a fixed context and a fine-tuned model without that context, absorbing prompted behaviour into the weights. **Our objective is that equation with the context moved to the student's side**: rather than internalising a context so the prompt can be dropped, we neutralise one so an additional span becomes inert. We describe the method as *context distillation run backwards* and claim no novelty for the machinery.

We must also distinguish a concurrent method with the same asymmetry. In a prompt-injection defence published shortly before this work [Peng et al., 2026], a LoRA student processes the attacked input while the frozen initialisation model — the same base model — scores those tokens under the clean, injection-removed input. That is our teacher/student asymmetry in a different domain. The differences that remain are real but should be stated precisely rather than overclaimed: their signal is on-policy, converting a teacher/student log-ratio into a per-token advantage under an RL objective, whereas ours is teacher-forced forward KL over the full vocabulary at every position of a fixed reference; forward KL is mass-covering, which is why the teacher's entropy acts as a floor for our student and is the mechanism behind §5.7. Their removed span is adversarial and there exists a ground-truth correct behaviour; ours is a benign attribute for which the base model's own blind behaviour is the target by definition. And they do not run a selectivity evaluation, because their threat model does not require one.

Classical debiasing-by-distillation is the other relevant ancestor: distilling from a teacher whose distribution has been rescaled by a separately trained bias model improves out-of-distribution robustness without an in-distribution cost [Clark et al., 2019; Utama et al., 2020]. Our teacher requires no bias model at all — it is obtained by input ablation on the same network. Symmetric consistency objectives with self-generated anchors [Hejabi et al., 2026] are the closest 2026 neighbour and the natural baseline for future comparison. Counterfactual logit pairing [Garg et al., 2019] and counterfactual invariance [Veitch et al., 2021] supply the definitional backbone; both are symmetric, and neither has a frozen anchor, which is precisely what prevents both branches from drifting to a degenerate constant.

Finally, the soft-versus-hard target literature explains §5.7 in advance. Cross-entropy against a one-hot target keeps being minimised long after accuracy saturates [Guo et al., 2017]; label smoothing improves calibration but erases the within-class similarity structure that makes a soft target informative [Müller et al., 2019]; and instruction tuning degrades calibration, with smoothing scaling badly as vocabulary grows [Huang et al., 2025] — an argument for using a real teacher distribution rather than a smoothed one-hot. What has not been shown, to our knowledge, is that this predictable collapse renders a *fairness* metric non-reproducible in sign.

### 2.5  Behaviour, representation, and what fine-tuning actually changes

A body of work now separates what a model *contains* from what it *uses*. Debiased word embeddings still cluster by the removed attribute [Gonen and Goldberg, 2019]; the metric was satisfied and the geometry was not. Direct preference optimisation does not delete toxicity-writing MLP vectors but learns an offset that steers activations away from them, leaving the capability reactivatable [Lee et al., 2024]. Fine-tuning on procedurally defined tasks learns a minimal wrapper over pretrained capability rather than altering it [Jain et al., 2024]. Base-model sparse autoencoders reconstruct instruction-tuned activations comparably well, indicating that fine-tuning shifts the distribution over existing features rather than creating new ones [Galichin et al., 2026]. Recent work argues specifically that low-rank updates redistribute rather than eliminate [Basani and Chhabra, 2026] — the critique aimed most directly at a LoRA-based method such as ours.

Concept-erasure methods define the alternative standard: iterative nullspace projection [Ravfogel et al., 2020] and closed-form linear erasure [Belrose et al., 2023] aim at representations rather than behaviour. Amnesic probing [Elazar et al., 2021] supplies the framework for separating the two directions, showing that decodability and use come apart. Our §5.6 is a measurement in this frame, and our §6 states the direction we tested and the direction we did not.

We also inherit the caution that safety-relevant subspaces may not be linearly separable from task-useful ones at all [Ponkshe et al., 2026], which means a failure to erase can be the correct outcome rather than a defect — but only for a study that measured it.

### 2.6  Evaluation-metric validity

Intrinsic bias metrics do not reliably correlate with application-level bias [Goldfarb-Tarrant et al., 2021]; fairness benchmarks contain pervasive validity failures [Blodgett et al., 2021]; and trivial, bias-irrelevant perturbations of benchmark items move measured bias scores as much as genuine debiasing does [Selvam et al., 2023]. A survey of debiasing techniques finds effectiveness inconsistent across metrics with most methods degrading language modelling ability [Meade et al., 2022]. This literature is the reason our paper reports a distributional sanity check beside every fairness number, and the reason we treat §5.7 as a contribution rather than as a discarded ablation.

---

## 3  Problem Formulation

### 3.1  Objects

Let $M$ be a language model inducing a next-token distribution $p_M(\cdot \mid x)$ over a vocabulary $V$ for a prompt $x$.

A **scenario** $F \in \mathcal{F}$ is a block of propositional content — in our corpus, a patient's clinical history — expressed as a fixed string. A **style cue** is a pair $(s^+, s^-)$ of single sentences describing how a speaker communicates along one dimension $d$, differing in the level of that dimension and in nothing else that we can control. A **placebo cue** is a pair $(q^+, q^-)$ of sentences that are irrelevant to the content and to the speaker's manner.

A prompt is assembled as the concatenation of a style sentence, a placebo sentence, and a scenario, in one of two orders:

$$\begin{gathered}x_{\text{first}}(s,q,F) = s \Vert q \Vert F, \\[2pt] x_{\text{second}}(s,q,F) = q \Vert s \Vert F .\end{gathered}$$

Both orders are evaluated and averaged, so that any effect of a manipulated sentence's *position* in the prompt is balanced across conditions.

### 3.2  The measurement

For each scenario we generate a **reference continuation** $r(F) \in V^{T}$ of $T = 24$ tokens by greedy decoding from the content-only prompt, so that the reference is by construction what the model would say knowing only the content. We then measure how far each manipulation moves the model's distribution *over the positions of that fixed reference*.

Writing $p_t(x)$ for the model's distribution at position $t$ of the reference given prompt $x$, define

$$\mathcal{D}(x, x') \;=\; \frac{1}{T}\sum_{t=1}^{T} \mathrm{JS}\big(p_t(x)\,\Vert\,p_t(x')\big),$$

with Jensen–Shannon divergence taken in bits. JS is bounded, symmetric, and defined when supports differ, which matters because one arm of §5.7 produces near-degenerate distributions.

**Style sensitivity** for dimension $d$, cue pair $k$, scenario $F$ is the divergence induced by swapping the style sentence with the placebo sentence held fixed:

$$\begin{aligned}D^{\text{style}}_{d,k}(F) \;=\; \tfrac{1}{2}\!\!\sum_{\pi \in \{\text{first},\text{second}\}}\!\! \mathcal{D}\big(&x_\pi(s^+_{d,k}, q^+_{d,k}, F), \\[-1pt] &x_\pi(s^-_{d,k}, q^+_{d,k}, F)\big).\end{aligned}$$

**Placebo sensitivity** is the divergence induced by swapping the placebo sentence with the style sentence held fixed:

$$\begin{aligned}D^{\text{plac}}_{d,k}(F) \;=\; \tfrac{1}{2}\!\!\sum_{\pi}\!\! \mathcal{D}\big(&x_\pi(s^+_{d,k}, q^+_{d,k}, F), \\[-1pt] &x_\pi(s^+_{d,k}, q^-_{d,k}, F)\big).\end{aligned}$$

The **effect** is their difference, $\Delta_{d,k}(F) = D^{\text{style}}_{d,k}(F) - D^{\text{plac}}_{d,k}(F)$, and the **normalised ratio** is $R_d = \bar{D}^{\text{style}}_d / \bar{D}^{\text{plac}}_d$, with bars denoting means over cue pairs and scenarios.

### 3.3  Why the placebo must be magnitude-matched

$D^{\text{plac}}$ is only a valid control if the placebo swap is *the same size of perturbation* as the style swap. Sensitivity to a prompt edit scales with how much of the prompt is edited, so a placebo pair that is shorter, or semantically closer together, than the style pair will understate the baseline and inflate $R_d$. We therefore require, for every cue pair:

$$\begin{gathered}|q^+| = |s^+|, \qquad |q^-| = |s^-|, \\[2pt] \big|\,\cos\text{-}d(q^+,q^-) - \cos\text{-}d(s^+,s^-)\,\big| \to \min,\end{gathered}$$

where $|\cdot|$ is length in tokens under the model's own tokenizer and $\cos\text{-}d$ is cosine distance between mean input embeddings. The first two constraints make the *change in prompt length* identical in the style and placebo conditions, so it cancels in $\Delta$. The third makes the *semantic size* of the swap as close as the pool allows. §5.3 measures how much the result depends on this construction, and finds that it does.

### 3.4  Why the positive control sets the scale

$\Delta$ is in bits, and bits have no natural reference. We therefore define the **positive control** as the divergence produced by replacing the entire propositional content while holding the style and placebo sentences and the reference continuation fixed:

$$\mathcal{P} \;=\; \mathop{\mathbb{E}}_{F \neq F'} \Big[\mathcal{D}\big(s \Vert q \Vert F,\; s \Vert q \Vert F'\big)\Big],$$

and report every divergence additionally as a percentage of $\mathcal{P}$. This answers "how large is the style effect compared with changing the medicine, measured the same way," and — because it is the same measurement machinery — it distinguishes a genuine null from an insensitive instrument.

Two properties of $\mathcal{P}$ require care and we state both. It depends on the **scenario set**: measured on the baseline set it is 0.1475 bits, on the held-out set 0.1992, and on the training set 0.2120 — a spread of 23% of its own value. And it depends on the **prompt form**: computed without the style and placebo prefix present it is 0.2341 bits, a different contrast entirely. We therefore match the denominator to the numerator on both axes, and report the spread rather than a single point estimate. A **negative control**, swapping a sentence for itself, returns exactly 0 in every configuration and is verified before each run.

### 3.5  The invariance objective

Let $M_0$ denote the frozen base model and $M_\theta$ the same model with a low-rank adapter. Write $x_\varnothing(F)$ for the prompt containing the content only — the **style-blind** input. The training objective is

$$\begin{aligned}\mathcal{L}(\theta) \;=\; \mathop{\mathbb{E}}_{F,\,s \in \{s^+,s^-\}} \; \frac{1}{T}\sum_{t=1}^{T} \mathrm{KL}\Big(\, & p^{M_0}_t\big(x_\varnothing(F)\big) \;\Big\Vert\; \\[-1pt] & p^{M_\theta}_t\big(s \Vert F\big)\Big).\end{aligned}$$

The teacher is the frozen model on the *cue-free* prompt; the student is the adapted model on the *cue-present* prompt; the target is a full distribution over $V$, not a token. Three properties follow. The objective requires no external labels and no judgment about how a system *ought* to adapt to a speaker — the base model's own blind behaviour is the target by definition. Because the teacher is the base model with the adapter disabled, it costs no additional memory. And because forward KL is mass-covering, the teacher's entropy acts as a floor on the student's: the objective cannot be minimised by becoming confident. §5.7 is the empirical demonstration that this property is load-bearing rather than decorative.


---

## 4  Experimental Setup

### 4.1  Model

All experiments use **Qwen2.5-1.5B-Instruct** (28 transformer layers, hidden size 1536), loaded from the public checkpoint with its own tokenizer. Measurement passes use float32; training uses a float16 base with float32 adapter parameters, mixed-precision forward passes, gradient scaling, and gradient checkpointing. The training script requires at least 5 GiB of free device memory and aborts otherwise; peak allocation during training was not recorded. Every prompt is rendered through the model's chat template with the system message *"You are a physician taking a patient history. Ask one follow-up question."* and the assembled prompt as the user turn.

### 4.2  Corpus and scenarios

Scenarios are drawn from a corpus of matched clinical history-taking conversations spanning 50 distinct clinical situations. From each conversation record we extract the latent fact list and render it as a single content block of the form `Reported history: <fact>; <fact>; …`. Because the same block is used on both sides of every comparison, propositional content is identical by construction, not by paraphrase. Content identity was verified by hash for all pairs.

We emphasise that the clinical material is the *testbed*, not the claim. Nothing in the measurement or the intervention is specific to medicine; the corpus is used because it supplies matched scenarios with verified content identity and a natural positive control.

### 4.3  Style dimensions and cue phrasings

Five dimensions are studied: **fluency**, **health literacy**, **confidence**, **emotional expressiveness**, and **communication style**. For each dimension we use **8 phrasings**, each a pair of third-person descriptor sentences differing in the level of that dimension — for example, *"The patient speaks fluent, grammatical English as a first language."* versus *"The patient speaks limited, ungrammatical English as a second language."* All 40 pairs are listed verbatim in Appendix A.

Writing these descriptors is an experimenter degree of freedom, and §5.2 is the test of whether it matters.

### 4.4  Placebo construction

Placebo sentences are generated from a template grammar: a stem describing a clinically irrelevant circumstance of the visit (`The patient arrived by bus`, `The patient booked this visit online`, …) optionally extended by 1–3 modifier phrases (`on a cloudy morning`, `in unusually heavy traffic`, …). Enumerating stems against modifier combinations yields a pool bucketed by exact token length, spanning 5–25 tokens.

For each style pair $(s^+, s^-)$ we search the pool for the pair $(q^+, q^-)$ with $|q^+| = |s^+|$, $|q^-| = |s^-|$ minimising $|\cos\text{-}d(q^+,q^-) - \cos\text{-}d(s^+,s^-)|$. Two builders are compared in §5.3:

- **family-constrained** — both placebo sentences must come from the same stem family (three families: mode of arrival, waiting behaviour, booking method), which keeps the pair semantically close in kind;
- **flat** — the pair is drawn from a single pool of 16 stems and may span families.

Under the family-constrained builder the achieved residual mismatch in cosine distance is 0.0022–0.0049 (mean across dimensions).

For the long-form cue arms of §5.2, whose sides run to 73–136 tokens, single sentences cannot match; we assemble paragraphs of irrelevant sentences summing to *exactly* the required token count by a coin-change search over the length-bucketed pool, then select among valid assemblies by the same cosine criterion.

### 4.5  Cue operationalisations

§5.2 compares three ways of expressing the same style contrast:

1. **Author-written descriptor** — the third-person sentences of §4.3.
2. **Mechanical second-to-third-person rewrite** — the corpus's own style instruction, transformed by *pronoun substitution only*. Because *they* shares *you*'s verb paradigm, no verb form is ever altered and no subject–verb agreement error can be introduced; quoted example dialogue is masked and left verbatim; a fixed frame supplies the antecedent. We verified zero residual second-person tokens across all ten instructions.
3. **Verbatim corpus instruction** — the original second-person instruction inside an identical frame on both sides.

The corpus contains **exactly one** style instruction per (dimension, variant), so arms 2 and 3 have $n = 1$ phrasing per dimension. To keep the comparison fair, arm 1 is restricted to one phrasing in this experiment; the eight-phrasing results of §5.1 remain the primary baseline.

### 4.6  Data splits

The split is fixed with a dedicated random seed and held constant across every training run, so that seed variation measures initialisation and data order rather than split luck.

- **Scenarios**: 30 training, 20 held out.
- **Phrasings**: 5 training, 3 held out, per dimension.
- All post-intervention numbers are computed on **held-out phrasings crossed with held-out scenarios** — an item the model has seen in neither respect.

### 4.7  Training

A LoRA adapter [Hu et al., 2022] of rank 16, α = 32, dropout 0.05 is attached to all attention and feed-forward projections (`q,k,v,o,gate,up,down`), giving **18,464,768 trainable parameters, 1.18% of the model**. Optimisation uses AdamW at learning rate $3\times10^{-5}$ for 2 epochs with gradient accumulation of 8 and gradient-norm clipping at 1.0. Each training item contributes both the high-style and low-style prompt, each distilled toward the same style-blind teacher distribution. A drift guard aborts training if divergence on style-neutral prompts exceeds 0.05 bits; it never fired in the soft-target runs. Three runs differ only in the seed governing adapter initialisation, data order, and dropout.

The **hard-target ablation** replaces the objective with token-level cross-entropy against the reference continuation, holding every other hyperparameter fixed. Two configurations are reported, at learning rates $1\times10^{-4}$ and $3\times10^{-5}$.

### 4.8  Probing

Residual-stream activations are extracted at layers 2, 4, 6, 8, 12, 16, 20, 24, 27, at two readouts: the **final prompt position**, and the **mean over the cue-token positions**, located by subsequence search over token ids rather than by string matching, which is unreliable after chat templating.

Two probes are trained, each with the split appropriate to its label:

- **Style probe** — logistic regression classifying high versus low style. Split by **phrasing** (5 train / 3 held out per dimension), so the probe cannot memorise cue wording. The unit of analysis is the (dimension, phrasing) pair: **15 held-out units**.
- **Content probe** — logistic regression classifying urgent/emergent versus routine/soon, using the corpus's own urgency annotation. Split by **scenario** (18 train / 18 held out, balanced), so the probe cannot memorise a history. Activations are averaged within a scenario before fitting, and regularisation is strengthened (C = 0.05), because 18 independent units against 1536 dimensions otherwise memorise. The unit of analysis is the scenario: **18 held-out units**.

Features are standardised on the training split only. Both probes are run with the adapter engaged and bypassed on identical items.

### 4.9  Statistical treatment

**Unit of analysis.** Scenarios vary content, not style wording, and are therefore items rather than independent evidence about style. The unit is the phrasing for §5.1 (*n* = 8 per dimension), the dimension for §5.2–5.3 (*n* = 5), the seeded run for §5.4 (*n* = 3), and the cluster defined above for §5.6. We state this explicitly because treating scenarios as units would inflate every interval in this paper by roughly the square root of the cluster size.

**Tests.** §5.1 uses a one-sided Wilcoxon signed-rank test across phrasings; at *n* = 8 the smallest attainable *p* is 1/256 = .0039, so we report the number of phrasings positive alongside it. §5.2–5.3 use an exact one-sided sign test across dimensions; at *n* = 5 the floor is .0312. §5.6 uses a **cluster bootstrap** — resampling whole units with replacement, never items — and a **cluster permutation test** that exchanges the base/intervened assignment for entire units, giving $2^{15}$ sign patterns for the style probe.

**Effect sizes and intervals.** Ratios and percentages of the positive control are reported in preference to standardised effect sizes, which are not interpretable for bounded, skewed divergences. Bootstrap intervals use 2,000–4,000 resamples at the percentile level. For §5.4 we report **mean and range across seeds and decline to construct a confidence interval at *n* = 3**, where a three-point standard deviation is not a meaningful quantity.

**Pre-specification.** The magnitude floor of ±0.03 used to interpret the activation-patching curves (§6.2) was fixed before those analyses were run.

---

## 5  Results

### 5.1  Speaker-style sensitivity exists, and replicates

Swapping the style sentence moves the model's next-token distribution substantially more than swapping a matched placebo sentence, on every dimension tested (Table 1, Figure 2).

Ratios range from **3.6× (fluency) to 7.4× (health literacy)**. Every dimension has **8 of 8 phrasings positive**, giving a one-sided Wilcoxon *p* of **.0039** — the floor at this sample size — with bootstrap intervals over phrasings excluding zero throughout. The negative control returns exactly 0 and the positive control 0.1475 bits on this scenario set, so the instrument is behaving.

In absolute terms the effect is small: **3.1% (confidence) to 6.0% (communication style)** of the divergence produced by replacing the patient's entire clinical history. We regard the modest size as a result rather than a weakness. It is a quantity that was previously unmeasurable in these units, and stating it honestly is the point of building a denominator.

Position audit shows the same sign in both prompt slots with the first slot approximately 1.5× the second, which is why all reported effects average the two orders.

**Internal replication.** An independently rewritten implementation — different placebo generator, different code path, same style sentences and scenarios — reproduced the five ratios to a **mean absolute difference of 0.02**, with its own positive control at 0.1475 bits against the original 0.147. We report this because a measurement of this kind is only as good as its reproducibility, and because §5.3 shows that one implementation choice does move the numbers.

### 5.2  The effect survives every cue operationalisation

Table 2 and Figure 3 compare the three cue forms of §4.5. **All three arms are positive in 5 of 5 dimensions**, giving an exact sign-test *p* of **.0312** in each — again the floor at *n* = 5. Median ratios are 2.5× (descriptor), 2.6× (mechanical rewrite), and 3.7× (verbatim instruction).

Three cautions belong with this result and we state them in the text rather than a footnote.

**Percentages are not comparable across arms.** Raw corpus instructions run 73–136 tokens against 11–15 for the descriptors — an order of magnitude more prompt modified. The verbatim arm reaches up to 75.3% of a full content swap, but that reflects perturbation size, not a larger style effect. The ratio, which is length-controlled within each arm by construction, is the comparable quantity.

**The verbatim arm's magnitude tracks content contamination.** An automated audit of clinical vocabulary appearing on only one side of each cue found asymmetry in three of five dimensions. Those three are the three largest effects (75.3%, 32.2%, 21.6% of a content swap); the two clean dimensions are the two smallest (12.4%, 9.9%). The communication-style instruction, at 75.3%, embeds a long narrative example that a length-matched irrelevant paragraph does not absorb. The full audit is Appendix G. This is the strongest available argument for using controlled style sentences rather than naturally occurring instructions, and we regard it as a contribution of the robustness experiment rather than an inconvenience.

**Dimension ordering does not transfer.** The Spearman correlation between descriptor and verbatim dimension ratios is **ρ = +0.50 (*p* = .39, *n* = 5)**. We therefore make no claim about which style dimension the model is most sensitive to — a caution that recent work on cue operationalisation makes mandatory rather than optional.

### 5.3  Direction is robust to placebo construction; magnitude is not

Table 3 and Figure 4 hold the style sentences, scenarios, and phrasing count fixed and vary only the placebo builder.

Every dimension remains positive under both builders (7–8 of 8 phrasings, all *p* ≤ .0117), and the median ratio barely moves (4.9× → 4.7×). But three of five dimensions fall by about a third — fluency 3.6× → 2.5×, health literacy 7.4× → 4.9×, confidence 5.0× → 3.3× — because a placebo pair permitted to span stem families sits further apart semantically, raising mean placebo divergence by a factor of 1.30.

The honest summary is **robust in sign, sensitive in magnitude**. We report the family-constrained builder as primary because it is the protocol under which the baseline was collected, specify it precisely in §4.4 so it is reproducible, and report the flat-pool values as the robustness row. Given published evidence that trivial perturbations of benchmark construction can move measured bias scores as much as genuine interventions do, we consider stating this dependence more useful than a single number that a reader cannot reconstruct.

### 5.4  Self-distillation reduces held-out style sensitivity, stably across seeds

Training against the style-blind teacher reduces held-out style sensitivity by a mean of **91.5%**, with a range of **90.9–91.8%** across three seeded runs — a spread of 0.9 percentage points (Table 4, Figure 5). The original unseeded run, at 91.6%, falls inside that range.

Stability holds per dimension as well as in aggregate. Across the three seeds, confidence falls 94.0–94.6%, fluency 92.9–93.5%, emotional expressiveness 92.0–92.7%, communication style 90.2–91.6%, and health literacy 87.4–88.4%. Every dimension's cross-seed range is under 1.5 points, and the *ordering* of dimensions by reduction is identical in all three runs. Figure 6 shows the before/after mapping: the largest pre-intervention effect (health literacy, 0.01179 bits) remains the largest afterwards (0.00141 bits), so the intervention scales the profile down rather than reshaping it.

We report mean and range and deliberately do not construct a confidence interval, which at *n* = 3 would convey precision the design cannot support.

### 5.5  Neutral and clinical behaviour remain stable

The intervention is worthless if it achieves invariance by making the model unresponsive. Three guards, specified in advance, say it does not.

**Perplexity on genuine clinician text** *improves* slightly in every run: 12.23 → 12.01, 12.08, and 12.01 for seeds 1–3 (12.02 in the original run). The model has not become worse at clinical language.

**Drift on style-neutral prompts** — divergence between base and adapted model on prompts containing no style sentence at all — is 0.0023–0.0026 bits, or **0.01× the form-matched positive control**. Behaviour in the absence of a style cue is essentially unchanged.

**Selectivity.** Sensitivity to a full clinical-content swap changes by **+2.7%**, i.e. is unchanged; style influence relative to the clinical effect falls from **5.22% to 0.43%**, a 12.2-fold reduction. Sensitivity to the clinically irrelevant placebo contrast also falls, by **75.4%**.

That last number is a genuine limit on the claim and we foreground it rather than bury it. The model did not learn to disregard communication style specifically; it learned to disregard *non-content framing in general* while retaining full sensitivity to clinical content. This is weaker than a style-targeted edit. It is arguably a cleaner mechanism — the resulting model conditions on the medicine and discounts social framing — but it is not what a reader would assume from the headline number, so we state it in the abstract, here, and in §7.

### 5.6  Style remains linearly decodable after the intervention

A linear probe recovers the style dimension from the base model's residual stream at up to **0.884 AUC** (Table 6, Figure 10a). Decodability rises with depth from 0.63 at layer 2 to a plateau above 0.82 from layer 12 onward at the final position, and is consistently **higher at the cue-token positions than at the final position** — by **+0.064 AUC on average through layer 20** — before the ordering reverses at layers 24–27. Shuffled-label controls sit at chance (0.47–0.52) at every layer, so the probe is not overfitting.

After the intervention, **no layer shows a reduction in style decodability that survives both a cluster-corrected bootstrap interval excluding zero and a cluster permutation test at *p* < .05** (Table 6, Figure 10b). The largest point estimate is +0.049 AUC at layer 8, with interval [+0.005, +0.107] and permutation *p* = .064; the late layers, where an earlier item-level analysis suggested a large effect, give +0.011 (*p* = .657) and +0.029 (*p* = .416) at layers 24 and 27.

We state the result as a **bound**: with 95% confidence, no layer's style-decodability drop exceeds **0.107 AUC** at 15 held-out units. The urgency control probe likewise shows no layer surviving both tests, but we report it as underpowered rather than as evidence of selectivity — the corpus contains 50 scenarios of which 18 are urgent, so a scenario-level control cannot exceed 18 units here, and its base AUC is correspondingly unstable. The load-bearing selectivity evidence in this paper is behavioural (§5.5), not representational.

**We do not claim the representation is unchanged.** We claim that behavioural influence fell by an order of magnitude while linear decodability did not fall detectably at the resolution our design affords. §6 develops the distinction.

### 5.7  Hard-target fine-tuning collapses the output distribution and destabilises the metric

Replacing the soft-target objective with token-level cross-entropy against the same reference continuations, holding every other hyperparameter fixed, produces a model that is broken in a way the fairness metric does not report (Table 5, Figures 7–9).

**The distribution collapses.** Output entropy falls from **2.024 to 0.278 bits**, a factor of 7.3. Mean top-1 probability rises from 0.604 to **0.936**. The fraction of positions where the model places more than 0.99 of its mass on a single token rises from 12.7% to **68.8%**. Final training loss reaches $3.22\times10^{-4}$, consistent with the analytic observation that cross-entropy against a one-hot target is minimised at unit confidence and therefore has no stopping point short of memorisation.

**The soft-target model does not.** Its entropy is **2.022 bits** against the base model's 2.024 and the teacher's 1.985; mean top-1 probability 0.607 against 0.604; near-deterministic fraction 0.124 against 0.127. This is the mass-covering property of forward KL made visible: the teacher's entropy is a floor, so the objective cannot be minimised by becoming confident. The soft-target student is distributionally indistinguishable from the base model while its style sensitivity has fallen by 91.6%.

**Neutral behaviour moves more than the medicine does.** Hard-target drift on style-neutral prompts is **0.3025 bits, 1.29× the form-matched positive control** of 0.2341 bits. Changing the objective moved the model's style-blind behaviour further than replacing a patient's entire clinical history moves it. Soft-target drift is 0.0026 bits, 0.011× the same control (Figure 8).

**And the fairness metric stops being reproducible.** Two hard-target runs at different learning rates report a style-gap change of **−24.2%** (lr $1\times10^{-4}$) and **+19.2%** (lr $3\times10^{-5}$) — opposite signs — while neutral-prompt drift reproduces at **0.302 and 0.3025 bits** (Figure 9). The distributional damage replicates to three decimal places; the quantity the intervention was nominally optimising does not replicate even in direction.

We take this to be the more general statement, and it is stronger than the one we set out to make. The problem is not that hard-target training produces a *false positive* on a fairness metric. It is that once the output distribution has collapsed, Jensen–Shannon divergence between two near-one-hot distributions is governed by whether they happen to peak on the same token, so the sensitivity number no longer measures sensitivity at all. **The instrument breaks before the metric does**, and a study reporting only the metric cannot tell which happened. Since perplexity remained within tolerance throughout, the standard quality guard does not detect this; the distributional statistics of Table 5 do, at negligible cost.


---

## 6  Mechanistic Interpretation

### 6.1  Two questions that are easy to conflate

The probe and the behavioural measurement answer different questions, and the value of §5.6 depends entirely on keeping them apart.

- **Representation.** *Does the hidden state contain linearly readable information about the speaker's style?* Answered by the probe: yes, up to 0.884 AUC in the base model, and with no detectable reduction after the intervention.
- **Behavioural use.** *Does changing the style cue alter the model's output distribution?* Answered by the divergence measurement: yes in the base model, and 91.5% less so after the intervention.

These can come apart in both directions. Information can be present and unused; information can be absent while behaviour still varies for other reasons. Amnesic probing established the first direction — decodability does not imply use. Our result is a measurement in the second frame: strong behavioural change without a detectable change in decodability.

**Observation.** No layer shows a post-intervention reduction in linear probe AUC that survives a cluster bootstrap and a cluster permutation test at 15 held-out units; the bound is 0.107 AUC.

**Interpretation.** This is consistent with reduced downstream *use* of style information rather than with detectable representational erasure. It is not proof that the representation is unchanged, and we do not assert that. A linear probe answers a question about linear decodability from a particular readout at particular layers; a rotated or non-axis-aligned encoding [Geiger et al., 2024], a change concentrated at positions we did not read, or a change smaller than our resolution would all be invisible to it.

The pattern we observe matches what has been reported for other alignment interventions. Direct preference optimisation does not delete toxicity-writing vectors but learns an offset that routes activations around them; fine-tuning on procedural tasks learns a removable wrapper over intact capability; and base-model sparse autoencoders reconstruct instruction-tuned activations well enough to suggest that fine-tuning shifts the distribution over existing features rather than creating new ones. That a rank-16 adapter behaves the same way is unsurprising in light of the argument that low-rank updates redistribute rather than eliminate. We regard our contribution here as measuring the thing carefully, with the right unit of analysis and a stated bound, rather than as discovering the phenomenon.

### 6.2  Where the effect is, and why we do not claim to know

We ran layerwise activation patching to ask whether the style effect could be attributed to a specific depth. The donor state from the high-style run replaced the low-style state at the final prompt position, one layer at a time, and we measured the fraction of the output difference restored. Because raw restoration rises toward 1.0 by construction, the primary measure is the **difference between the style curve and the matched placebo curve**, with a pre-specified magnitude floor of ±0.03.

Style-specific restoration exceeds the matched contrast at **layers 3–8**, peaking at **+0.065 for fluency (layer 4)** and **+0.074 for confidence (layer 5)**. Health literacy peaks at +0.027, below the floor. Layers 10–19 run the other way, reaching **−0.355** for health literacy at layer 17. Layers 20–26 reach significance for every dimension at magnitudes of 0.005–0.041; because restoration there approaches 1.0 with minimal variance, arbitrarily small differences acquire tight intervals, and we do not interpret them.

**We report a failed replication.** An earlier analysis on a 0.5-billion-parameter model appeared to show a sharp transition at layer 14. It did not replicate at 1.5 billion parameters, where the large transition occurs at layer 20 and is **shared with the placebo contrast** — identifying the model's general commitment point rather than a style mechanism. We report this because a reader deciding how much to trust single-model interpretability claims should be able to see how one behaved.

The negative mid-network region is, we think, diagnostic of the measurement rather than the model, and §5.6 supplies independent evidence for that reading. Factual recall decomposes into enrichment at subject positions and extraction at the final position [Geva et al., 2023]; sentiment is aggregated at intermediate summary tokens rather than at the end [Tigges et al., 2024]. Both predict that a final-position-only patch under-detects a feature carried elsewhere. **Our own probe shows exactly that in this model**: cue-position readout beats final-position readout by +0.064 AUC on average through layer 20. That is a measurement, not an excuse, and it is about the base model, so no claim about the adapter rests on it.

Accordingly we make **no localisation claim**. What we report is a bounded null about final-position single-layer sufficiency, together with the observation that the large late transition is style-independent. This is compatible with — and does not contradict — work reporting concentrated control at head, neuron, or feature resolution, or using multi-position generation-time steering; those methods answer a different question. It is corroborated by results requiring 8–16 simultaneous layers to steer a persona and four causally selected layers to steer dialect.

---

## 7  Discussion

**What the intervention is, stated precisely.** The method trains a model to produce, in the presence of a style cue, the distribution it would have produced without one. The teacher is the model itself; the supervision is free; no judgment about how a system ought to adapt to a speaker is introduced. It is context distillation with the context moved to the student's side — internalising *ignorance* of a span rather than internalising the span — and it is concurrent with a same-skeleton method in the prompt-injection literature, which we cite rather than claim priority over. Our distinct contributions on the method axis are the invariance framing, the teacher-forced full-distribution forward-KL form and its entropy-floor consequence, and the selectivity evaluation.

**Why soft targets preserve uncertainty, and why that is the point.** Forward KL is mass-covering: matching a teacher that assigns 1.985 bits of entropy requires the student to keep spreading mass, so over-confidence *increases* the loss. Cross-entropy against a one-hot target has the opposite property — it is minimised at unit confidence, so optimisation has no stopping point short of memorisation. §5.7 is that difference made empirical: 2.022 bits versus 0.278, 12.4% versus 68.8% of positions near-deterministic, at otherwise identical hyperparameters. The choice of target is not an implementation detail; it determines whether the resulting model is usable.

**Fairness metrics need distributional sanity checks, and this is cheap.** The general lesson from §5.7 is that any intervention scored by a sensitivity metric can satisfy that metric by reducing sensitivity to everything, and that conventional guards miss it: perplexity stayed in tolerance in the arm where neutral behaviour moved 1.29× a full content swap. Worse, once the distribution has collapsed the metric itself stops being reproducible in sign. Three numbers — entropy, top-1 probability, and the fraction of near-deterministic positions — cost one evaluation pass and would have caught it immediately. We recommend reporting them alongside any distributional fairness metric as routine practice, and we recommend a paired specificity set of the kind that exaggerated-safety benchmarks established for over-refusal [Röttger et al., 2024].

**Linearly decodable information need not drive behaviour.** Figure 11 places the two results side by side: behavioural sensitivity falls by 91.5% while probe AUC is statistically unchanged at every layer. §5.6 is a reminder that "the model still encodes X" and "the model still acts on X" are different claims requiring different evidence. For deployment this is the more important distinction: a system whose behaviour is invariant to style while still representing it is, behaviourally, what was asked for — but it is also one adapter-disable away from the original behaviour, and its invariance has not been shown to be robust to further fine-tuning. For auditing, the implication runs the other way: a probe finding style information in a deployed model is not by itself evidence that the model uses it.

**Implications for model editing.** The result adds to a growing body indicating that parameter-efficient behavioural edits operate by redistribution rather than deletion. If that is general, then editing methods should be evaluated on behaviour *and* on representation, with the unit of analysis stated, because the two dissociate. It also suggests a practical hierarchy: where the requirement is behavioural invariance, a distribution-preserving objective of this kind is cheap and effective; where the requirement is that information be *gone*, a behavioural objective is the wrong tool and concept-erasure methods are the right family.

**Implications for conversational systems in high-stakes settings.** Our measurement lives at the level of next-token distributions. It says that a model's immediate predictive behaviour depends on how a speaker is described, that the dependence can be removed cheaply and stably, and that removal does not damage the model. It does not say that removal improves anything a user would experience, and we ran the transfer experiment that would have addressed this and report it as uninformative (§8, Appendix I). The responsible framing is that this is a controlled property of models, measured well, with an intervention that works on the property measured.

**Implications for alignment evaluation.** Two transfer. First, the unit of analysis determines whether an interval means anything: our own probe analysis produced an apparently significant result that dissolved once intervals were computed over clusters rather than items, and the corrected version is the only one we report. Second, a bounded null is a result. "No layer shows a drop exceeding 0.107 AUC at 15 units" is an informative statement; "no significant difference" is not.

---

## 8  Limitations

**Model family and scale.** Everything reported here is Qwen2.5-1.5B-Instruct at one size with one system prompt. Whether the baseline magnitude, the intervention's efficacy, or the representational null hold at other scales or families is untested, and the mechanistic literature gives specific reason for caution: an apparent layer-14 effect at 0.5B did not survive the move to 1.5B (§6.2).

**Stated, not enacted, style.** Our cues are third-person descriptors of how a speaker communicates, not speakers who actually communicate that way. This buys byte-identical content and matched controls at the cost of construct validity. Work using real annotator populations shows quality differences for genuinely non-native writers, and a model may respond to *enacted* dysfluency differently from a description of it. §5.2 partially addresses this by testing the corpus's own instructions, but those are also written descriptions.

**Five predefined dimensions.** The dimensions were fixed in advance and are not exhaustive; the ordering among them does not transfer across cue operationalisations (ρ = +0.50, *p* = .39), so we make no claim about which matters most.

**Limited scenario set.** Fifty scenarios from one corpus, 20 used for measurement and 20 held out. The positive control varies 23% across scenario sets, so percentages inherit that uncertainty.

**Next-token measurement.** We measure divergence over a fixed 24-token reference continuation. We do not show that a reader would judge two generations different, and we have not run a generation-level human or judge evaluation. This is the gap between our metric and the phenomenon the metric stands for, and it is the cheapest remaining experiment.

**No conversation-level transfer claim.** We attempted one and it was uninformative rather than negative: across 20 held-out scenario pairs run as full simulated consultations, the baseline friction gap was −0.025 against absolute levels near 1.55 — indistinguishable from zero and opposite in sign to the disparity the detectors were built to find — so there was no gap available to narrow, and the interval (−0.113 to +0.125) is an order of magnitude wider than the effect it brackets. Two design choices plausibly explain it: the simulated patient was the base model in both conditions, so style entered only through the scenario instruction rather than the patient's actual language, and 20 of 100 available pairs is too few at this variance. Full details are in Appendix I. **We make no claim of conversation-level improvement.**

**The probe shows decodability, not necessity.** A linear probe answers a question about linear readability from a chosen readout at chosen layers. Attribute information realised in a rotated or non-axis-aligned subspace, carried at positions we did not read, or changed by less than 0.107 AUC would be invisible to us. Probe sensitivity also depends on the classifier and regularisation; ours is logistic regression on standardised features.

**The content control is underpowered.** The urgency probe has at most 18 independent units in this corpus and its base AUC is correspondingly unstable. We report it as an underpowered check; the selectivity evidence we rely on is behavioural.

**Activation patching did not produce stable localisation**, and the negative mid-network region indicates that final-position patching is the limiting factor. Patching at cue-token positions is the indicated follow-up and was not run.

**The hard-target comparison covers two configurations.** Both collapsed; both showed drift near 0.302 bits; their style-gap changes had opposite signs. We claim the collapse is reproducible across the configurations tested and that the metric was not, not that no hard-target configuration could behave differently.

**The intervention is broader than its target.** Placebo sensitivity fell 75.4% alongside the 91.5% style reduction, so the model learned to disregard non-content framing generally. A genuinely style-specific edit was not achieved and may not be achievable by this objective.

**The placebo is not a token-matched paraphrase.** We match on length and embedding distance but our placebo is a different irrelevant sentence rather than a magnitude-matched paraphrase of the style sentence, which is the stricter standard recent work argues for. We meet part of it.

**The adapter is removable.** Disabling the adapter restores the original behaviour exactly. This is a property of every adapter-based intervention and is not a robustness claim in either direction.

---

## 9  Conclusion

Speaker-style sensitivity in a language model's output distribution is real, small, and measurable when the measurement is controlled: 3.6–7.4× a magnitude-matched placebo across five dimensions, 3.1–6.0% of a full content swap, replicated across phrasings, cue operationalisations, and an independent reimplementation.

It can be removed. Self-distillation against the model's own style-blind output distribution reduces held-out sensitivity by a mean of 91.5% (range 90.9–91.8 over three seeds) at 1.18% of parameters, while leaving perplexity, neutral behaviour, and — critically — the shape of the output distribution intact.

What removal does *not* do is erase the representation. Style remains linearly decodable after the intervention, with no layer showing a drop beyond a bound of 0.107 AUC at the resolution our design supports. Behavioural influence and linear decodability dissociate.

And the choice of training target decides whether any of this is meaningful. The same objective with hard token targets collapses the output distribution to 0.278 bits of entropy with 68.8% of positions near-deterministic, moves style-blind behaviour 1.29× more than replacing the entire clinical content does, and yields a style-gap change of −24.2% in one run and +19.2% in another. The damage reproduces; the fairness metric does not reproduce in sign. Interventions of this kind should be reported with the distributional statistics of the resulting model, not with the sensitivity metric alone.


---

## References

Abeliuk, A., Alarcón, V., Sanchez Macias, C., Madariaga, Á., & Lopez, C. (2026). Beyond Third-Person Audits: Situated Interaction Auditing. arXiv:2606.12247.

Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024). Refusal in Language Models Is Mediated by a Single Direction. *NeurIPS 2024*. arXiv:2406.11717.

Askell, A., Bai, Y., Chen, A., Drain, D., Ganguli, D., Henighan, T., et al. (2021). A General Language Assistant as a Laboratory for Alignment. arXiv:2112.00861.

Basani, A. R., & Chhabra, A. (2026). Exposing the Illusion of Erasure in Knowledge Editing for LLMs. arXiv:2606.23276.

Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S. (2023). LEACE: Perfect Linear Concept Erasure in Closed Form. *NeurIPS 2023*. arXiv:2306.03819.

Biderman, D., Portes, J., Gonzalez Ortiz, J., et al. (2024). LoRA Learns Less and Forgets Less. *TMLR 2024*. arXiv:2405.09673.

Blodgett, S. L., Lopez, G., Olteanu, A., Sim, R., & Wallach, H. (2021). Stereotyping Norwegian Salmon. *ACL 2021*.

Bouchaud, P., & Ramaciotti, P. (2025). Linear socio-demographic representations emerge in LLMs from indirect cues. arXiv:2512.10065.

Bui, M. D., Holtermann, C., Hofmann, V., Lauscher, A., & von der Wense, K. (2025). Large Language Models Discriminate Against Speakers of German Dialects. *EMNLP 2025*.

Chatterjee, A., Renduchintala, H. S. V. N. S. K., Bhatia, S., & Chakraborty, T. (2024). POSIX: A Prompt Sensitivity Index For Large Language Models. *Findings of EMNLP 2024*. arXiv:2410.02185.

Chen, R., Arditi, A., Sleight, H., Evans, O., Lindsey, J., et al. (2025). Persona Vectors. arXiv:2507.21509.

Clark, C., Yatskar, M., & Zettlemoyer, L. (2019). Don't Take the Easy Way Out. *EMNLP 2019*.

Dodge, J., Ilharco, G., Schwartz, R., Farhadi, A., Hajishirzi, H., & Smith, N. A. (2020). Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping. arXiv:2002.06305.

Elazar, Y., Ravfogel, S., Jacovi, A., & Goldberg, Y. (2021). Amnesic Probing. *TACL* 9. arXiv:2006.00995.

Eloundou, T., Beutel, A., Robinson, D. G., et al. (2025). First-Person Fairness in Chatbots. *ICLR 2025*. arXiv:2410.19803.

Fleisig, E., Smith, G., Bossi, M., Rustagi, I., Yin, X., & Klein, D. (2024). Linguistic Bias in ChatGPT. *EMNLP 2024*.

Furlanello, T., Lipton, Z., Tschannen, M., Itti, L., & Anandkumar, A. (2018). Born-Again Neural Networks. *ICML 2018*. arXiv:1805.04770.

Galichin, A. V., Korznikov, A., Dontsov, A., et al. (2026). Feature Drift: How Fine-Tuning Repurposes Representations in LLMs. *Findings of EACL 2026*.

Garg, S., Perot, V., Limtiaco, N., Taly, A., Chi, E. H., & Beutel, A. (2019). Counterfactual Fairness in Text Classification through Robustness. *AIES 2019*. arXiv:1809.10610.

Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. (2024). Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations (DAS). *CLeaR 2024*. arXiv:2303.02536.

Geva, M., Bastings, J., Filippova, K., & Globerson, A. (2023). Dissecting Recall of Factual Associations in Auto-Regressive Language Models. *EMNLP 2023*. arXiv:2304.14767.

Goldfarb-Tarrant, S., Marchant, R., Muñoz Sánchez, R., Pandya, M., & Lopez, A. (2021). Intrinsic Bias Metrics Do Not Correlate with Application Bias. *ACL 2021*.

Gonen, H., & Goldberg, Y. (2019). Lipstick on a Pig. *NAACL 2019*. arXiv:1903.03862.

Gourabathina, A., Gerych, W., Hao, Y., & Ghassemi, M. (2025). The Medium is the Message: How Non-Clinical Information Shapes Clinical Decisions in LLMs. *FAccT '25*. DOI 10.1145/3715275.3732121.

Gourabathina, A., Hao, Y., Gerych, W., & Ghassemi, M. (2025). The MedPerturb Dataset. arXiv:2506.17163.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. (2017). On Calibration of Modern Neural Networks. *ICML 2017*. arXiv:1706.04599.

Hase, P., Bansal, M., Kim, B., & Ghandeharioun, A. (2023). Does Localization Inform Editing? *NeurIPS 2023*. arXiv:2301.04213.

Heimersheim, S., & Nanda, N. (2024). How to use and interpret activation patching. arXiv:2404.15255.

Hejabi, A., Rahmati, M., Ziabari, A., & Dehghani, M. (2026). Flip-Flop Consistency: Unsupervised Training for Robustness to Prompt Perturbations. *ACL 2026*.

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531.

Hofmann, V., Kalluri, P. R., Jurafsky, D., & King, S. (2024). AI generates covertly racist decisions about people based on their dialect. *Nature* 633:147–154. arXiv:2403.00742.

Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA. *ICLR 2022*. arXiv:2106.09685.

Huang, et al. (2025). Calibrated Language Models and How to Find Them with Label Smoothing. *ICML 2025*, PMLR v267. arXiv:2508.00264.

Izawa, Y., Minegishi, G., Eguchi, K., Hosokawa, S., & Taura, K. (2026). Steering at the Source: Style Modulation Heads. arXiv:2603.13249.

Jain, S., Kirk, R., Lubana, E. S., et al. (2024). Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks. *ICLR 2024*. arXiv:2311.12786.

Konen, K., Jentzsch, S., Diallo, A., et al. (2024). Style Vectors for Steering Generative Large Language Models. *Findings of EACL 2024*.

Lee, A., Bai, X., Pres, I., Wattenberg, M., Kummerfeld, J. K., & Mihalcea, R. (2024). A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity. *ICML 2024*. arXiv:2401.01967.

Lin, F., et al. (2025). Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks (ReDial). *ACL 2025*. arXiv:2410.11005.

Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J. (2026). The Assistant Axis. arXiv:2601.10387.

Makelov, A., Lange, G., & Nanda, N. (2024). Is This the Subspace You Are Looking For? *ICLR 2024*. arXiv:2311.17030.

Meade, N., Poole-Dayan, E., & Reddy, S. (2022). An Empirical Survey of the Effectiveness of Debiasing Techniques for PLMs. *ACL 2022*. arXiv:2110.08527.

Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. *NeurIPS 2022*. arXiv:2202.05262.

Mizrahi, M., et al. (2024). State of What Art? A Call for Multi-Prompt LLM Evaluation. *TACL* 12.

Mobahi, H., Farajtabar, M., & Bartlett, P. (2020). Self-Distillation Amplifies Regularization in Hilbert Space. *NeurIPS 2020*. arXiv:2002.05715.

Méloux, M., Portet, F., & Peyrard, M. (2026). Mechanistic Interpretability as Statistical Estimation: A Variance Analysis. arXiv:2510.00845.

Müller, R., Kornblith, S., & Hinton, G. (2019). When Does Label Smoothing Help? *NeurIPS 2019*. arXiv:1906.02629.

Neplenbroek, V., Sarti, G., Bisazza, A., & Fernández, R. (2026). Topics as Proxies for Sociodemographics. arXiv:2606.02776.

Neplenbroek, V., Bisazza, A., & Fernández, R. (2025). Reading Between the Prompts. arXiv:2505.16467.

Omar, M., et al. (2026). Impact of Patient Communication Style on Agentic AI-Generated Clinical Advice in E-Medicine. *The American Journal of Medicine*.

Peng, Y., Lian, L., Wagner, D., & Chen, S. (2026). SecOPD: Mitigating Adaptive Prompt Injections by On-Policy Distillation. arXiv:2608.21500.

Ponkshe, K., Shah, S., Singhal, R., & Vepakomma, P. (2026). Safety Subspaces are Not Linearly Distinct. *ICLR 2026*. arXiv:2505.14185.

Qi, X., Panda, A., Lyu, K., et al. (2025). Safety Alignment Should Be Made More Than Just a Few Tokens Deep. *ICLR 2025*. arXiv:2406.05946.

Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2020). Null It Out: Iterative Nullspace Projection. *ACL 2020*. arXiv:2004.07667.

Reusens, M., Borchert, P., De Weerdt, J., & Baesens, B. (2024). Native Design Bias. arXiv:2406.17385.

Rupprecht, J., Ahnert, G., & Strohmaier, M. (2025). Prompt Perturbations Reveal Human-Like Biases in LLM Survey Responses. arXiv:2507.07188.

Röttger, P., Kirk, H. R., Vidgen, B., Attanasio, G., Bianchi, F., & Hovy, D. (2024). XSTest. *NAACL 2024*. arXiv:2308.01263.

Salinas, A., & Morstatter, F. (2024). The Butterfly Effect of Altering Prompts. *Findings of ACL 2024*. arXiv:2401.03729.

Sclar, M., Choi, Y., Tsvetkov, Y., & Suhr, A. (2024). Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design. *ICLR 2024*. arXiv:2310.11324.

Selvam, N. R., Dev, S., Khashabi, D., Khot, T., & Chang, K.-W. (2023). The Tail Wagging the Dog. *ACL 2023*. arXiv:2210.10040.

Shi, C., Beltran-Velez, N., Nazaret, A., et al. (2024). Hypothesis Testing the Circuit Hypothesis in LLMs. *NeurIPS 2024*. arXiv:2410.13032.

Snell, C., Klein, D., & Zhong, R. (2022). Learning by Distilling Context. arXiv:2209.15189.

Tigges, C., Hollinsworth, O. J., Geiger, A., & Nanda, N. (2024). Language Models Linearly Represent Sentiment. *BlackboxNLP 2024*. arXiv:2310.15154.

Tonneau, M., Sehgal, N., Malhotra, K., et al. (2026). Different Demographic Cues Yield Inconsistent Conclusions About LLM Personalization and Bias. arXiv:2601.18486.

Utama, P. A., Moosavi, N. S., & Gurevych, I. (2020). Mind the Trade-off: Debiasing NLU Models without Degrading the In-distribution Performance. *ACL 2020*.

Veitch, V., D'Amour, A., Yadlowsky, S., & Eisenstein, J. (2021). Counterfactual Invariance to Spurious Correlations. *NeurIPS 2021*. arXiv:2106.00545.

Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., & Shieber, S. (2020). Investigating Gender Bias in Language Models Using Causal Mediation Analysis. *NeurIPS 2020*. arXiv:2004.12265.

Wu, H., Emami, A., Hassan, M. F., et al. (2026). LLMs Silently Correct African American English. arXiv:2607.06845.

Yang, Z., Levy, M., Goldberg, Y., & Wallace, B. C. (2026). Compared to What? Baselines and Metrics for Counterfactual Prompting. arXiv:2605.01048.

Yun, H. S., Kapoor, G., Mackert, M., Kouzy, R., Xu, W., Li, J. J., & Wallace, B. C. (2026). This Treatment Works, Right? LLM Sensitivity to Patient Question Framing in Medical QA. arXiv:2604.05051.

Zhang, F., & Nanda, N. (2024). Towards Best Practices of Activation Patching in Language Models. *ICLR 2024*. arXiv:2309.16042.

---

## Appendix

### A  Style cue phrasings

All 40 phrasing pairs (5 dimensions × 8 phrasings) are released verbatim in the repository at `scripts/EXP_A_seeds.py`, where they are the literal `CUES` dictionary used by every experiment in this paper. The first pair of each dimension, used as the single-phrasing arm of §5.2:

| Dimension | High-style side | Low-style side |
|---|---|---|
| Fluency | The patient speaks fluent, grammatical English as a first language. | The patient speaks limited, ungrammatical English as a second language. |
| Health literacy | The patient is college-educated and has high health literacy. | The patient is grade-school-educated and has low health literacy. |
| Confidence | The patient describes symptoms confidently, without hedging or uncertainty. | The patient describes symptoms hesitantly, hedging with constant uncertainty. |
| Emotional expressiveness | The patient speaks emotionally, openly expressing fear and distress. | The patient speaks unemotionally, flatly withholding fear and distress. |
| Communication style | The patient answers directly, giving only the information requested. | The patient answers indirectly, giving long stories around questions. |

### B  Placebo construction

Stems: *arrived by bus · travelled by train · came by taxi · returned by tram · drove by car · walked from home · waited in reception · sat in the lobby · stood near the entrance · rested in the seating area · booked this visit online · booked this visit by phone · scheduled this visit online · arranged this visit by phone · parked in the visitor lot · signed in at the front desk*, each prefixed *The patient*.

Modifiers: *on a cloudy morning · on a sunny afternoon · during a quiet weekday · after a short wait · with a relative present · earlier than scheduled · from the north side of town · from the south side of town · carrying a folder of paperwork · in unusually heavy traffic · just before the doors opened · later than originally planned · today · yesterday*, with 0–3 sampled per sentence.

The family-constrained builder partitions the first ten stems into three families (arrival mode, waiting behaviour, booking method) and requires both placebo sentences to come from one family. Achieved cosine residuals per dimension: fluency 0.0047, health literacy 0.0046, confidence 0.0022, emotional expressiveness 0.0049, communication style 0.0046.

### C  Second-to-third-person conversion

Pronoun substitution only. Because *they* shares *you*'s verb paradigm, no verb form is altered and no agreement error can be introduced. Quoted spans are masked by a regular expression that requires the opening quote not to follow a letter and the closing quote not to precede one, so apostrophes inside contractions do not break the match. A fixed frame — *The patient is described as follows.* — supplies the antecedent and is byte-identical on both sides. Object-position pronouns after a small closed list of verbs and prepositions are mapped to *them*. Verified: **zero residual second-person tokens outside quoted spans across all ten instructions**.

### D  Positive controls

| Prompt form | Scenario set | Value (bits) |
|---|---|---:|
| style + placebo prefix present | baseline set (first 20) | 0.1475 |
| style + placebo prefix present | held-out set (20) | 0.1992 |
| style + placebo prefix present | training set (30) | 0.2120 |
| bare content, no prefix | held-out set (20) | 0.2341 |

The denominator is matched to the numerator on **both** axes. Baseline percentages in Table 1 use 0.1475; drift multiples in §5.7 use 0.2341, because drift is measured on prompts containing no style line. The negative control (swapping a sentence for itself) returns exactly 0 in every configuration.

### E  Training configuration

| Setting | Value |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct (28 layers, hidden 1536) |
| Adapter | LoRA r=16, α=32, dropout 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 18,464,768 (1.18% of model) |
| Optimiser | AdamW, lr 3e-5, 2 epochs |
| Gradient accumulation | 8 |
| Gradient clipping | norm 1.0 |
| Precision | fp16 base, fp32 adapter, GradScaler, gradient checkpointing |
| Device memory | at least 5 GiB free, required by the training script; peak allocation not recorded |
| Reference length | 24 tokens, greedy decode from the content-only prompt |
| Drift abort threshold | 0.05 bits (never triggered in soft-target runs) |
| Split seed | fixed at 0 for all runs |
| Run seeds | 1, 2, 3 (init, data order, dropout) plus one unseeded original run |
| Hardware | single NVIDIA T4, ~30 min per training run |

### F  Seed-by-seed and per-dimension results

| Run | Mean reduction | Perplexity | Drift (bits) | Health literacy | Confidence | Comm. style | Emotional expr. | Fluency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original (unseeded) | 91.600% | 12.2345 → 12.0237 | 0.002632 | 88.077% | 94.370% | 91.711% | 92.342% | 93.214% |
| seed 1 | 90.939% | 12.2345 → 12.009 | 0.002621 | 87.358% | 93.982% | 90.204% | 91.987% | 92.891% |
| seed 2 | 91.721% | 12.2345 → 12.081 | 0.002632 | 88.286% | 94.523% | 91.496% | 92.417% | 93.451% |
| seed 3 | 91.816% | 12.2345 → 12.013 | 0.002342 | 88.397% | 94.571% | 91.583% | 92.672% | 93.392% |

Teacher entropy was 1.9850 bits in every run. Pre-intervention style gap by dimension: Fluency 0.00777, Health literacy 0.01179, Confidence 0.00821, Emotional expressiveness 0.00915, Communication style 0.01021; mean 0.00942 bits.

### G  Raw-instruction contamination audit

Clinical vocabulary appearing on exactly one side of each corpus instruction, from a fixed lexicon applied identically to all five dimensions:

| Dimension | High-side only | Low-side only | Verbatim-arm effect |
|---|---|---|---:|
| Communication style | days | symptoms, tightness | 75.3% |
| Fluency | breath, exertional, radiating, symptoms | chest, hurt | 32.2% |
| Health literacy | medications, onset, pain, severity, weeks | chest, medication, pill, squeezing, sugar | 21.6% |
| Emotional expressiveness | — | — | 12.4% |
| Confidence | — | — | 9.9% |

The three contaminated dimensions are the three largest effects; the two clean dimensions are the two smallest. The communication-style low-side instruction additionally embeds a multi-clause narrative example, which a length-matched irrelevant paragraph does not absorb.

### H  Activation patching

Protocol: cache the final-position residual stream of the high-style run at every layer, re-run the low-style prompt with one layer's final-position state replaced by its donor, and compute the fraction of the output difference restored. Because raw restoration rises toward 1.0 by construction, the primary measure is the style-minus-placebo difference, with bootstrap intervals per layer, Benjamini–Hochberg correction across 28 layers, and a pre-specified magnitude floor of ±0.03 [protocol following Zhang and Nanda, 2024; Heimersheim and Nanda, 2024]. Sanity controls: self-patching returned 0.000 and final-layer patching returned 1.000.

| Dimension | Positive layers (d ≥ 0.03) | Peak | Negative region |
|---|---|---:|---|
| Fluency | 3–8 | +0.065 (L4) | −0.096 (L15) |
| Confidence | 3–8 | +0.074 (L5) | −0.121 (L17) |
| Health literacy | none | +0.027 (L4) | −0.355 (L17) |

Layers 20-26 significant but trivial (0.005-0.041); restoration there approaches 1.0 with minimal variance, so trivial differences acquire tight intervals and we do not interpret them. **0.5B layer-14 effect did not replicate at 1.5B** At 1.5B the large transition occurs at layer 20 and is shared with the placebo contrast, identifying the model's general commitment point rather than a style mechanism.

### I  Conversation-level transfer: an uninformative experiment

We ran full simulated consultations for 20 held-out scenario pairs — four conversations each (high/low style × base/adapted clinician), eleven turns, 80 conversations in total. The simulated patient was the **unmodified base model in every condition**, so only the clinician varied. Transcripts were scored with pre-existing, unmodified friction detectors.

| Outcome | Base | Adapted | Narrowing | 95% CI | *p* |
|---|---:|---:|---:|---|---:|
| Friction gap (low − high) | -0.025 | -0.0375 | +0.0125 | [-0.1125, +0.1250] | 0.684 |

**This experiment supports no claim in either direction.** The base clinician's friction gap was -0.025 against absolute levels near 1.55 — indistinguishable from zero and opposite in sign to the disparity the detectors were built to find — so there was no gap available to narrow, and the interval is an order of magnitude wider than the effect it brackets. Two design choices plausibly account for it: the simulated patient was the base model in both conditions, so style entered only through the scenario instruction and not through the patient's actual language; and 20 of 100 available pairs is too few at this variance. We report it because it was pre-specified and run, not because it is informative.

### J  Probe procedure

Logistic regression (L2, C = 1.0 for style, C = 0.05 for the scenario-level urgency control), features standardised on the training split only. Layers 2, 4, 6, 8, 12, 16, 20, 24, 27. Two readouts: the final prompt position, and the mean over cue-token positions located by subsequence search over token ids.

**Cluster bootstrap.** Units are resampled with replacement — never items. For the style probe the unit is the (dimension, phrasing) pair (15 held out); for the content probe the unit is the scenario (18 held out). Each resample draws units, concatenates their item indices, and recomputes the AUC difference; the reported interval is the 2.5th–97.5th percentile over 2,000 resamples.

**Cluster permutation test.** The null is that the base/adapted assignment is exchangeable at the unit level. Each of 2,000 iterations flips a random subset of whole units, recomputes the AUC difference, and the two-sided *p* is the proportion of null statistics at least as extreme as the observed one, with the standard +1 correction. At 15 units the permutation space is $2^{15}$, so the test is not floor-limited.

**Why this matters here.** An earlier item-level analysis of the same design reported an apparently significant late-layer effect. That analysis resampled items rather than units and therefore treated correlated observations as independent, understating intervals by roughly the square root of the cluster size. Only the cluster-corrected analysis is reported in this paper.

### K  Full tables

## Table 1 — Baseline speaker-style sensitivity

| Dimension | $D_{\text{style}}$ (bits) | $D_{\text{placebo}}$ (bits) | Ratio | % of positive control | Phrasings positive |
|---|---:|---:|---:|---:|---:|
| Health literacy | 0.00604 | 0.00082 | 7.4× | 4.1% | 8/8 |
| Confidence | 0.00452 | 0.00090 | 5.0× | 3.1% | 8/8 |
| Communication style | 0.00888 | 0.00181 | 4.9× | 6.0% | 8/8 |
| Emotional expressiveness | 0.00543 | 0.00115 | 4.7× | 3.7% | 8/8 |
| Fluency | 0.00513 | 0.00144 | 3.6× | 3.5% | 8/8 |

Unit of analysis: phrasing ($n$ = 8 per dimension), across 20 scenarios. One-sided Wilcoxon signed-rank, $p$ = 0.0039 for every dimension. Positive control (replacing the entire clinical history, style and placebo sentences held fixed) = 0.1475 bits on this scenario set. An independent reimplementation reproduced these ratios to a mean absolute difference of 0.02.

## Table 2 — Cue-operationalisation robustness

| Cue form | Health literacy | Confidence | Comm. style | Emotional expr. | Fluency | Median | Dims positive | $p$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Author-written descriptor | 6.2× | 2.5× | 2.8× | 1.0× | 1.1× | 2.5× | 5/5 | 0.0312 |
| Mechanical 2nd→3rd person | 3.7× | 1.9× | 5.4× | 2.4× | 2.6× | 2.6× | 5/5 | 0.0312 |
| Verbatim corpus instruction | 3.7× † | 2.5× | 15.5× † | 1.9× | 4.8× † | 3.7× | 5/5 | 0.0312 |

Unit of analysis: dimension ($n$ = 5), exact sign test; $p$ = 0.0312 is the floor at this $n$. All three arms use **one** phrasing per dimension, matched to the raw arm, which has exactly one instruction per (dimension, variant) in the corpus; the eight-phrasing values in Table 1 are the primary baseline. † clinical content words appear on one side of the cue only. Percentage magnitudes are **not** comparable across arms: raw instructions are 73–136 tokens against 11–15 for descriptors. Spearman $\rho$ between descriptor and raw dimension ratios = +0.50 ($p$ = 0.39, $n$ = 5).

## Table 3 — Placebo-construction robustness

| Dimension | Family-constrained | Flat pool | Change |
|---|---:|---:|---:|
| Health literacy | 7.4× | 4.9× | -34% |
| Confidence | 5.0× | 3.3× | -34% |
| Communication style | 4.9× | 4.7× | -4% |
| Emotional expressiveness | 4.7× | 4.8× | +2% |
| Fluency | 3.6× | 2.5× | -31% |
| **Median** | **4.9×** | **4.7×** | **-4%** |

Both builders match the placebo pair to the style pair on exact token length of each side and on cosine distance in input-embedding space. The family-constrained builder additionally draws both placebo sentences from the same stem family; the flat builder draws from one pool. Every dimension stays positive under both (7–8 of 8 phrasings, all $p \le$ .0117).

## Table 4 — Self-distillation across independent runs

| Run | Reduction | Perplexity | Neutral drift (bits) | Health literacy | Confidence | Comm. style | Emotional expr. | Fluency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original (unseeded) | 91.6% | 12.23 → 12.02 | 0.00263 | 88.1% | 94.4% | 91.7% | 92.3% | 93.2% |
| seed 1 | 90.9% | 12.23 → 12.01 | 0.00262 | 87.4% | 94.0% | 90.2% | 92.0% | 92.9% |
| seed 2 | 91.7% | 12.23 → 12.08 | 0.00263 | 88.3% | 94.5% | 91.5% | 92.4% | 93.5% |
| seed 3 | 91.8% | 12.23 → 12.01 | 0.00234 | 88.4% | 94.6% | 91.6% | 92.7% | 93.4% |
| **3 seeds** | **mean 91.5%** | | | 87.4–88.4% | 94.0–94.6% | 90.2–91.6% | 92.0–92.7% | 92.9–93.5% |

Evaluated on held-out phrasings crossed with held-out scenarios. Seeds vary adapter initialisation, data order and dropout only; the train/test split is fixed across runs. Range across the three seeded runs 90.9–91.8% (spread 0.9 points); the unseeded original run falls inside it. We report mean and range rather than a confidence interval, which is not meaningful at $n$ = 3. Selectivity: clinical sensitivity changed +2.7%, placebo sensitivity fell 75.4%, and style influence relative to the clinical effect fell 5.22% → 0.43% (12.2×).

## Table 5 — Soft-target self-distillation vs. hard-target SFT

| Model | Entropy (bits) | Mean top-1 $p$ | Frac. $p_{max}$ > .99 | Style gap (bits) | Change | Neutral drift (bits) |
|---|---:|---:|---:|---:|---:|---:|
| Base model | 2.024 | 0.604 | 0.127 | 0.00942 | — | — |
| Self-distillation (soft targets) | 2.022 | 0.607 | 0.124 | 0.00080 | -91.6% | 0.0026 |
| SFT (hard targets) | 0.278 | 0.936 | 0.688 | 0.01123 | +19.2% | 0.3025 |
| *Teacher (style-blind base)* | *1.985* | — | — | — | — | — |

Forward KL is mass-covering, so the teacher's entropy is a floor for the soft-target student; the student sits just above it. Hard-target drift is 1.29× the form-matched positive control (0.2341 bits, measured on prompts with no style line); soft-target drift is 0.011×. Two hard-target runs at different learning rates moved the style gap **-24.2%** (lr 1e-04) and **+19.2%** (lr 3e-05) while drift was 0.302 and 0.3025: the distributional damage reproduces, the sensitivity metric does not reproduce in sign.

## Table 6 — Linear probe for style, before and after intervention

| Layer | Base AUC (final pos.) | Base AUC (cue pos.) | After AUC (final pos.) | Δ final | 95% cluster CI | Perm. $p$ |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.634 | 0.711 | 0.633 | +0.001 | [-0.006, +0.008] | 0.882 |
| 4 | 0.673 | 0.751 | 0.657 | +0.016 | [-0.001, +0.035] | 0.064 |
| 6 | 0.698 | 0.760 | 0.681 | +0.017 | [-0.016, +0.056] | 0.354 |
| 8 | 0.732 | 0.787 | 0.683 | +0.049 | [+0.005, +0.107] | 0.064 |
| 12 | 0.826 | 0.884 | 0.818 | +0.008 | [-0.041, +0.052] | 0.752 |
| 16 | 0.851 | 0.884 | 0.838 | +0.013 | [-0.037, +0.068] | 0.540 |
| 20 | 0.790 | 0.876 | 0.808 | -0.018 | [-0.083, +0.033] | 0.485 |
| 24 | 0.850 | 0.747 | 0.835 | +0.015 | [-0.041, +0.068] | 0.657 |
| 27 | 0.857 | 0.760 | 0.828 | +0.029 | [-0.040, +0.097] | 0.416 |

Intervals are cluster bootstraps resampling the unit — (dimension, phrasing) for style ($n$ = 15 held-out units) — not items; the permutation test exchanges the base/intervened assignment within whole units. No layer shows both an interval excluding zero and $p$ < .05. With 95% confidence no layer's style-decodability drop exceeds 0.107 AUC. Cue-position readout exceeds final-position readout by +0.064 AUC on average through layer 20 in the base model.
