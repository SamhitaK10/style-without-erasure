# Data provenance

**Nothing in this directory is committed.** The scenario file is generated:

```bash
python tools/fetch_scenarios.py        # → data/scenarios.jsonl (50 rows)
```

Read this whole file before redistributing anything derived from it.

---

## What the experiments actually consume

Fifty **clinical content blocks**, one per scenario, each of the form

```
Reported history: started 3 days ago; pressure-like, comes on with exertion; …
```

Each is the semicolon-joined `latent_facts` list of one scenario, with trailing
periods stripped. The same block appears on both sides of every style
comparison, so propositional content is identical **by construction, not by
paraphrase**; identity is hash-verified. The paper uses 30 scenarios for
training and 20 held out.

No conversation text, note text, or model output enters the experiments — only
these fact blocks.

---

## The derivation chain

```
MTS-Dialog  (real transcribed clinical text, CC BY 4.0)
    │
    │  40 of 50 scenarios grounded in a specific record;
    │  content restructured into a ten-fact schema,
    │  identifying specifics generalised, NOT verbatim
    ▼
DiaLense scenarios S1–S50
    │
    │  latent_facts → single "Reported history: …" block
    ▼
this paper's 50 content blocks
```

### Step 1 — upstream corpus

| | |
|---|---|
| **Name** | MTS-Dialog |
| **Citation** | Ben Abacha A, Yim W, Fan Y, Lin T. *An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters.* EACL 2023, pp. 2291–2302. |
| **Official source** | <https://github.com/abachaa/MTS-Dialog> |
| **License** | **CC BY 4.0.** The upstream repository states: "This work is published under a Creative Commons Attribution 4.0 International License (CC BY 4.0)." |
| **Contents** | 1,701 conversation / clinical-note-section pairs (1,201 train + 100 validation + 200 test-1 + 200 test-2) |
| **Nature** | Doctor–patient conversations paired with clinical note sections. The upstream repository states no de-identification procedure, so this project makes no de-identification claim and no such claim should be added without confirmation from the corpus authors. |
| **Redistribution** | Permitted. CC BY 4.0 allows redistribution and adaptation **with attribution**. |

DiaLense ships an annotated copy at `data/mts_dialog/MTS-Dialog-Annotated.csv`
— the 1,701 upstream rows with an added `split` column and nine derived
annotation columns. That is an adaptation, which CC BY explicitly covers.

### Step 2 — the DiaLense scenarios

Source: <https://github.com/SamhitaK10/DiaLense> (MIT code, CC BY 4.0 data)

| Scenarios | Provenance |
|---|---|
| **S11–S50** (40) | Each grounded in a specific MTS-Dialog record. `DiaLense/scenarios/provenance.csv` records `source_id`, `source_section`, a match score and a resolved excerpt for every one; all 40 carry verdict `OK`. `scenarios.py` states the clinical content follows the source note, was restructured into the ten-fact schema, identifying specifics were generalised, and the scenarios are **not verbatim copies**. |
| **S1–S10** (10) | **No source record was ever recorded for these ten scenarios.** They predate the provenance schema: `scenarios.py` documents the expanded format as "SCHEMA (matches S1-S10 exactly, plus two provenance fields)", so `source_id` and `source_section` were introduced *with* S11–S50 and never backfilled. S1–S10 are defined inline in `experiment.py` and their field set confirms it — they carry `id`, `name`, `phase`, `chief_complaint`, `correct_urgency`, `expected_recommendation`, `latent_facts` and `critical_facts`, and no provenance fields. They are absent from `provenance.csv` entirely. See "Can S1–S10 provenance be recovered?" below. |

### Can S1–S10 provenance be recovered?

Not from these files, and not by post-hoc matching.

DiaLense resolves provenance for S11–S50 with `scripts/resolve_provenance.py`,
which scores a scenario's chief complaint and latent facts against candidate
MTS-Dialog rows by lexical overlap. That works because those scenarios already
carry a `source_id` naming the record; the script only disambiguates *which row*
that ID refers to.

Running the same lexical scoring for S1–S10 against all 1,701 MTS-Dialog rows
cannot establish provenance, because the score does not discriminate. Best-match
scores for S1–S10 fall between **0.180 and 0.368**, which sits inside the range
of the recorded, verified matches for S11–S50 (**0.130 to 0.550**, median 0.350).
A score in that band is equally consistent with derivation from a record and
with independent authorship using ordinary clinical vocabulary, so no
conclusion can be drawn from it. That diagnostic is reported here so nobody
repeats it and mistakes it for evidence.

The `phase` field on S1–S10 names MTS-Dialog section headers (`CC`, `GENHX`),
which shows the scenario schema was designed around MTS-Dialog. It is not
evidence that any individual scenario was derived from a record.

**What this means in practice.** Describe the corpus as 50 scenarios of which
**40 are traceable to specific MTS-Dialog records and 10 are not**. The CC BY
attribution applies to the whole set regardless: the ten untraceable scenarios
were built to the same schema inside the same project and the conservative
reading is that CC BY obligations flow to them too. Only the author can say
whether S1–S10 were derived or authored; if they were authored, adding one
sentence to `DiaLense/src/dialense/generation/experiment.py` recording that
would close the gap permanently.

---

## How to describe this material

**Accurate:** derived from real transcribed clinical text, restructured into a
fixed fact schema and generalised, then rendered as synthetic scenarios.

**Not accurate:** "real clinical data" (the scenarios are not records);
"fully synthetic" (the clinical grounding for 40 of 50 traces to real
transcribed text); "de-identified" (unverified upstream).

---

## Redistribution

Permitted under CC BY 4.0, with attribution. This repository chooses **not** to
vendor a copy, for three reasons: the data already lives in a public, licensed,
cited repository; the per-scenario provenance table travels with it there; and
a single source avoids two copies drifting apart.

If you redistribute the scenarios or anything derived from them, the CC BY
attribution to Ben Abacha et al. travels with them. See `../LICENSE-DATA`.

---

## Reproducibility of the fetch

`tools/fetch_scenarios.py` clones DiaLense and re-runs the same extraction the
experiment scripts perform inline.

**Pinned.** `DEFAULT_REF` is
`a1adecdd31fa6905583f7beb79e58eb4b062bc06`, the DiaLense commit verified to
yield exactly 50 scenarios. The fetch script also writes
`scenarios.jsonl.meta.json` recording the commit it actually used, and exits
with an error rather than a warning if the scenario count is not 50.

The pin guarantees that every future run extracts the same scenarios. It is
**not** a claim that this commit is the one the original experiments ran
against — the DiaLense repository has a single squashed commit, so its history
cannot distinguish the state at experiment time from the state today.

The experiment scripts themselves still run
`git clone --depth 1 https://github.com/SamhitaK10/DiaLense.git` inline, which
takes the default branch. Until they are changed to use this script, run
`tools/fetch_scenarios.py` first if you need a pinned extraction.

DiaLense is public: this pinned fetch was executed successfully against the
remote with no credentials.

---

## Ethics

**No IRB or ethics determination is documented anywhere in this project.** No
claim is made here. State the actual status — including "none was sought,
because …" — before release. Relevant facts a reader will want: the material is
derived from a public CC BY 4.0 corpus; no human subjects were recruited; no
patients or clinicians were involved in this work; the scenarios are not
clinical records; and the clinical setting is a testbed rather than the claim.
