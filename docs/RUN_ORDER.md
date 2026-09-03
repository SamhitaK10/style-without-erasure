# What to run, in order — Colab T4

**Runtime → Change runtime type → T4 GPU → Save.** One session, run the cells in
this order. Total ~2h20m of GPU time, a small fraction of 100 compute units.

Download the files and paste each into its own cell. Do **not** copy code out of
chat messages — older versions are still sitting up there and one of them cost
you a run already.

| # | Script | Time | Needs | Produces |
|---|---|---|---|---|
| 1 | `FIX_positive_control.py` | ~5 min | — | `positive_control.json` |
| 2 | `EXP_B_raw_instructions.py` | ~25–35 min | — | `exp_B_raw_instructions.json` |
| 3 | `EXP_D_probe.py` | ~10 min | `dialense_lora` on Drive | `exp_D_probe.json` |
| 4 | `EXP_E_objective_comparison.py` | ~35 min | `dialense_lora` on Drive | `exp_E_objective_comparison.json` + `dialense_lora_hard` |
| 5 | `EXP_A_seeds.py` with `SEED = 1` | ~30 min | — | `exp_A_seed1.json` |
| 6 | same file, change to `SEED = 2` | ~30 min | — | `exp_A_seed2.json` |
| 7 | the aggregate cell script 5 prints at the end | instant | both seed files | printed table |

**Disconnect the runtime when you finish.** Compute units burn while the runtime
is connected, not while it is computing. A tab left open overnight is the main
way people waste a purchase.

---

## Why this order

**1 first because it is five minutes and everything else is quoted against it.**
Your baseline used one definition of "replace the entire clinical history"
(0.147 bits, with the style+placebo prefix present) and EXP_E used another
(0.235, bare facts). Until you pick one, no percentage in the paper means
anything. Use PREFIXED / HELD-OUT.

**2 next because it can change what the paper is about.** It tests whether the
effect survives the corpus's own `style_instruction` text instead of the
descriptors you wrote. If RAW and CONV both come back weak, stop and narrow the
framing before spending time on 3–6, which all assume the phenomenon is real.

**3 before 4** because it is ten minutes and answers the objection with the
deepest literature behind it: did the adapter erase the style representation, or
only suppress the readout?

**5 and 6 last** because they are the most expensive and the least likely to
change a conclusion — they turn "91.6%" into a number you are entitled to write.

---

## What to watch in each

**1 — positive control.** The four variants should differ. Note how much of the
0.147 vs 0.235 gap was convention (prefixed vs bare) and how much was scenario
choice. Use the prefixed/held-out number everywhere after this.

**2 — raw instructions.** Read the content-confound audit it prints before the
model loads: low health-literacy and low fluency instructions contain clinical
words the high side lacks. Then the verdict block, three arms, 5 dimensions each.
5/5 positive with median ratio above 2× means the effect is not an artifact of
your writing.

**3 — probe.** Compare `OFF final` against `ON final`, and the same for the cue
positions. A drop under ~0.05 means suppression without erasure — that is a
finding, not a failure. Check the content-control column stayed flat; if it
dropped too, the adapter damaged the representation globally.

**4 — objective comparison.** Two numbers decide whether the ablation holds:
the soft-target student's entropy should sit at or above the teacher's ~1.99 bits
(forward KL is mass-covering, so the teacher's entropy is a floor), and the
hard-target student's should sit well below the base model's with a substantial
`p>.99` fraction. If the hard-target arm did not collapse, the failure mode is
not what we think it is and that changes the paper.

**5–6 — seeds.** Spread under ~8 points across three runs means report mean ±
range and the criticism is dead. Wider means restate the claim as a range, which
is still publishable.

---

## If something breaks

Every script preflights its assumptions and exits in seconds rather than failing
after a model load. If one stops early, the message names what it could not find.

`EXP_E` checkpoints its hard-target training every 40 steps, so an interrupted
run resumes rather than restarting.

Scripts 3 and 4 need `dialense_lora`. They search the Drive folder, the working
directory and `/kaggle/input/*/` automatically; if they cannot find it they say
so immediately and list every path they tried.

---

## After these six

With 100 compute units you have room for the experiments that move this from
Findings-tier to main-conference-tier, in priority order:

1. **Paraphrase-matched placebo arm** — meets the Yang et al. (2026) control
   standard in full rather than halfway. Baseline measurement only.
2. **Cue-position patching** — a position × layer sweep instead of layer-only.
   Upgrades the mechanistic section from a bound to a location.
3. **A second model** — Qwen2.5-3B, baseline plus intervention, no patching.
   The single biggest step toward a main-track paper.

Ask when you get there and I will write them.
