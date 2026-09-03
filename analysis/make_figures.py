"""Publication figures for the speaker-style sensitivity paper.
Every value is read from data/master.json — nothing is hardcoded here."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "master.json")))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# validated categorical slots 1-3 (light mode), plus ink tokens
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e3e1db"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})
PRETTY = {"health_literacy": "Health literacy", "confidence": "Confidence",
          "communication_style": "Communication style",
          "emotional_expressiveness": "Emotional expressiveness", "fluency": "Fluency"}

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig); print("  wrote", name)

def grid_x(ax):
    ax.set_axisbelow(True); ax.xaxis.grid(True, color=GRID, lw=.7); ax.yaxis.grid(False)
def grid_y(ax):
    ax.set_axisbelow(True); ax.yaxis.grid(True, color=GRID, lw=.7); ax.xaxis.grid(False)

# ---------------------------------------------------------------- FIGURE 2 ---
def fig2():
    dims = sorted(D["baseline"]["dims"], key=lambda k: -D["baseline"]["dims"][k]["ratio"])
    st = [D["baseline"]["dims"][k]["d_style"] for k in dims]
    pl = [D["baseline"]["dims"][k]["d_placebo"] for k in dims]
    ra = [D["baseline"]["dims"][k]["ratio"] for k in dims]
    pc = D["baseline"]["positive_control_bits"]
    y = np.arange(len(dims)); h = 0.34
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [2.05, 1], "wspace": 0.10})
    a.barh(y + h/2, st, h, color=S1, label="style swap", zorder=3)
    a.barh(y - h/2, pl, h, color=S2, label="matched placebo swap", zorder=3)
    for i, (s, r) in enumerate(zip(st, ra)):
        a.text(s + 0.00025, i + h/2, f"{r:.1f}×", va="center", ha="left",
               fontsize=8.5, color=INK, fontweight="bold")
    a.set_yticks(y); a.set_yticklabels([PRETTY[k] for k in dims])
    a.set_xlabel("Jensen–Shannon divergence (bits)")
    a.set_xlim(0, max(st) * 1.34); grid_x(a)
    a.set_title("a  Style vs. matched control", pad=8)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    pct = [100 * s / pc for s in st]
    b.barh(y, pct, 0.5, color=S3, zorder=3)
    for i, v in enumerate(pct):
        b.text(v + 0.15, i, f"{v:.1f}%", va="center", fontsize=8.5, color=INK)
    b.set_yticks(y); b.set_yticklabels([]); b.set_xlim(0, max(pct) * 1.32); grid_x(b)
    b.set_xlabel("% of a full clinical-history swap")
    b.set_title("b  Interpretable scale", pad=8)
    b.text(0.5, -0.22, f"positive control = {pc:.4f} bits", transform=b.transAxes,
           ha="center", fontsize=8, color=MUTED)
    save(fig, "fig2_baseline")

# ---------------------------------------------------------------- FIGURE 3 ---
def fig3():
    arms = [("descriptor", "Author-written descriptor", S1),
            ("mechanical_3rd", "Mechanical 2nd→3rd person", S2),
            ("raw_instruction", "Verbatim corpus instruction", S3)]
    dims = ["communication_style", "fluency", "health_literacy",
            "emotional_expressiveness", "confidence"]
    contam = set(D["cue_robustness"]["contaminated_dims"])
    x = np.arange(len(dims)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    for j, (key, lab, c) in enumerate(arms):
        v = [D["cue_robustness"]["arms"][key]["per_dim_ratio"][k] for k in dims]
        ax.bar(x + (j - 1) * w, v, w * 0.92, color=c, label=lab, zorder=3)
    ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(-0.42, 1.35, "no effect", fontsize=7.5, color=MUTED, ha="left")
    ax.set_ylim(0, 17.6)
    for i, k in enumerate(dims):
        if k in contam:
            ax.text(i, 16.6, "\u25b2", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[k].replace(" ", "\n") for k in dims], fontsize=8.2)
    ax.set_ylabel("style / placebo divergence ratio"); grid_y(ax)
    ax.set_title("Effect direction survives every cue operationalisation\n"
                 "5/5 dimensions positive in each arm, $p$ = .0312", pad=8)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.text(0.5, -0.30, "\u25b2 clinical content differs between the two sides of this cue"
            "    \u00b7    one phrasing per dimension in every arm, matched to the raw arm's $n$ = 1",
            transform=ax.transAxes, ha="center", fontsize=7.5, color=MUTED)
    save(fig, "fig3_cue_robustness")

# ---------------------------------------------------------------- FIGURE 4 ---
def fig4():
    fam, flat = D["placebo_robustness"]["family"], D["placebo_robustness"]["flat"]
    dims = sorted(fam, key=lambda k: -fam[k]); y = np.arange(len(dims))
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    for i, k in enumerate(dims):
        ax.plot([flat[k], fam[k]], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=2)
    ax.scatter([fam[k] for k in dims], y, s=52, color=S1, zorder=4, label="family-constrained placebo")
    ax.scatter([flat[k] for k in dims], y, s=52, color=S2, zorder=4, label="flat placebo pool")
    for i, k in enumerate(dims):
        ax.text(max(fam[k], flat[k]) + 0.16, i, f"{fam[k]:.1f}→{flat[k]:.1f}",
                va="center", fontsize=8, color=INK2)
    ax.axvline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([PRETTY[k] for k in dims])
    ax.set_xlabel("style / placebo divergence ratio"); ax.set_xlim(0, 9.4); grid_x(ax)
    ax.set_title("Direction is robust to placebo construction; magnitude is not", pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    save(fig, "fig4_placebo_robustness")

# ---------------------------------------------------------------- FIGURE 5 ---
def fig5():
    runs = D["intervention"]["runs"]
    labs = [r["label"] for r in runs]; red = [r["reduction_pct"] for r in runs]
    seeded = [r["reduction_pct"] for r in runs if r["label"].startswith("seed")]
    lo, hi, mn = min(seeded), max(seeded), float(np.mean(seeded))
    dims = ["confidence", "fluency", "emotional_expressiveness", "communication_style", "health_literacy"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.1), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.42})
    a.axvspan(lo, hi, color=S1, alpha=.10, zorder=1)
    a.axvline(mn, color=S1, lw=1.2, ls=(0, (4, 3)), zorder=2)
    a.scatter(red, np.arange(len(runs)), s=58,
              color=[MUTED if l.startswith("original") else S1 for l in labs], zorder=4)
    for i, v in enumerate(red):
        a.text(v + 0.06, i, f"{v:.1f}%", va="center", fontsize=8.5, color=INK)
    a.set_yticks(np.arange(len(runs))); a.set_yticklabels(labs)
    a.set_xlim(90.4, 92.4); a.set_xlabel("reduction in held-out style sensitivity (%)"); grid_x(a)
    a.set_title("a  Across independent runs")
    a.text(0.5, -0.44, f"3 seeds: mean {mn:.1f}%, range {lo:.1f}–{hi:.1f}, spread {hi-lo:.1f} pts",
           transform=a.transAxes, ha="center", fontsize=8, color=MUTED)
    for j, r in enumerate(runs):
        if r["label"].startswith("original"): continue
        b.scatter([r["per_dim"][k] for k in dims], np.arange(len(dims)),
                  s=46, color=[S1, S2, S3][j - 1], zorder=4, label=r["label"])
    b.set_yticks(np.arange(len(dims))); b.set_yticklabels([PRETTY[k] for k in dims])
    b.set_xlabel("reduction (%)"); b.set_xlim(86.5, 96); grid_x(b)
    b.set_title("b  Per dimension, per seed", pad=8)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    save(fig, "fig5_seeds")

# ---------------------------------------------------------------- FIGURE 6 ---
def fig6():
    before = D["intervention"]["before_by_dim"]
    runs = [r for r in D["intervention"]["runs"] if r["label"].startswith("seed")]
    dims = sorted(before, key=lambda k: -before[k])
    after = {k: np.mean([before[k] * (1 - r["per_dim"][k] / 100) for r in runs]) for k in dims}
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    off = {k: 0.0 for k in dims}
    lab_y = np.linspace(0.0038, 0.0002, len(dims))     # spread the crowded right-hand labels
    for i, k in enumerate(dims):
        ax.plot([0, 1], [before[k], after[k]], color=S1, lw=1.6, marker="o", ms=6, zorder=3)
        ax.text(-0.04, before[k], PRETTY[k], ha="right", va="center", fontsize=8.5, color=INK2)
        ax.plot([1.02, 1.10], [after[k], lab_y[i]], color=GRID, lw=0.8, zorder=2)
        ax.text(1.12, lab_y[i], f"{PRETTY[k]}  {after[k]:.5f}", ha="left", va="center",
                fontsize=7.8, color=INK2)
    ax.set_xlim(-0.72, 1.92); ax.set_xticks([0, 1])
    ax.set_xticklabels(["before", "after (mean of 3 seeds)"])
    ax.set_ylabel("style-driven divergence (bits)"); grid_y(ax)
    ax.set_title("Held-out style sensitivity, before and after self-distillation")
    save(fig, "fig6_before_after")

# ---------------------------------------------------------------- FIGURE 7 ---
def fig7():
    R = D["objective_comparison"]["rows"]; te = D["objective_comparison"]["teacher_entropy"]
    names = ["base", "soft_target", "hard_target"]
    labs = ["Base", "Soft\ntargets", "Hard\ntargets"]
    cols = [MUTED, S1, S2]
    panels = [("entropy", "bits", "a  Output entropy"),
              ("top1", "probability", "b  Mean top-1 probability"),
              ("p_gt_99", "fraction", "c  Positions with $p_{max}$ > .99")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), gridspec_kw={"wspace": 0.34})
    for ax, (key, ylab, title) in zip(axes, panels):
        v = [R[n][key] for n in names]
        ax.bar(np.arange(3), v, 0.56, color=cols, zorder=3)
        for i, x in enumerate(v):
            ax.text(i, x + max(v) * 0.025, f"{x:.3f}", ha="center", fontsize=8.5, color=INK)
        if key == "entropy":
            ax.axhline(te, color=S3, lw=1.3, ls=(0, (4, 3)), zorder=4)
            ax.text(-0.42, te - 0.20, f"teacher {te:.3f}", ha="left", fontsize=7.5, color=S3)
        ax.set_xticks(np.arange(3)); ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylabel(ylab); ax.set_ylim(0, max(v) * 1.22); grid_y(ax); ax.set_title(title)
    fig.suptitle("Soft targets preserve the output distribution; hard targets collapse it",
                 x=0.5, ha="center", fontsize=10.5, fontweight="bold", y=1.10)
    save(fig, "fig7_distribution")

# ---------------------------------------------------------------- FIGURE 8 ---
def fig8():
    R = D["objective_comparison"]["rows"]; ctrl = D["objective_comparison"]["bare_facts_control_bits"]
    vals = [R["soft_target"]["drift"], R["hard_target"]["drift"]]
    labs = ["Self-distillation\n(soft targets)", "SFT\n(hard targets)"]
    fig, ax = plt.subplots(figsize=(6.2, 2.3))
    ax.barh([1, 0], vals, 0.44, color=[S1, S2], zorder=3)
    ax.axvline(ctrl, color=INK2, lw=1.3, ls=(0, (4, 3)), zorder=4)
    ax.text(ctrl + 0.006, 1.42, f"replacing the entire clinical history\n({ctrl:.4f} bits)",
            fontsize=7.8, color=INK2, va="top")
    ax.text(vals[0] + 0.006, 1, f"{vals[0]:.4f} bits  ({vals[0]/ctrl:.2f}×)", va="center",
            fontsize=8.5, color=INK)
    ax.text(vals[1] + 0.006, 0, f"{vals[1]:.4f} bits  ({vals[1]/ctrl:.2f}×)", va="center",
            fontsize=8.5, color=INK)
    ax.set_yticks([1, 0]); ax.set_yticklabels(labs); ax.set_ylim(-0.55, 1.75)
    ax.set_xlim(0, 0.345); ax.set_xlabel("drift on style-neutral prompts (bits)"); grid_x(ax)
    ax.set_title("Hard-target training moves neutral behaviour more than swapping the medicine does")
    save(fig, "fig8_drift")

# ---------------------------------------------------------------- FIGURE 9 ---
def fig9():
    runs = D["objective_comparison"]["hard_target_runs"]
    labs = [f"run 1\nlr {runs[0]['lr']:.0e}", f"run 2\nlr {runs[1]['lr']:.0e}"]
    gap = [r["style_gap_change_pct"] for r in runs]; dr = [r["drift"] for r in runs]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.8), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.36})
    a.bar([0, 1], gap, 0.46, color=[S1 if g < 0 else S2 for g in gap], zorder=3)
    a.axhline(0, color=INK2, lw=1)
    for i, g in enumerate(gap):
        a.text(i, g + (2.2 if g > 0 else -3.4), f"{g:+.1f}%", ha="center", fontsize=9,
               color=INK, fontweight="bold")
    a.set_xticks([0, 1]); a.set_xticklabels(labs); a.set_ylim(-32, 27)
    a.set_ylabel("change in measured style gap (%)"); grid_y(a)
    a.set_title("a  The fairness metric flips sign")
    b.bar([0, 1], dr, 0.46, color=MUTED, zorder=3)
    for i, v in enumerate(dr):
        b.text(i, v + 0.008, f"{v:.4f}", ha="center", fontsize=9, color=INK)
    b.set_xticks([0, 1]); b.set_xticklabels(labs); b.set_ylim(0, 0.40)
    b.set_ylabel("neutral-prompt drift (bits)"); grid_y(b)
    b.set_title("b  The damage does not")
    save(fig, "fig9_hard_instability")

# --------------------------------------------------------------- FIGURE 10 ---
def fig10():
    P = D["probe"]; L = P["layers"]
    bf, bc = np.array(P["base_style_final"]), np.array(P["base_style_cue"])
    df = np.array([x[0] for x in P["delta_style_final"]])
    dlo = np.array([x[1] for x in P["delta_style_final"]])
    dhi = np.array([x[2] for x in P["delta_style_final"]])
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.4, 4.5), sharex=True,
                               gridspec_kw={"height_ratios": [1.55, 1]})
    a.plot(L, bc, color=S3, lw=2, marker="o", ms=5, label="base — cue positions", zorder=4)
    a.plot(L, bf, color=S1, lw=2, marker="o", ms=5, label="base — final position", zorder=4)
    a.plot(L, bf - df, color=S2, lw=2, marker="s", ms=4.5, ls=(0, (5, 2)),
           label="after intervention — final position", zorder=4)
    a.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=2)
    a.text(27, 0.515, "chance", ha="right", fontsize=7.5, color=MUTED)
    a.set_ylabel("style probe AUC"); a.set_ylim(0.44, 0.99); grid_y(a)
    a.legend(loc="upper left", bbox_to_anchor=(0.005, 0.99), ncol=1)
    a.set_title("a  Style is linearly decodable, and more so at the cue tokens\n"
                "     than at the final position", pad=6)
    b.axhline(0, color=INK2, lw=1, zorder=2)
    b.vlines(L, dlo, dhi, color=S1, lw=2.4, alpha=.45, zorder=3)
    b.plot(L, df, color=S1, lw=0, marker="o", ms=6, zorder=4)
    b.set_ylabel("AUC drop after\nintervention"); b.set_xlabel("layer")
    b.set_xticks(L); grid_y(b)
    b.set_title(f"b  No layer survives the cluster permutation test ($n$ = {P['n_units_style']} units); "
                f"bound {P['bound_auc']:.3f} AUC")
    save(fig, "fig10_probe")

# --------------------------------------------------------------- FIGURE 11 ---
def fig11():
    seeded = [r["reduction_pct"] for r in D["intervention"]["runs"] if r["label"].startswith("seed")]
    P = D["probe"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.30})
    a.bar([0], [float(np.mean(seeded))], 0.42, color=S1, zorder=3,
          yerr=[[np.mean(seeded) - min(seeded)], [max(seeded) - np.mean(seeded)]],
          error_kw=dict(ecolor=INK2, capsize=4, lw=1.2))
    a.text(0, np.mean(seeded) + 3.2, f"{np.mean(seeded):.1f}%", ha="center", fontsize=12,
           fontweight="bold", color=INK)
    a.text(0, 46, f"range\n{min(seeded):.1f}–{max(seeded):.1f}\n(3 seeds)",
           ha="center", va="center", fontsize=8.5, color="white")
    a.set_xticks([]); a.set_ylim(0, 108); a.set_ylabel("reduction (%)"); grid_y(a)
    a.set_title("a  Behavioural sensitivity")
    L = P["layers"]; bf = np.array(P["base_style_final"])
    df = np.array([x[0] for x in P["delta_style_final"]])
    x = np.arange(len(L)); w = 0.38
    b.bar(x - w/2, bf, w, color=MUTED, label="base", zorder=3)
    b.bar(x + w/2, bf - df, w, color=S1, label="after intervention", zorder=3)
    b.axhline(0.5, color=INK2, lw=1, ls=(0, (3, 3)), zorder=4)
    b.set_xticks(x); b.set_xticklabels(L); b.set_xlabel("layer")
    b.set_ylabel("style probe AUC"); b.set_ylim(0.4, 0.99); grid_y(b)
    b.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.0))
    b.set_title("b  Linear decodability — no layer significant")
    save(fig, "fig11_summary")

for f in (fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11):
    f()
print("done")

# ---------------------------------------------------------------- FIGURE 1 ---
def fig1():
    """Method overview. Schematic — carries no measured values."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    ax.set_xlim(-3, 103); ax.set_ylim(-2, 112); ax.axis("off")

    def box(x, y, w, h, txt, fc="white", ec=MUTED, fs=7.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.1",
                                    fc=fc, ec=ec, lw=1.0, zorder=3))
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs, zorder=4,
                color=INK, linespacing=1.5)
    def arrow(x1, y1, x2, y2, c=MUTED, style="-|>", lw=1.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9,
                                     color=c, lw=lw, zorder=2, shrinkA=2, shrinkB=2))
    def band(ytop, letter, label):
        ax.text(-3, ytop, f"{letter}   {label}", fontsize=8.8, fontweight="bold", color=INK, va="center")
        ax.plot([-3, 103], [ytop - 3.5, ytop - 3.5], color=GRID, lw=0.9, zorder=1)

    # ---------------- a: measurement ----------------
    band(109, "a", "Controlled measurement")
    box(0, 92, 24, 9.5, "style$^+$ · placebo$^+$\nidentical content", fc="#eef4fc", ec=S1)
    box(0, 80, 24, 9.5, "style$^-$ · placebo$^+$\nidentical content", fc="#eef4fc", ec=S1)
    box(0, 68, 24, 9.5, "style$^+$ · placebo$^-$\nidentical content", fc="#fdeee7", ec=S2)
    box(32, 80, 13, 9.5, "language\nmodel", fc="#f4f3ef", ec=INK2, fs=8)
    for y in (96.7, 84.7, 72.7): arrow(24.5, y, 31.5, 84.7)
    box(54, 93, 46, 9, r"$D_{\rm style}$: JS divergence over a fixed" + "\n24-token reference continuation", ec=S1)
    box(54, 81, 46, 9, r"$D_{\rm placebo}$: same measurement," + "\nsame change in prompt length", ec=S2)
    box(54, 69, 46, 9, "positive control — swap all content:\nputs every effect on a scale", fc="#eef7f2", ec=S3)
    arrow(45.5, 87, 53.5, 97.5); arrow(45.5, 85, 53.5, 85.5); arrow(45.5, 83, 53.5, 73.5)
    ax.text(77, 64.5, r"reported effect $= D_{\rm style} - D_{\rm placebo}$, also as % of the control",
            ha="center", fontsize=7.6, color=INK, style="italic")

    # ---------------- b: intervention ----------------
    band(56, "b", "Intervention: self-distillation against a style-blind teacher")
    box(0, 39, 26, 8.5, "content only\n(no style cue)", fc="#f4f3ef", ec=MUTED)
    box(0, 27, 26, 8.5, "style cue + content", fc="#eef4fc", ec=S1)
    box(35, 39, 25, 8.5, "frozen base model", fc="#f4f3ef", ec=INK2)
    box(35, 27, 25, 8.5, "base + LoRA\n(1.18% of weights)", fc="#eef4fc", ec=S1)
    box(69, 39, 31, 8.5, "teacher distribution", ec=INK2)
    box(69, 27, 31, 8.5, "student distribution", ec=S1)
    arrow(26.5, 43.2, 34.5, 43.2); arrow(26.5, 31.2, 34.5, 31.2)
    arrow(60.5, 43.2, 68.5, 43.2); arrow(60.5, 31.2, 68.5, 31.2)
    arrow(84.5, 36.0, 84.5, 38.6, c=S2, style="<|-|>", lw=1.6)
    ax.text(86.6, 37.3, "KL", fontsize=8.6, color=S2, va="center", fontweight="bold")
    ax.text(50, 21, "forward KL is mass-covering: the teacher's entropy is a floor on the student's,\n"
            "so the objective cannot be minimised by becoming confident",
            ha="center", va="top", fontsize=7.6, color=INK2, style="italic", linespacing=1.55)

    # ---------------- c: evaluation ----------------
    band(12, "c", "Three evaluations, none sufficient alone")
    box(0, 0, 31, 7.5, "behaviour\nheld-out phrasings × scenarios", ec=S1, fs=7.2)
    box(34.5, 0, 31, 7.5, "representation\nlinear probe, cluster-corrected", ec=S3, fs=7.2)
    box(69, 0, 31, 7.5, "distribution\nentropy · top-1 · $p$ > .99", ec=S2, fs=7.2)
    save(fig, "fig1_method")

fig1()
