"""LaTeX booktabs tables from master.json. Same source as the markdown tables."""
import json, numpy as np, os
_HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(_HERE, "master.json")))
P = {"health_literacy":"Health literacy","confidence":"Confidence","communication_style":"Comm.\\ style",
     "emotional_expressiveness":"Emotional expr.","fluency":"Fluency"}
# compact forms for wide multi-dimension headers
PS = {"health_literacy":"Health lit.","confidence":"Confid.","communication_style":"Comm.",
      "emotional_expressiveness":"Emot.","fluency":"Fluency"}
os.makedirs(os.path.join(_HERE, "tables"), exist_ok=True)
T = {}

B = D["baseline"]; dims = sorted(B["dims"], key=lambda k: -B["dims"][k]["ratio"])
rows = "\n".join(
 f"{P[k]} & {B['dims'][k]['d_style']:.5f} & {B['dims'][k]['d_placebo']:.5f} & "
 f"{B['dims'][k]['ratio']:.1f}$\\times$ & {100*B['dims'][k]['d_style']/B['positive_control_bits']:.1f}\\% & 8/8 \\\\"
 for k in dims)
T[1] = r"""\begin{table}[t]\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Dimension & $D_{\mathrm{style}}$ & $D_{\mathrm{plac}}$ & Ratio & \%\,ctrl & $k$/8 \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Baseline speaker-style sensitivity.} Divergence in bits over a fixed 24-token reference
continuation. Unit of analysis is the phrasing ($n=8$ per dimension) across 20 scenarios; one-sided Wilcoxon
$p=.0039$ for every dimension, the floor at this $n$. \%\,ctrl is the style effect as a percentage of the
positive control (0.1475 bits on this scenario set). An independent reimplementation reproduced these ratios
to a mean absolute difference of 0.02.}
\label{tab:baseline}
\end{table}"""

C = D["cue_robustness"]
names = {"descriptor":"Author-written descriptor","mechanical_3rd":"Mechanical 2nd$\\to$3rd person",
         "raw_instruction":"Verbatim corpus instruction"}
rows = ""
for a in ("descriptor","mechanical_3rd","raw_instruction"):
    r = C["arms"][a]
    cells = " & ".join(f"{r['per_dim_ratio'][k]:.1f}" + ("$^\\dagger$" if k in C["contaminated_dims"] and a=="raw_instruction" else "")
                       for k in dims)
    rows += f"{names[a]} & {cells} & {r['median_ratio']:.1f} & {r['positives']}/5 \\\\\n"
T[2] = r"""\begin{table*}[t]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}l""" + "r"*len(dims) + r"""rr@{}}
\toprule
Cue form & """ + " & ".join(PS[k] for k in dims) + r""" & Median & Positive \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}
\caption{\textbf{Cue-operationalisation robustness}, as style/placebo ratio. Unit of analysis is the dimension
($n=5$), exact sign test, $p=.0312$ for every arm --- the floor at this $n$. All three arms use one phrasing
per dimension, matched to the raw arm, which has exactly one instruction per (dimension, variant) in the
corpus; the eight-phrasing values in Table~\ref{tab:baseline} are the primary baseline.
$^\dagger$ clinical content words appear on one side of the cue only. Percentage magnitudes are \emph{not}
comparable across arms: raw instructions run 73--136 tokens against 11--15 for descriptors, which is why the
length-controlled ratio is tabulated. Spearman $\rho$ between descriptor and verbatim orderings
$= +0.50$ ($p=.39$, $n=5$).}
\label{tab:cue}
\end{table*}"""

R = D["placebo_robustness"]
rows = "\n".join(f"{P[k]} & {R['family'][k]:.1f}$\\times$ & {R['flat'][k]:.1f}$\\times$ & "
                 f"{100*(R['flat'][k]-R['family'][k])/R['family'][k]:+.0f}\\% \\\\" for k in dims)
fm=[R["family"][k] for k in dims]; fl=[R["flat"][k] for k in dims]
T[3] = r"""\begin{table}[t]\centering\small
\begin{tabular}{@{}lrrr@{}}
\toprule
Dimension & Family & Flat pool & Change \\
\midrule
""" + rows + f"""
\\midrule
\\textbf{{Median}} & \\textbf{{{np.median(fm):.1f}$\\times$}} & \\textbf{{{np.median(fl):.1f}$\\times$}} & \\textbf{{{100*(np.median(fl)-np.median(fm))/np.median(fm):+.0f}\\%}} \\\\
""" + r"""\bottomrule
\end{tabular}
\caption{\textbf{Placebo-construction robustness.} Both builders match the placebo pair to the style pair on
exact token length of each side and on cosine distance in input-embedding space; the family-constrained
builder additionally requires both placebo sentences to come from one stem family. Every dimension stays
positive under both (7--8 of 8 phrasings, all $p \le .0117$), but three of five fall by about a third.
Direction is robust; magnitude is not.}
\label{tab:placebo}
\end{table}"""

I = D["intervention"]
rows = ""
for r in I["runs"]:
    rows += (f"{r['label']} & {r['reduction_pct']:.1f}\\% & {r['after_ppl']:.2f} & {r['drift']:.5f} & "
             + " & ".join(f"{r['per_dim'][k]:.1f}" for k in dims) + " \\\\\n")
sd=[r["reduction_pct"] for r in I["runs"] if r["label"].startswith("seed")]
T[4] = r"""\begin{table*}[t]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrr""" + "r"*len(dims) + r"""@{}}
\toprule
& & & & \multicolumn{5}{c}{Per-dimension reduction (\%)} \\
\cmidrule(l){5-9}
Run & Reduc. & Perplex. & Drift (bits) & """ + " & ".join(PS[k] for k in dims) + r""" \\
\midrule
""" + rows + r"""\midrule
\textbf{3 seeds} & \textbf{""" + f"{np.mean(sd):.1f}\\% mean" + r"""} & \multicolumn{2}{l}{range """ + \
 f"{min(sd):.1f}--{max(sd):.1f}\\%" + r"""} & \multicolumn{5}{l}{every dimension's range $<1.5$ points} \\
\bottomrule
\end{tabular}
\caption{\textbf{Self-distillation across independent runs.} Evaluated on held-out phrasings crossed with
held-out scenarios. Pre-intervention perplexity is 12.23 in every run. Seeds vary adapter initialisation, data
order and dropout only; the train/test split is fixed across runs. We report mean and range and decline to
construct a confidence interval at $n=3$. Selectivity (single run): clinical sensitivity $+2.7\%$, placebo
sensitivity $-75.4\%$, style influence relative to the clinical effect $5.22\% \to 0.43\%$ ($12.2\times$).}
\label{tab:seeds}
\end{table*}"""

O = D["objective_comparison"]; RO = O["rows"]
lab = {"base":"Base model","soft_target":"Self-distillation (soft targets)","hard_target":"SFT (hard targets)"}
rows = ""
for k in ("base","soft_target","hard_target"):
    r = RO[k]
    rows += (f"{lab[k]} & {r['entropy']:.3f} & {r['top1']:.3f} & {r['p_gt_99']:.3f} & {r['style_gap']:.5f} & "
             + (f"{r['change_pct']:+.1f}\\%" if r["change_pct"] is not None else "---") + " & "
             + (f"{r['drift']:.4f}" if r["drift"] is not None else "---") + " \\\\\n")
T[5] = r"""\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Model & Entropy (bits) & Top-1 $p$ & Frac.\ $p_{\max}>.99$ & Style gap (bits) & Change & Drift (bits) \\
\midrule
""" + rows + r"""\midrule
\emph{Teacher (style-blind base)} & \emph{""" + f"{O['teacher_entropy']:.3f}" + r"""} & --- & --- & --- & --- & --- \\
\bottomrule
\end{tabular}
\caption{\textbf{Soft-target self-distillation vs.\ hard-target SFT}, identical hyperparameters apart from the
loss. Forward KL is mass-covering, so the teacher's entropy is a floor for the soft-target student, which sits
just above it. Hard-target drift is $1.29\times$ the form-matched positive control (0.2341 bits, measured on
prompts with no style line); soft-target drift is $0.011\times$. Two hard-target runs at different learning
rates moved the style gap $-24.2\%$ (lr $10^{-4}$) and $+19.2\%$ (lr $3\times10^{-5}$) while drift was 0.302
and 0.3025: the distributional damage reproduces, the sensitivity metric does not reproduce in sign.}
\label{tab:objective}
\end{table*}"""

PR = D["probe"]
rows = ""
for i, L in enumerate(PR["layers"]):
    d = PR["delta_style_final"][i]
    rows += (f"{L} & {PR['base_style_final'][i]:.3f} & {PR['base_style_cue'][i]:.3f} & "
             f"{PR['base_style_final'][i]-d[0]:.3f} & {d[0]:+.3f} & [{d[1]:+.3f}, {d[2]:+.3f}] & "
             f"{PR['perm_p_style'][i]:.3f} \\\\\n")
T[6] = r"""\begin{table}[t]\centering\footnotesize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}rrrrrcr@{}}
\toprule
& \multicolumn{2}{c}{Base AUC} & After & & & \\
\cmidrule(lr){2-3}
Layer & final & cue & final & $\Delta$ & 95\% CI & $p$ \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}
\caption{\textbf{Linear probe for style, before and after intervention.} Intervals are cluster bootstraps
resampling the (dimension, phrasing) unit --- 15 held-out units, not 1{,}080 items --- and $p$ is a cluster
permutation test exchanging the base/intervened assignment for whole units. No layer shows both an interval
excluding zero and $p<.05$. With 95\% confidence no layer's style-decodability drop exceeds 0.107 AUC.
Cue-position readout exceeds final-position readout by $+0.064$ AUC on average through layer 20 in the base
model.}
\label{tab:probe}
\end{table}"""

for n, tex in T.items():
    open(os.path.join(_HERE, f"tables/table{n}.tex"),"w").write(tex)
print("wrote", len(T), "LaTeX tables")
