"""Assemble the ACL-format two-column LaTeX paper end to end."""
import io, os, re, json, subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
def _p(*parts): return os.path.join(_ROOT, *parts)
_BUILD = os.path.join(_HERE, "_build")
os.makedirs(_BUILD, exist_ok=True)
def _b(name): return os.path.join(_BUILD, name)

md = io.open(_p("paper", "paper.md"), encoding="utf8").read()

# ---- strip the markdown title block; drop Appendix K (tables live inline now)
md = md[md.index("## Abstract"):]
if "### K  Full tables" in md:
    md = md[:md.index("### K  Full tables")]

abstract = md[md.index("## Abstract")+len("## Abstract"):md.index("## 1  Introduction")].strip()
body = md[md.index("## 1  Introduction"):]

FIG = {1:"fig1_method",2:"fig2_baseline",3:"fig3_cue_robustness",4:"fig4_placebo_robustness",
       5:"fig5_seeds",6:"fig6_before_after",7:"fig7_distribution",8:"fig8_drift",
       9:"fig9_hard_instability",10:"fig10_probe",11:"fig11_summary"}
CAP = json.load(open(os.path.join(_HERE, "captions.json")))
CAP["fig1_method"] = ("**Figure 1. The measurement, the intervention, and the three evaluations.** "
 "(a) Every comparison holds the propositional content byte-identical and changes one sentence. The style "
 "contrast is paired with a placebo contrast matched on the exact token length of both sides and on cosine "
 "distance in input-embedding space, so a change in prompt length cancels in the difference; the positive "
 "control replaces all content and puts every effect on an interpretable scale. (b) The teacher is the frozen "
 "base model on the cue-free prompt and the student is the LoRA-adapted model on the cue-present prompt — "
 "context distillation with the context moved to the student's side. (c) No single evaluation is sufficient: "
 "behaviour can fall because the model was damaged, so the distributional diagnostics are part of the "
 "evaluation rather than supplementary. Schematic; carries no measured values.")

def cited(line, n):
    for m in re.finditer(r"(?:Figure|Table)s?\s+(\d+)\s*(?:[–-]\s*(\d+))?", line):
        lo=int(m.group(1)); hi=int(m.group(2)) if m.group(2) else lo
        if lo <= n <= hi: return True
    return False

# ---- markdown -> latex fragment, with float placeholders
# Placeholders must never land inside a list item (pandoc mangles them and a float
# cannot be emitted mid-list), so citations found in a list are buffered and
# flushed at the next ordinary paragraph.
placed_f, placed_t, out, pending = set(), set(), [], []
LIST = re.compile(r"^\s*(?:\d+\.|[-*+])\s")
for ln in body.split("\n"):
    out.append(ln)
    skip = (ln.startswith("|") or ln.startswith("#") or not ln.strip()
            or bool(LIST.match(ln)) or ln.startswith("   "))
    if not skip:
        for n in sorted(FIG):
            if n not in placed_f and re.search(r"Figures?\s", ln) and cited(ln, n):
                pending.append(f"ZFIGZ{n}Z"); placed_f.add(n)
        for n in range(1, 7):
            if n not in placed_t and re.search(r"Tables?\s", ln) and cited(ln, n):
                pending.append(f"ZTABZ{n}Z"); placed_t.add(n)
    if pending and not skip and not LIST.match(ln) and not ln.startswith("   "):
        out += [""] + sum(([tok, ""] for tok in pending), []); pending = []
if pending:
    out += [""] + sum(([tok, ""] for tok in pending), [])
body = "\n".join(out)
print("figures placed:", sorted(placed_f), "| tables placed:", sorted(placed_t))

io.open(_b("_body.md"),"w",encoding="utf8").write(body)
io.open(_b("_abs.md"),"w",encoding="utf8").write(abstract)
for src, dst in (("_body.md","_body.tex"), ("_abs.md","_abs.tex")):
    cmd = ["pandoc", _b(src), "-t", "latex", "-o", _b(dst)]
    if src == "_body.md":
        cmd.insert(2, "--shift-heading-level-by=-1")   # paper.md starts at '##'
    subprocess.run(cmd, check=True)

tex = io.open(_b("_body.tex"), encoding="utf8").read()
abs_tex = io.open(_b("_abs.tex"), encoding="utf8").read()

# ---- substitute real floats
for n, fn in FIG.items():
    cap = CAP[fn]
    # \caption already emits "Figure N."; drop the label carried in the caption text
    cap = re.sub(r"^\*\*Figure\s+\d+\.\s*", "**", cap)
    cap = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", cap)
    cap = cap.replace("**","").replace("&","\\&").replace("%","\\%").replace("_","\\_")
    cap = re.sub(r"\*(.+?)\*", r"\\emph{\1}", cap)
    star = "*"                       # every figure is wider than one column
    # Figure 1 is near-square; at full width it exceeds \dbltopfraction and would
    # force every later float onto a float page, so it is scaled to fit a page top.
    w = "0.72\\linewidth" if n == 1 else "\\linewidth"
    # floats are not placed in numeric order (5.6 cites Fig. 10 before 5.7 cites
    # Fig. 7), so pin the counter to the number used in the prose
    blk = ("\\begin{figure%s}[tbp]\\centering\n\\includegraphics[width=%s]{%s}\n"
           "\\setcounter{figure}{%d}\n\\caption{%s}\n\\label{fig:%d}\n\\end{figure%s}") % (
           star, w, _p("figures", f"{fn}.pdf"), n - 1, cap, n, star)
    tex = tex.replace(f"ZFIGZ{n}Z", blk)
for n in range(1, 7):
    tex = tex.replace(f"ZTABZ{n}Z", io.open(os.path.join(_HERE, f"tables/table{n}.tex"), encoding="utf8").read())
tex = re.sub(r"Z(FIG|TAB)Z\d+Z", "", tex)
tex = re.sub(r"\\begin\{center\}\\rule\{[^}]*\}\{[^}]*\}\\end\{center\}", "", tex)

# pandoc longtables (appendix) cannot run in two-column mode -> full-width table* floats
def _longtable_to_float(m):
    blk = m.group(0)
    spec_end = blk.index("}}", blk.index("{@{}")) + 2
    spec = blk[blk.index("[]{") + 3: spec_end - 1]        # column specification
    inner = blk[spec_end:blk.rindex("\\end{longtable}")]
    spec = spec.replace("\\columnwidth", "\\textwidth")
    inner = inner.replace("\\columnwidth", "\\textwidth")
    # longtable puts the footer rule before the body; tabular wants it at the end
    inner = inner.replace("\\bottomrule\\noalign{}\n\\endlastfoot", "")
    inner = inner.replace("\\bottomrule\\noalign{}", "").replace("\\endlastfoot", "")
    inner = inner.replace("\\endhead", "").replace("\\endfirsthead", "")
    inner = inner.replace("\\noalign{}", "")
    inner = inner.strip().rstrip("\\\\").rstrip()
    ncol = spec.count(">{") + len([c for c in spec if c in "lrc"])
    size = "\\footnotesize" if ncol >= 5 else "\\small"
    return ("\\begin{table*}[t]\n\\centering" + size + "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{" + spec + "}\n"
            + inner + "\\\\\n\\bottomrule\n\\end{tabular}\n\\end{table*}")

n_lt = len(re.findall(r"\\begin\{longtable\}", tex))
tex = re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}", _longtable_to_float, tex, flags=re.S)
assert "\\begin{longtable}" not in tex
print("appendix longtables converted to table*:", n_lt)

TPL = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm,columnsep=0.6cm]{geometry}
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\setsansfont{TeX Gyre Heros}
\setmonofont{DejaVu Sans Mono}[Scale=0.85]
\usepackage{calc}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs,multirow,array}
\usepackage[font=small,labelfont=bf,labelsep=period]{caption}
\usepackage{titlesec}
\usepackage{microtype}
\usepackage{longtable}
\usepackage{stfloats}
% float placement: 17 full-width floats over a short results section, so the
% defaults (which reserve most of a page for text) must be relaxed.
\setcounter{topnumber}{3}\setcounter{bottomnumber}{2}\setcounter{totalnumber}{5}
\setcounter{dbltopnumber}{3}
\renewcommand{\topfraction}{0.92}\renewcommand{\bottomfraction}{0.75}
\renewcommand{\dbltopfraction}{0.92}\renewcommand{\textfraction}{0.06}
\renewcommand{\floatpagefraction}{0.75}\renewcommand{\dblfloatpagefraction}{0.75}
\usepackage[hidelinks]{hyperref}
\usepackage{url}
\usepackage{enumitem}
\setlist{nosep,leftmargin=1.2em}
\setcounter{secnumdepth}{0}
\titleformat{\section}{\normalfont\fontsize{12}{14}\bfseries}{}{0em}{}
\titleformat{\subsection}{\normalfont\fontsize{11}{13}\bfseries}{}{0em}{}
\titleformat{\subsubsection}{\normalfont\fontsize{11}{13}\bfseries\itshape}{}{0em}{}
\titlespacing*{\section}{0pt}{2.0ex plus .6ex minus .2ex}{1.0ex plus .2ex}
\titlespacing*{\subsection}{0pt}{1.6ex plus .5ex minus .2ex}{0.8ex plus .2ex}
\setlength{\parindent}{1em}
\setlength{\emergencystretch}{3em}
\sloppy
\renewcommand{\UrlFont}{\small\ttfamily}
\pagestyle{plain}
\begin{document}
\twocolumn[
\begin{center}
{\fontsize{15}{18}\selectfont\bfseries Speaker-Style Sensitivity in Language Models:\\[2pt]
Controlled Measurement, Distribution-Preserving Removal, and What Survives It\par}
\vspace{5mm}
{\fontsize{11}{13}\selectfont [AUTHOR NAME]\par}
\vspace{1mm}
{\fontsize{10}{12}\selectfont [AFFILIATION] \quad\textbullet\quad \texttt{[EMAIL]}\par}
\end{center}
\vspace{7mm}
]
{\begin{center}{\fontsize{12}{14}\selectfont\bfseries Abstract}\end{center}
\vspace{-1mm}
\begingroup\small\leftskip0.6cm\rightskip0.6cm
%%ABSTRACT%%
\par\endgroup}
\vspace{4mm}

%%BODY%%

\end{document}
"""
doc = TPL.replace("%%ABSTRACT%%", abs_tex).replace("%%BODY%%", tex)
doc = re.sub(r"\\begin\{center\}\\rule\{[^}]*\}\{[^}]*\}\\end\{center\}", "", doc)
io.open(_p("paper", "acl_paper.tex"),"w",encoding="utf8").write(doc)
print("wrote paper/acl_paper.tex")
