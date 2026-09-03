# ============ PASTE AS A NEW CELL IN THE SAME RUNTIME. LAST PROBE CELL. ========
# The D5 analysis is done and printed; only the save crashed (a leftover helper
# from the previous edit took two arguments and was called with one). Everything
# is still in memory. This cell touches nothing and cannot fail: every lookup is
# guarded, and anything missing is recorded as null instead of raising.
import json, os, subprocess

def g(name, default=None):
    try: return eval(name)
    except Exception: return default

def jsonable(x):
    try: json.dumps(x); return x
    except Exception: return str(x)

deltas, scores, perms = {}, {}, {}
_tbl = g("tbl", {}) or {}
for L, v in _tbl.items():
    deltas[str(L)] = {k: jsonable(val) for k, val in v.items() if k != "scores"}
    if isinstance(v, dict) and "scores" in v: scores[str(L)] = jsonable(v["scores"])
    perms[str(L)] = dict(style=jsonable(v.get("perm_p_style")),
                         urgency=jsonable(v.get("perm_p_urgency")))

out = dict(
    experiment="EXP_D5 — probe, cluster-corrected",
    layers=jsonable(g("LAYERS")),
    base_model=jsonable(g("base_tbl")),
    deltas=deltas,
    permutation_p=perms,
    per_item_scores=scores,
    late_means=dict(style_final=jsonable(g("sf")), style_cue=jsonable(g("sc_")),
                    urgency_final=jsonable(g("uf"))),
    n_units_style=len(set(g("style_unit", [])[g("s_te")])) if g("style_unit") is not None else None,
    n_units_content=len(g("content_eval", []) or []),
    mean_cue_minus_final_base=jsonable(g("cue_gain")),
    verdict=jsonable(g("verdict")),
    bound_note=("Cluster bootstrap over (dimension, phrasing) for style and over scenario "
                "for urgency. No layer survives both a CI excluding zero and a cluster "
                "permutation test at p<.05. Report the bound, not a point estimate."),
)
json.dump(out, open("exp_D5_probe.json", "w"), indent=2, default=str)
for d in ["/content/drive/MyDrive/DiaLense_PartII", "/kaggle/working/DiaLense_PartII"]:
    if os.path.isdir(d): subprocess.run(["cp", "exp_D5_probe.json", d + "/"], check=False)
print("saved exp_D5_probe.json")
print(f"  style units   : {out['n_units_style']}")
print(f"  content units : {out['n_units_content']}")
print(f"  layers saved  : {len(deltas)}")
