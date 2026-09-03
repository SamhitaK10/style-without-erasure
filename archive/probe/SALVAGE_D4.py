# ============ PASTE AS A NEW CELL, SAME RUNTIME, BEFORE ANYTHING ELSE ==========
# The D4 run finished; only the final json.dump crashed on a variable I deleted
# when I rewrote the verdict block. Everything is still in memory. This saves it.
import json, os, subprocess, numpy as np
ll_late = float("nan")          # log-loss term was dropped in the D4 rewrite
out = dict(
    layers=LAYERS, base=base_tbl,
    deltas={str(k): {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in tbl.items()},
    per_item_scores={str(k): v["scores"] for k, v in tbl.items()},
    late_means=dict(style_final=sf, style_cue=sc_, urgency_final=uf,
                    urgency_logloss_increase=ll_late),
    permutation_p={str(k): dict(style=v["perm_p_style"], urgency=v["perm_p_urgency"])
                   for k, v in tbl.items()},
    n_style_eval=int(s_te.sum()), n_content_eval=int(c_te.sum()),
    n_units_style=int(len(set(style_unit[s_te]))),
    n_units_content=int(len(set(content_unit[c_te]))),
    mean_cue_minus_final_base=cue_gain, verdict=verdict,
    KNOWN_FLAW=("style clusters were keyed on the phrasing INDEX alone, so all five "
                "dimensions sharing an index collapsed into one unit: 3 clusters, not 15. "
                "The style intervals here are far too wide and the permutation test has "
                "almost no power (2^3 sign patterns). Do not report the style null from "
                "this run. The content side, keyed on scenario, has 18 units and is valid."))
json.dump(out, open("exp_D4_probe.json", "w"), indent=2)
for d in ["/content/drive/MyDrive/DiaLense_PartII", "/kaggle/working/DiaLense_PartII"]:
    if os.path.isdir(d): subprocess.run(["cp", "exp_D4_probe.json", d + "/"], check=False)
print("saved exp_D4_probe.json")
print(f"  style units  : {out['n_units_style']}  <- should be 15, see KNOWN_FLAW")
print(f"  content units: {out['n_units_content']}")
