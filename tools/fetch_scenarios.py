"""Fetch the 50 clinical scenarios this paper uses, from the DiaLense repository.

WHAT THIS DOES, AND WHY IT EXISTS
    Every experiment in this repository currently begins with

        git clone --depth 1 https://github.com/SamhitaK10/DiaLense.git

    and then reads DiaLense/results/raw/*/conversations.jsonl, deduplicating by
    scenario_id and rendering each record's latent_facts as a single block:

        "Reported history: <fact>; <fact>; ..."

    That inline clone always takes HEAD, so the extraction is not deterministic
    across time. This script is that same loader, lifted out and pinned to a
    specific commit. The rendering logic is byte-for-byte the logic in
    experiments/07_selfdistill_seeds.py (lines 145-155 of the original script);
    nothing about the content has been changed.

PROVENANCE
    The scenarios are CC BY 4.0 adaptations of MTS-Dialog records. See
    data/README.md for the full chain, the attribution you must carry, and the
    one gap (scenarios S1-S10 have no recorded source record).

USAGE
    python tools/fetch_scenarios.py
    python tools/fetch_scenarios.py --ref <sha-or-tag> --out data/scenarios.jsonl
"""

import argparse
import glob
import json
import os
import subprocess
import sys

REPO = "https://github.com/SamhitaK10/DiaLense.git"

# Pinned so the extraction is deterministic. This is the DiaLense commit that
# was verified to yield exactly the 50 scenarios the paper's 30/20 split assumes.
#
# It is NOT a claim that this commit is the one the original experiments ran
# against: the DiaLense repository has a single squashed commit, so its history
# cannot distinguish the state at experiment time from the state today. What the
# pin guarantees is that everyone who runs this script from now on extracts the
# same 50 scenarios. Pass --ref to override.
DEFAULT_REF = "a1adecdd31fa6905583f7beb79e58eb4b062bc06"
EXPECTED_N_SCENARIOS = 50


def _head(dest):
    return subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def clone(dest, ref):
    """Clone DiaLense at `ref`. Works for a tag, a branch or a full commit SHA."""
    if os.path.isdir(dest):
        print(f"  {dest} already present, not re-cloning")
        return _head(dest)
    if ref and len(ref) == 40 and all(c in "0123456789abcdef" for c in ref):
        # `git clone --branch` does not accept a SHA; init + fetch does.
        subprocess.run(["git", "init", "--quiet", dest], check=True)
        subprocess.run(["git", "-C", dest, "remote", "add", "origin", REPO], check=True)
        subprocess.run(["git", "-C", dest, "fetch", "--quiet", "--depth", "1",
                        "origin", ref], check=True)
        subprocess.run(["git", "-C", dest, "checkout", "--quiet", "FETCH_HEAD"], check=True)
    else:
        cmd = ["git", "clone", "--quiet", "--depth", "1"]
        if ref:
            cmd += ["--branch", ref]
        subprocess.run(cmd + [REPO, dest], check=True)
    sha = _head(dest)
    print(f"  cloned at {sha}")
    return sha


def extract(repo_dir):
    """Deduplicate by scenario_id and render each latent-fact list to one block."""
    rows, seen = [], set()
    pattern = os.path.join(repo_dir, "results", "raw", "*", "conversations.jsonl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"no conversation files matched {pattern}")
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            sid = r["scenario_id"]
            if sid in seen:
                continue
            seen.add(sid)
            facts = "; ".join(str(v).strip().rstrip(".")
                              for v in r["latent_facts"].values())
            doctor_turn = None
            for t in r["dialogue_turns"]:
                if t["speaker"] == "Doctor" and len(t["text"].split()) > 6:
                    doctor_turn = t["text"]
                    break
            rows.append({
                "scenario_id": sid,
                "content_block": "Reported history: " + facts + ".",
                "first_doctor_turn": doctor_turn,
                "source_file": os.path.relpath(p, repo_dir),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default=DEFAULT_REF,
                    help="DiaLense tag, branch or 40-character commit SHA "
                         f"(default: {DEFAULT_REF})")
    ap.add_argument("--repo-dir", default=".dialense_cache",
                    help="where to clone DiaLense")
    ap.add_argument("--out", default="data/scenarios.jsonl")
    a = ap.parse_args()

    print(f"fetching DiaLense (ref={a.ref or 'UNPINNED default branch'})")
    sha = clone(a.repo_dir, a.ref)

    rows = extract(a.repo_dir)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {"repo": REPO, "ref": a.ref, "commit": sha, "n_scenarios": len(rows)}
    with open(a.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    print(f"wrote {len(rows)} scenarios to {a.out}")
    if len(rows) != EXPECTED_N_SCENARIOS:
        sys.exit(f"  ERROR: expected {EXPECTED_N_SCENARIOS} scenarios, got {len(rows)}.\n"
                 "  The paper's 30 train / 20 held-out split assumes 50. Check --ref.")
    if not a.ref:
        print("  WARNING: --ref was set to empty, so this extraction is not pinned "
              "and may drift as DiaLense changes.")


if __name__ == "__main__":
    main()
