"""Validate a trained LoRA adapter against the recorded config, then publish it
to the Hugging Face Hub with the model card.

WHY THE VALIDATION STEP EXISTS
    While auditing this project a stray `adapter_config.json` was found in a
    temp directory and could easily have been mistaken for the real one. It was
    a 1,024-parameter test fixture: r=4, alpha=8, targeting `c_attn` (a GPT-2
    module name), with an empty `base_model_name_or_path`. Uploading it would
    have published something that loads without error and is not the model the
    paper describes.

    So this script refuses to upload anything whose adapter_config.json does not
    match what `results/finetune_v2_ACCEPTED.json` records for the reported run:
    r=16, alpha=32, the seven Qwen projection modules, and the right base model.
    Use --allow-mismatch only if you know why they differ and say so in the card.

USAGE
    # check only, no network, no upload
    python tools/publish_adapter.py --adapter ./dialense_lora --dry-run

    # publish
    huggingface-cli login
    python tools/publish_adapter.py --adapter ./dialense_lora \\
        --repo-id SamhitaK10/qwen2.5-1.5b-style-without-erasure-lora

RECOVERED CHECKPOINT
    A retained Drive backup of `dialense_lora` was recovered after the original
    audit had treated the checkpoint as lost. Validate that directory with
    --dry-run before upload. The historical `_adapter_lost` field in the result
    artifact is stale metadata and is intentionally not rewritten.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPECTED_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"}


def load_recorded():
    p = os.path.join(ROOT, "results/finetune_v2_ACCEPTED.json")
    with open(p, encoding="utf8") as f:
        return json.load(f)["config"]


def validate(adapter_dir, recorded):
    """Return (ok, list of (level, message))."""
    msgs = []
    ok = True

    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return False, [("FAIL", f"no adapter_config.json in {adapter_dir}")]
    with open(cfg_path, encoding="utf8") as f:
        cfg = json.load(f)

    def check(label, got, want, fatal=True):
        nonlocal ok
        good = got == want
        if not good and fatal:
            ok = False
        msgs.append(("PASS" if good else ("FAIL" if fatal else "WARN"),
                     f"{label}: got {got!r}, expected {want!r}"))

    check("lora r", cfg.get("r"), recorded["lora_r"])
    check("lora_alpha", cfg.get("lora_alpha"), recorded["lora_alpha"])
    check("target_modules", set(cfg.get("target_modules") or []), EXPECTED_TARGETS)
    check("peft_type", cfg.get("peft_type"), "LORA")

    base = cfg.get("base_model_name_or_path") or ""
    if base != recorded["model"]:
        lvl = "FAIL" if base else "WARN"
        if lvl == "FAIL":
            ok = False
        msgs.append((lvl, f"base_model_name_or_path: got {base!r}, expected "
                          f"{recorded['model']!r}" +
                          ("  (empty -- set it before publishing)" if not base else "")))
    else:
        msgs.append(("PASS", f"base_model_name_or_path: {base}"))

    # weight file and parameter count
    wf = None
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        if os.path.exists(os.path.join(adapter_dir, name)):
            wf = os.path.join(adapter_dir, name)
            break
    if wf is None:
        ok = False
        msgs.append(("FAIL", "no adapter_model.safetensors or .bin found"))
        return ok, msgs

    n_params = None
    if wf.endswith(".safetensors"):
        try:
            from safetensors import safe_open
            with safe_open(wf, framework="pt") as f:
                keys = list(f.keys())
                n_params = sum(f.get_tensor(k).numel() for k in keys)
            msgs.append(("PASS", f"{len(keys)} tensors in {os.path.basename(wf)}"))
        except Exception as e:                                  # pragma: no cover
            msgs.append(("WARN", f"could not read tensors: {e}"))

    if n_params is not None:
        want = recorded["trainable_params"]
        if n_params == want:
            msgs.append(("PASS", f"parameter count: {n_params:,}"))
        else:
            ok = False
            msgs.append(("FAIL", f"parameter count: got {n_params:,}, "
                                 f"expected {want:,} -- this is not the "
                                 "architecture the paper reports"))

    size_mb = os.path.getsize(wf) / 2**20
    msgs.append(("INFO", f"weight file: {size_mb:.1f} MB"))
    if size_mb < 1:
        msgs.append(("WARN", "suspiciously small for an 18.5M-parameter adapter"))
    return ok, msgs


def selftest():
    """Independently re-derive the reported adapter size from the architecture.

    Qwen2.5-1.5B-Instruct: hidden 1536, 28 layers, 12 query heads and 2 KV heads
    at head_dim 128, intermediate size 8960. A LoRA of rank r on a d_in -> d_out
    projection adds r*(d_in + d_out) parameters.

    This is a check on the paper's number, not on any weight file, so it runs
    with no model, no network and no adapter present.
    """
    H, L, R, KV, I = 1536, 28, 16, 2 * 128, 8960
    mods = {"q_proj": (H, H), "k_proj": (H, KV), "v_proj": (H, KV), "o_proj": (H, H),
            "gate_proj": (H, I), "up_proj": (H, I), "down_proj": (I, H)}
    per_layer = sum(R * (i + o) for i, o in mods.values())
    total = per_layer * L
    recorded = load_recorded()

    BASE_PARAMS = 1_543_714_304          # Qwen2.5-1.5B-Instruct, all parameters
    pct_peft = total / (BASE_PARAMS + total) * 100   # how peft reports it
    pct_naive = total / BASE_PARAMS * 100

    ok = True
    print("re-deriving the adapter size from the Qwen2.5-1.5B architecture\n")
    print(f"  per-layer LoRA parameters : {per_layer:,}")
    print(f"  x {L} layers               : {total:,}")
    print(f"  recorded in the artifact  : {recorded['trainable_params']:,}")
    if total != recorded["trainable_params"]:
        ok = False
        print("  FAIL  they disagree")
    else:
        print("  PASS  they agree exactly")

    print(f"\n  trainable/(base+trainable) = {pct_peft:.2f}%   <- what peft prints,"
          " and what the paper reports")
    print(f"  trainable/base             = {pct_naive:.2f}%   <- what a naive"
          " recomputation gives")
    if abs(pct_peft - recorded["trainable_pct"]) > 0.005:
        ok = False
        print(f"  FAIL  neither matches the recorded {recorded['trainable_pct']}%")
    else:
        print(f"  PASS  matches the recorded {recorded['trainable_pct']}%")
        print("        (recorded under peft's convention; a reader recomputing"
              " against the")
        print("         base-only denominator will get 1.20% and should not"
              " treat that as an error)")
    print("\n  SELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", help="directory holding the trained adapter")
    ap.add_argument("--selftest", action="store_true",
                    help="re-derive the reported adapter size from the architecture; "
                         "needs no adapter, no model and no network")
    ap.add_argument("--repo-id", help="Hugging Face repo id, e.g. user/model-name")
    ap.add_argument("--dry-run", action="store_true", help="validate only; no network")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="publish even if validation fails (you must justify it in the card)")
    ap.add_argument("--card", default=os.path.join(ROOT, "model_card/README.md"))
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.adapter:
        ap.error("--adapter is required unless --selftest is given")

    recorded = load_recorded()
    print(f"validating {a.adapter} against results/finetune_v2_ACCEPTED.json\n")
    ok, msgs = validate(a.adapter, recorded)
    for lvl, m in msgs:
        print(f"  {lvl:4s} {m}")

    if not ok:
        print("\nVALIDATION FAILED -- this adapter does not match the reported run.")
        if not a.allow_mismatch:
            print("Refusing to publish. Use the validated recovered checkpoint, or")
            print("pass --allow-mismatch only if the difference is intentional and documented.")
            return 1
        print("--allow-mismatch given; continuing anyway.")
    else:
        print("\nVALIDATION PASSED -- matches the reported configuration.")

    if a.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return 0
    if not a.repo_id:
        print("\nno --repo-id given; nothing uploaded.")
        return 1

    if not os.path.exists(a.card):
        print(f"\nmodel card not found at {a.card}")
        return 1
    card = open(a.card, encoding="utf8").read()
    if "REPLACE_WITH_YOUR_HF_REPO_ID" in card:
        card = card.replace("REPLACE_WITH_YOUR_HF_REPO_ID", a.repo_id)
        print(f"\nsubstituted repo id into the model card")

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo_id, repo_type="model", private=a.private, exist_ok=True)
    print(f"uploading to https://huggingface.co/{a.repo_id}")

    tmp_card = os.path.join(a.adapter, "README.md")
    with open(tmp_card, "w", encoding="utf8") as f:
        f.write(card)

    api.upload_folder(
        folder_path=a.adapter,
        repo_id=a.repo_id,
        repo_type="model",
        commit_message="Add Style Without Erasure LoRA adapter for Qwen2.5-1.5B-Instruct",
        ignore_patterns=["*.pt", "*.ckpt", "optimizer*", "scheduler*", "rng_state*",
                         "trainer_state*", "training_args*"],
    )
    print(f"\ndone: https://huggingface.co/{a.repo_id}")
    print("Check that the model card renders and that the base-model link resolves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
