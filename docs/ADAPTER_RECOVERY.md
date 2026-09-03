# Adapter recovery and validation

The initial repository audit treated the soft-target LoRA checkpoint as lost
because the Colab runtime copy had disappeared and `results/finetune_v2_ACCEPTED.json`
contains a historical `_adapter_lost` note. A retained Google Drive backup of the
`dialense_lora` directory was later found.

## Validation

The recovered checkpoint was checked with `tools/publish_adapter.py --dry-run`
against the configuration recorded in `results/finetune_v2_ACCEPTED.json`.

| Check | Recovered value | Expected | Status |
|---|---|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` | same | PASS |
| LoRA rank | 16 | 16 | PASS |
| LoRA alpha | 32 | 32 | PASS |
| Target modules | q/k/v/o + gate/up/down projections | same | PASS |
| PEFT type | LORA | LORA | PASS |
| Tensors | 392 | structurally consistent | PASS |
| Adapter parameters | 18,464,768 | 18,464,768 | PASS |
| Weight file size | 73,911,112 bytes | consistent with fp32 LoRA | PASS |

## File hashes

```text
adapter_config.json       884371482b02bb30481ffd2c6406cff933976b0e2a9e42505a6b683f06a21159
adapter_model.safetensors d48899aab39a1d18060825b2384cafcf889b4110020ba246824c572e11d12d8c
```

## What this proves

The recovered files match the architecture and training configuration reported
for the accepted soft-target run. They are suitable for release as the preserved
checkpoint associated with the project.

## What this does not prove

No cryptographic hash was recorded at training/evaluation time. Therefore the
repository cannot independently prove that these exact bytes are identical to
the bytes loaded during every reported evaluation. This limitation should stay
visible rather than being replaced with a stronger claim.

The raw accepted result artifact is intentionally left unchanged. Its
`_adapter_lost` field is historical metadata that was superseded by this recovery.
