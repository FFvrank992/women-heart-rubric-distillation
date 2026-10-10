"""Does QLoRA actually fit and run on this machine? Twenty steps, real shapes.

    python smoke_train.py                       # Qwen3-1.7B, the safe rung
    python smoke_train.py --model Qwen/Qwen3-4B-Instruct-2507
    python smoke_train.py --seq-len 2048 --batch-size 2

Phase 1 of the plan: prove the stack works before any of the real data depends
on it. The three things that break on this setup, in order of likelihood:

1. **bitsandbytes on Windows.** Needs a build with CUDA kernels; a CPU-only
   wheel imports fine and then fails at the first 4-bit forward pass. This
   script checks the kernels, not just the import.
2. **Version drift.** transformers 5.x is recent and peft/trl move fast. The
   versions that worked are printed at the end for pinning.
3. **8 GB of VRAM**, of which ~1.3 GB is usually already taken by the desktop.
   The peak is measured and extrapolated to the real model, rather than
   guessed.

The examples are built from the *real* teacher prompt and a real schema answer,
so sequence lengths — and therefore the memory figure — mean something. Loss is
masked to the answer tokens only, which is what the real training does.

TRL is checked for import and version but not used to run the steps: its API
surface moves between releases, and a mismatch there would block the memory
measurement, which is the thing this script exists to produce. The real trainer
can use it for assistant-only loss once the stack is known to work.
"""
from __future__ import annotations

import argparse
import json
import os
import time

# Must be set before torch allocates anything. Windows fragments badly at 8 GB
# without it, and a fragmented allocator OOMs well below the real limit.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

import schema
from fewshot import EXAMPLES
from teacher import build_system, student_system

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def report_environment() -> dict:
    import transformers

    versions = {"torch": torch.__version__, "transformers": transformers.__version__}
    print("ENVIRONMENT")
    for name in ("peft", "trl", "bitsandbytes", "accelerate", "datasets"):
        try:
            versions[name] = __import__(name).__version__
        except Exception as exc:                  # noqa: BLE001 - report, do not crash
            versions[name] = f"MISSING ({type(exc).__name__})"
    for name, version in versions.items():
        print(f"  {name:<14} {version}")

    if not torch.cuda.is_available():
        raise SystemExit("\nCUDA is not available: nothing to measure.")
    free, total = torch.cuda.mem_get_info()
    print(f"  {'gpu':<14} {torch.cuda.get_device_name(0)}")
    print(f"  {'vram':<14} {free / 2**30:.2f} GiB free of {total / 2**30:.2f} GiB")
    if free / 2**30 < 5.5:
        print("\n  WARNING: under 5.5 GiB free. Close the browser and anything else")
        print("  using the GPU, or the measurement below will not be representative.")
    return versions


def check_bitsandbytes() -> None:
    """Import is not enough: a CPU-only wheel fails at the first 4-bit matmul."""
    import bitsandbytes as bnb

    print("\nBITSANDBYTES")
    try:
        weight = torch.randn(256, 256, dtype=torch.float16, device="cuda")
        quantised, state = bnb.functional.quantize_4bit(weight, quant_type="nf4")
        restored = bnb.functional.dequantize_4bit(quantised, state)
        error = (restored.float() - weight.float()).abs().mean().item()
        print(f"  4-bit quantise/dequantise on GPU ok, mean abs error {error:.4f}")
    except Exception as exc:                      # noqa: BLE001 - this is the check
        raise SystemExit(
            f"  4-bit kernels do not work: {type(exc).__name__}: {exc}\n\n"
            f"  This is the usual Windows failure. Try:\n"
            f"      pip install -U bitsandbytes\n"
            f"  and if that wheel is CPU-only, WSL2 is the reliable fallback.") from exc


def build_examples(tokenizer, count: int, seq_len: int,
                   student: bool = True) -> list[dict]:
    """Prompt/answer pairs in the real format, with loss on the answer only."""
    system = student_system() if student else build_system()
    system_tokens = tokenizer(system, add_special_tokens=False)["input_ids"]
    rows = []
    for i in range(count):
        example = EXAMPLES[i % len(EXAMPLES)]
        user = (f"Platform: Inspire  Type: comment\n"
                f"Post: {example['text']}\nAnswer:")
        answer = json.dumps({k: v for k, v in example["labels"].items()
                             if k in schema.NAMES}, ensure_ascii=False)

        # Render to text, then tokenise. apply_chat_template(tokenize=True)
        # returns a BatchEncoding on transformers 5.x and a plain list on 4.x;
        # going through text sidesteps the difference. enable_thinking is a
        # Qwen3 extension - the hybrid models emit <think> blocks without it,
        # which the -Instruct-2507 checkpoints do not accept at all.
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer + tokenizer.eos_token,
                               add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + answer_ids)[:seq_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:seq_len]
        rows.append({"input_ids": input_ids, "labels": labels,
                     "attention_mask": [1] * len(input_ids)})
    lengths = [len(r["input_ids"]) for r in rows]
    print(f"\n  {count} examples, {min(lengths)}-{max(lengths)} tokens "
          f"(mean {sum(lengths) / len(lengths):.0f}), loss on the answer only")
    return rows


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(row["input_ids"]) for row in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for row in batch:
        gap = width - len(row["input_ids"])
        out["input_ids"].append(row["input_ids"] + [pad_id] * gap)
        out["labels"].append(row["labels"] + [-100] * gap)
        out["attention_mask"].append(row["attention_mask"] + [0] * gap)
    return {k: torch.tensor(v) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--teacher-prompt", action="store_true",
                        help="use the full rubric prompt instead of the short one")
    args = parser.parse_args()

    versions = report_environment()
    if any(str(v).startswith("MISSING") for v in versions.values()):
        raise SystemExit("\nInstall the missing packages first:\n"
                         "  pip install peft trl bitsandbytes accelerate datasets")
    check_bitsandbytes()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    print(f"\nLOADING {args.model} in 4-bit NF4")
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    weights_gib = torch.cuda.memory_allocated() / 2**30
    print(f"  loaded in {time.time() - started:.0f}s, weights {weights_gib:.2f} GiB")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA r={args.lora_r} on {len(TARGET_MODULES)} projections: "
          f"{trainable / 1e6:.1f}M trainable of {total / 1e6:.0f}M "
          f"({trainable / total:.2%})")

    rows = build_examples(tokenizer, args.steps * args.batch_size,
                          args.seq_len, student=not args.teacher_prompt)

    from torch.optim import AdamW
    try:
        import bitsandbytes as bnb
        optimiser = bnb.optim.PagedAdamW8bit(
            [p for p in model.parameters() if p.requires_grad], lr=2e-4)
        optimiser_name = "PagedAdamW8bit"
    except Exception:                             # noqa: BLE001 - fall back and say so
        optimiser = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        optimiser_name = "AdamW (paged 8-bit unavailable)"

    print(f"\nTRAINING {args.steps} steps, batch {args.batch_size} x "
          f"accum {args.grad_accum}, seq<= {args.seq_len}, {optimiser_name}")
    model.config.use_cache = False
    model.train()
    torch.cuda.reset_peak_memory_stats()

    started, losses = time.time(), []
    for step in range(args.steps):
        batch = collate(rows[step * args.batch_size:(step + 1) * args.batch_size],
                        tokenizer.pad_token_id)
        batch = {k: v.to("cuda") for k, v in batch.items()}
        loss = model(**batch).loss / args.grad_accum
        loss.backward()
        losses.append(loss.item() * args.grad_accum)
        if (step + 1) % args.grad_accum == 0:
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
        if (step + 1) % 5 == 0:
            print(f"  step {step + 1:3d}/{args.steps}  loss {losses[-1]:.3f}  "
                  f"peak {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")

    seconds = time.time() - started
    peak = torch.cuda.max_memory_reserved() / 2**30
    _, total_vram = torch.cuda.mem_get_info()
    print(f"\nRESULT")
    print(f"  loss            {losses[0]:.3f} -> {losses[-1]:.3f}")
    print(f"  peak VRAM       {peak:.2f} GiB of {total_vram / 2**30:.2f} GiB "
          f"({peak / (total_vram / 2**30):.0%})")
    print(f"  per step        {seconds / args.steps:.2f}s")
    print(f"  headroom        {total_vram / 2**30 - peak:.2f} GiB")

    # 8,300 training rows at 3 epochs is the planned real run.
    rows_real, epochs = 8300, 3
    hours = rows_real * epochs * (seconds / args.steps) / 3600
    print(f"\n  Extrapolated: {rows_real:,} rows x {epochs} epochs "
          f"= {hours:.1f} hours on this model.")
    if "1.7B" in args.model:
        print(f"  Qwen3-4B holds ~2.4 GiB of weights against this model's "
              f"{weights_gib:.2f} GiB, so expect roughly "
              f"{peak + (2.4 - weights_gib):.1f} GiB peak and ~2.5x the time.")

    print("\n  Versions that worked, for requirements-train.txt:")
    for name, version in versions.items():
        print(f"    {name}=={version}")


if __name__ == "__main__":
    main()
