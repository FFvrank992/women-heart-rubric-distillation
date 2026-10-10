"""QLoRA fine-tune on the teacher's labels.

    python train.py                          # the real run
    python train.py --rows 500 --tag curve500 # a point on the learning curve

Reads `data/sft/{train,val}.jsonl`, trains a LoRA adapter on a 4-bit base, and
writes it with its metrics to `runs/<timestamp>_<tag>/`.

Settings come from the smoke test rather than from guesswork: on this 8 GB card
Qwen3-4B in NF4 peaks at 4.31 GiB with batch 1, seq 1024 and gradient
checkpointing, leaving enough headroom that seq 2048 or batch 2 would also fit.

Loss is on the assistant tokens only. The prompt is the 142-token student
prompt, never the teacher's 2,637-token rubric: training on the latter overran
the window in the smoke test, masked every answer token and produced nan.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

import config
import schema

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encode(example: dict, tokenizer, seq_len: int) -> dict | None:
    """Tokenise one exchange, masking everything before the assistant turn."""
    messages = example["messages"]
    try:
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(messages[-1]["content"] + tokenizer.eos_token,
                           add_special_tokens=False)["input_ids"]
    if len(prompt_ids) >= seq_len:
        return None                        # nothing left to supervise
    ids = (prompt_ids + answer_ids)[:seq_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:seq_len]
    if all(v == -100 for v in labels):
        return None
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(row["input_ids"]) for row in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for row in batch:
        gap = width - len(row["input_ids"])
        out["input_ids"].append(row["input_ids"] + [pad_id] * gap)
        out["labels"].append(row["labels"] + [-100] * gap)
        out["attention_mask"].append(row["attention_mask"] + [0] * gap)
    return {k: torch.tensor(v) for k, v in out.items()}


@torch.no_grad()
def validation_loss(model, rows, tokenizer, batch_size: int) -> float:
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(rows), batch_size):
        batch = collate(rows[start:start + batch_size], tokenizer.pad_token_id)
        batch = {k: v.to("cuda") for k, v in batch.items()}
        total += model(**batch).loss.item() * len(batch["input_ids"])
        count += len(batch["input_ids"])
    model.train()
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--tag", default="sft")
    parser.add_argument("--rows", type=int, default=0, help="cap the training rows")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers import get_cosine_schedule_with_warmup

    out_dir = config.new_run(args.tag)
    print(f"run: {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft = config.DATA / "sft"
    train_raw = load_jsonl(sft / "train.jsonl")
    val_raw = load_jsonl(sft / "val.jsonl")
    if args.rows:
        train_raw = train_raw[: args.rows]

    train = [e for e in (encode(x, tokenizer, args.seq_len) for x in train_raw) if e]
    val = [e for e in (encode(x, tokenizer, args.seq_len) for x in val_raw) if e]
    lengths = [len(r["input_ids"]) for r in train]
    print(f"  train {len(train):,}  val {len(val):,}  "
          f"tokens {min(lengths)}-{max(lengths)} (mean {sum(lengths)/len(lengths):.0f})")
    if len(train) < len(train_raw):
        print(f"  {len(train_raw) - len(train)} examples dropped: prompt filled the window")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA r={args.lora_r}: {trainable / 1e6:.1f}M trainable")

    import bitsandbytes as bnb
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = bnb.optim.PagedAdamW8bit(params, lr=args.lr)

    steps_per_epoch = max(1, len(train) // args.batch_size)
    total_steps = int(steps_per_epoch * args.epochs)
    updates = max(1, total_steps // args.grad_accum)
    scheduler = get_cosine_schedule_with_warmup(
        optimiser, num_warmup_steps=max(1, int(updates * 0.03)),
        num_training_steps=updates)

    print(f"  {total_steps:,} steps ({updates:,} updates), batch {args.batch_size} "
          f"x accum {args.grad_accum}, lr {args.lr}")
    model.config.use_cache = False
    model.train()
    torch.cuda.reset_peak_memory_stats()

    history, started, step = [], time.time(), 0
    order = list(range(len(train)))
    for epoch in range(int(args.epochs) + 1):
        torch.manual_seed(config.SEED + epoch)
        shuffled = [train[i] for i in torch.randperm(len(order)).tolist()]
        for start in range(0, len(shuffled) - args.batch_size + 1, args.batch_size):
            if step >= total_steps:
                break
            batch = collate(shuffled[start:start + args.batch_size], tokenizer.pad_token_id)
            batch = {k: v.to("cuda") for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)
            if step % 100 == 0:
                rate = step / (time.time() - started)
                print(f"  step {step:,}/{total_steps:,}  loss "
                      f"{loss.item() * args.grad_accum:.3f}  "
                      f"peak {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB  "
                      f"~{(total_steps - step) / rate / 60:.0f} min left", flush=True)
            if step % 500 == 0 or step == total_steps:
                value = validation_loss(model, val, tokenizer, args.batch_size)
                history.append({"step": step, "val_loss": round(value, 4)})
                print(f"    val loss {value:.4f}", flush=True)
        if step >= total_steps:
            break

    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    metrics = {
        "model": args.model, "tag": args.tag,
        "train_rows": len(train), "val_rows": len(val),
        "epochs": args.epochs, "steps": total_steps,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "lr": args.lr, "lora_r": args.lora_r, "seq_len": args.seq_len,
        "fields": list(schema.NAMES),
        "peak_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "minutes": round((time.time() - started) / 60, 1),
        "val_loss_history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nsaved {out_dir / 'adapter'}")
    print(f"  peak {metrics['peak_gib']} GiB, {metrics['minutes']} min, "
          f"final val loss {history[-1]['val_loss'] if history else 'n/a'}")


if __name__ == "__main__":
    main()
