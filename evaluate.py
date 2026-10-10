"""Score a model against the blind gold set, per field and per source.

    python evaluate.py --mode zero-short          # base model, 142-token prompt
    python evaluate.py --mode zero-rubric         # base model, 2,637-token prompt
    python evaluate.py --mode adapter --adapter runs/<run>/adapter
    python evaluate.py --compare                  # the table of everything scored

Three settings over the *same base weights*, which is what makes the comparison
clean:

| setting      | input tokens | what it shows                        |
|--------------|--------------|--------------------------------------|
| zero-short   | 142 + doc    | the task is not in the weights       |
| zero-rubric  | 2,637 + doc  | in-context learning; this is the teacher |
| adapter      | 142 + doc    | the rubric distilled into the weights |

Results go to `runs/eval/<mode>.json` so `--compare` can lay them side by side.

Every figure is split by how its gold cell was produced: `agreed` cells were
reached by two annotators independently, `adjudicated` ones were settled in the
calibration session and are the contested rows by construction. Fields whose
disagreements were never settled are scored on their agreed subset only, which
makes them optimistic, and the caveat is printed with them rather than left to
a footnote.
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from sklearn.metrics import f1_score

import config
import schema
from teacher import build_system, build_user, ollama_provider, parse_reply, student_system

EVAL_DIR = config.RUNS / "eval"


def gold_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = config.GOLD / "gold_blind.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run `python calibrate.py --apply` first.")
    gold = pd.read_csv(path, dtype=str).fillna("")
    gold.columns = [c.lstrip("﻿") for c in gold.columns]
    sources = pd.read_csv(config.GOLD / "gold_blind_sources.csv", dtype=str).fillna("")
    sources.columns = [c.lstrip("﻿") for c in sources.columns]
    return gold.set_index("id"), sources.set_index("id")


def predict_ollama(rows: pd.DataFrame, system: str, model: str, workers: int) -> dict:
    """Base-model predictions through Ollama, in parallel like the teacher run."""
    from concurrent.futures import ThreadPoolExecutor

    ask = ollama_provider(model, system, temperature=0.0)
    users = {row["id"]: build_user(row) for _, row in rows.iterrows()}

    def one(item):
        row_id, user = item
        try:
            return row_id, parse_reply(ask(user))[0]
        except Exception:                          # noqa: BLE001 - a miss is a wrong answer
            return row_id, schema.blank()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, users.items()))


def predict_adapter(rows: pd.DataFrame, system: str, base: str, adapter: str) -> dict:
    """Predictions from the fine-tuned adapter, same prompt as training."""
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    out = {}
    for n, (_, row) in enumerate(rows.iterrows(), start=1):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": build_user(row)}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            generated = model.generate(**ids, max_new_tokens=200, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
        reply = tokenizer.decode(generated[0][ids["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        out[row["id"]] = parse_reply(reply)[0]
        if n % 25 == 0:
            print(f"  ({n}/{len(rows)})", flush=True)
    return out


def as_label(field: schema.Field, value) -> str:
    if field.kind == "multi":
        return "|".join(sorted(value)) if isinstance(value, list) else str(value)
    if field.kind == "bool":
        return str(value).lower()
    return str(value).strip().lower()


def scored(predictions: dict, gold: pd.DataFrame, sources: pd.DataFrame) -> dict:
    """Per field: accuracy and macro-F1, split by gold provenance."""
    report = {}
    for field in schema.FIELDS:
        buckets = {"agreed": ([], []), "adjudicated": ([], [])}
        excluded = 0
        for row_id, prediction in predictions.items():
            if row_id not in gold.index:
                continue
            origin = str(sources.at[row_id, field.name]).strip()
            truth = str(gold.at[row_id, field.name]).strip().lower()
            if origin not in buckets or not truth or truth == "unclear":
                excluded += 1
                continue
            if field.kind == "multi":
                truth = "|".join(sorted(p.strip() for p in truth.split("|") if p.strip()))
            buckets[origin][0].append(truth)
            buckets[origin][1].append(as_label(field, prediction[field.name]))

        entry = {"excluded": excluded}
        for name, (truth, predicted) in buckets.items():
            if len(truth) < 5:
                entry[name] = None
                continue
            entry[name] = {
                "n": len(truth),
                "accuracy": round(sum(t == p for t, p in zip(truth, predicted)) / len(truth), 4),
                "macro_f1": round(f1_score(truth, predicted, average="macro",
                                           zero_division=0), 4)}
        combined = ([*buckets["agreed"][0], *buckets["adjudicated"][0]],
                    [*buckets["agreed"][1], *buckets["adjudicated"][1]])
        entry["all"] = ({"n": len(combined[0]),
                         "accuracy": round(sum(t == p for t, p in zip(*combined))
                                           / len(combined[0]), 4),
                         "macro_f1": round(f1_score(*combined, average="macro",
                                                    zero_division=0), 4)}
                        if len(combined[0]) >= 5 else None)
        entry["optimistic"] = entry["adjudicated"] is None and excluded >= 20
        report[field.name] = entry
    return report


def print_report(name: str, report: dict, extra: dict) -> None:
    print(f"\n{name.upper()}   {extra.get('note', '')}")
    print(f"  {'field':<18} {'agreed':>20} {'contested':>20} {'all':>14}")
    for field in schema.FIELDS:
        entry = report[field.name]
        cells = []
        for key in ("agreed", "adjudicated", "all"):
            value = entry[key]
            cells.append(f"{value['accuracy']:5.1%} F1 {value['macro_f1']:.2f}"
                         if value else f"{'n/a':>14}")
        mark = " *optimistic" if entry["optimistic"] else ""
        note = " (exploratory)" if field.name in schema.EXPLORATORY else ""
        print(f"  {field.name:<18} {cells[0]:>20} {cells[1]:>20} {cells[2]:>14}"
              f"{note}{mark}")
    if any(r["optimistic"] for r in report.values()):
        print("  * scored only on rows two annotators agreed on independently; the")
        print("    excluded rows are the harder ones, so these are optimistic.")


def compare() -> None:
    files = sorted(EVAL_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"Nothing in {EVAL_DIR} yet.")
    results = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in files}
    order = [k for k in ("zero-short", "zero-rubric", "adapter") if k in results]
    order += [k for k in results if k not in order]

    print(f"\nACCURACY ON THE GOLD SET, ALL ROWS   (base: same weights throughout)\n")
    header = "  " + f"{'field':<18}" + "".join(f"{k:>16}" for k in order)
    print(header)
    for field in schema.FIELDS:
        cells = []
        for key in order:
            entry = results[key]["report"][field.name]["all"]
            cells.append(f"{entry['accuracy']:14.1%} " if entry else f"{'n/a':>15} ")
        note = " (exploratory)" if field.name in schema.EXPLORATORY else ""
        print(f"  {field.name:<18}" + "".join(cells) + note)
    print()
    print(f"  {'input tokens':<18}" + "".join(
        f"{results[k].get('system_tokens', '?'):>16}" for k in order))
    print(f"  {'seconds/post':<18}" + "".join(
        f"{results[k].get('seconds_per_post', 0):>16.2f}" for k in order))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("zero-short", "zero-rubric", "adapter"))
    parser.add_argument("--adapter")
    parser.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--ollama-model", default="qwen3:4b")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if args.compare:
        compare()
        return
    if not args.mode:
        parser.error("give --mode or --compare")

    gold, sources = gold_frames()
    rows = pd.read_csv(config.POOL_PILOT)
    rows = rows[rows["id"].isin(gold.index)]
    system = student_system() if args.mode != "zero-rubric" else build_system()

    started = time.time()
    if args.mode == "adapter":
        if not args.adapter:
            parser.error("--mode adapter needs --adapter")
        predictions = predict_adapter(rows, system, args.base, args.adapter)
    else:
        predictions = predict_ollama(rows, system, args.ollama_model, args.workers)
    seconds = time.time() - started

    report = scored(predictions, gold, sources)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    payload = {
        "mode": args.mode,
        "model": args.adapter if args.mode == "adapter" else args.ollama_model,
        "system_tokens": len(tokenizer(system, add_special_tokens=False)["input_ids"]),
        "rows": len(rows),
        "seconds_per_post": round(seconds / max(len(rows), 1), 3),
        "report": report,
    }
    (EVAL_DIR / f"{args.mode}.json").write_text(json.dumps(payload, indent=2),
                                                encoding="utf-8")
    print_report(args.mode, report,
                 {"note": f"{payload['system_tokens']} prompt tokens, "
                          f"{payload['seconds_per_post']:.2f}s per post"})
    print(f"\nwrote {EVAL_DIR / f'{args.mode}.json'}")


if __name__ == "__main__":
    main()
