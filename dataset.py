"""Turn teacher labels into the SFT files the trainer reads.

    python dataset.py --pool pilot        # try it on the 100 pilot rows
    python dataset.py                     # the real build, from the full run

Writes `data/sft/{train,val}.jsonl` plus a manifest. Each line is a chat
exchange in the format the student is trained *and served* with:

    system  -> teacher.student_system()   142 tokens, not the teacher's 2,637
    user    -> teacher.build_user(row)    the same truncation and redaction
    assistant -> the label JSON, keys in schema order

Three things this enforces, each of which quietly ruins a run if missed:

**The student's prompt is not the teacher's.** The teacher needs the whole
rubric in context; the student learns it in its weights. Training on the
teacher's prompt overran the 1,024-token window in the smoke test, masked every
answer token and produced a nan loss.

**Gold rows never enter training.** `gold_blind` is the only unanchored
measurement in the project, and a model trained on its rows would score against
itself.

**Low-confidence teacher labels are dropped.** Every row was annotated three
times; where the votes split, the label is noise and teaching it is worse than
leaving the row out. The threshold is per field and the counts are reported, so
the trade is visible rather than silent.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import config
import guard
import schema
from teacher import build_user, student_system

# With three votes the agreement column is 1.0 (unanimous), 0.667 (a majority)
# or 0.333 (all three differed). Only the last is unusable: a two-thirds
# majority is exactly what voting is for, and 0.67 as a threshold silently
# dropped every majority row through floating point, costing 45% of the data.
MIN_AGREEMENT = 0.5           # keep unanimous and majority, drop no-majority


def load_labels(pool: str) -> pd.DataFrame:
    path = (config.DATA / "teacher_labels_trainset.csv" if pool == "all" else
            config.TEACHER_LABELS.with_name(f"teacher_labels_{pool}.csv"))
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run "
                         f"`python teacher.py --provider gemini --pool {pool}` first.")
    frame = pd.read_csv(path, dtype=str).fillna("")
    frame.columns = [c.lstrip("﻿") for c in frame.columns]
    return frame


def load_rows(pool: str) -> pd.DataFrame:
    paths = {"pilot": [config.POOL_PILOT], "debug": [config.POOL_DEBUG],
             "random": [config.POOL_RANDOM], "enriched": [config.POOL_ENRICHED],
             "all": [config.POOL_RANDOM, config.POOL_ENRICHED]}[pool]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing {missing[0]}. Run `python sample.py` first.")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).set_index("id")


def answer_json(row: pd.Series) -> str:
    """The label as the model must emit it: schema order, emotion as a list."""
    out: dict = {}
    for field in schema.FIELDS:
        value = str(row[field.name]).strip()
        if field.kind == "bool":
            out[field.name] = value.lower() == "true"
        elif field.kind == "multi":
            out[field.name] = [v.strip() for v in value.split("|") if v.strip()]
        else:
            out[field.name] = value
    return json.dumps(out, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="all",
                        choices=("all", "random", "enriched", "pilot", "debug"))
    parser.add_argument("--min-agreement", type=float, default=MIN_AGREEMENT)
    parser.add_argument("--keep-unsure", action="store_true",
                        help="keep rows the teacher split its votes on")
    args = parser.parse_args()

    labels = load_labels(args.pool)
    rows = load_rows(args.pool)
    out_dir = config.DATA / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)

    blind = guard.blind_ids()
    system = student_system()
    kept: dict[str, list[dict]] = {"train": [], "val": []}
    dropped = {"gold": 0, "unparsed": 0, "unsure": 0, "missing_row": 0}
    unsure_by_field: dict[str, int] = dict.fromkeys(schema.NAMES, 0)

    for _, label in labels.iterrows():
        row_id = label["id"]
        if row_id in blind:
            dropped["gold"] += 1
            continue
        if str(label.get("parse_ok", "True")).lower() != "true":
            dropped["unparsed"] += 1
            continue
        if row_id not in rows.index:
            dropped["missing_row"] += 1
            continue

        split_votes = [f.name for f in schema.FIELDS
                       if float(label.get(f"agree_{f.name}", 1.0)) < args.min_agreement]
        if split_votes and not args.keep_unsure:
            for name in split_votes:
                unsure_by_field[name] += 1
            dropped["unsure"] += 1
            continue

        split = "val" if str(label.get("split", "")) == "val" else "train"
        kept[split].append({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": build_user(rows.loc[row_id])},
            {"role": "assistant", "content": answer_json(label)},
        ]})

    for name, examples in kept.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"wrote {path}  ({len(examples):,} examples)")

    print(f"\n  dropped: " + "  ".join(f"{k} {v}" for k, v in dropped.items()))
    if dropped["unsure"]:
        print("  rows dropped for a split vote, by field:")
        for name, count in sorted(unsure_by_field.items(), key=lambda kv: -kv[1]):
            if count:
                print(f"    {name:<18} {count}")

    train = kept["train"]
    if train:
        print("\n  label balance in train:")
        for field in schema.FIELDS:
            counts: dict[str, int] = {}
            for example in train:
                value = json.loads(example["messages"][-1]["content"])[field.name]
                for item in (value if isinstance(value, list) else [value]):
                    counts[str(item)] = counts.get(str(item), 0) + 1
            thin = [v for v, n in counts.items() if n < 20]
            note = f"   <- under 20: {', '.join(thin)}" if thin else ""
            print(f"    {field.name:<18} " +
                  "  ".join(f"{k}={v}" for k, v in sorted(counts.items())) + note)

    manifest = {
        "pool": args.pool,
        "min_agreement": args.min_agreement,
        "train": len(kept["train"]), "val": len(kept["val"]),
        "dropped": dropped,
        "system_prompt": system,
        "fields": list(schema.NAMES),
        "exploratory_fields": sorted(schema.EXPLORATORY),
        "doc_chars": config.DOC_CHARS, "parent_chars": config.PARENT_CHARS,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    print(f"\nwrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
