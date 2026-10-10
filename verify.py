"""The prefilled half of the gold set, routed by where two models disagree.

    python verify.py --build-pool          # choose the rows
    # then label them with two different models:
    python teacher.py --provider gemini    --pool verify --model gemini-3.1-flash-lite \
        --out data/verify_model_a.csv
    python teacher.py --provider anthropic --pool verify --model claude-haiku-4-5-20251001 \
        --out data/verify_model_b.csv
    python verify.py data/verify_model_a.csv data/verify_model_b.csv

Labelling 400 rows from scratch alone is about ten hours. This cuts it to
roughly two and a half by prefilling a model's answer and having you correct it,
and by spending your attention where it is worth spending: rows where two
independent models disagree are the hard ones, and rows where they agree mostly
need a glance.

The cost is anchoring. A sheet you corrected is partly the model's work, so
these rows are **`gold_verified`: adjudicated, not independent**. They are
reported separately from `gold_blind` and the two are never pooled for a
headline number.

Two rules make that cost manageable:

**The prefill comes from a model that is not the teacher.** Anchoring the gold
toward the teacher would inflate teacher agreement, and the student is trained
to imitate the teacher, so it would flatter the student too — exactly the
comparison the project rests on. A different model family anchors toward
something neither the teacher nor the student shares.

**The blind rows are excluded entirely.** They are already labelled by hand
twice, and they are the only unanchored measurement there is.

Because `gold_blind` is labelled with no prefill and `gold_verified` with one,
the difference between your agreement with the models on each is a measurement
of the anchoring effect. Report it: a stated bias is a strength.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
import guard
import schema
from teacher import prepare

# Any mention of the cardiovascular system. Used only to keep hand labelling off
# posts with no cardiac content; it is never a label.
CARDIAC_TERMS = (
    r"(?:heart|cardiac|cardio|afib|a-fib|arrhythm|angina|stent|palpitat|chest pain|"
    r"blood pressure|cholesterol|statin|\bscad\b|takotsubo|tachycard|bradycard|"
    r"ablation|pacemaker|\becg\b|\bekg\b|echocardio|artery|aortic|valve|bypass|"
    r"\bclot\b|stroke|coronary|myocard|\bchf\b|\bef\b|\bicd\b|defibrill)")

SHEET_HEADER = """\
# Verified gold — {n} posts, prefilled

Two models labelled each of these rows independently. Your job is to **correct**
the prefill, not to re-derive it from nothing.

## How to work through it

Sort by `needs_attention` (it is already sorted that way):

- **`needs_attention = yes`** ({attention} rows) — the two models disagree on at
  least one field. These are the hard cases. Read the post properly, decide from
  the rubric, and fix every field. Budget about a minute each.
- **`needs_attention = no`** ({easy} rows) — the models agree. Skim the post,
  confirm the labels look right, and move on. Budget about fifteen seconds. If
  one looks wrong, fix it: agreement between two models is not correctness.

The columns `a_<field>` and `b_<field>` hold the two models' answers and are
**read-only context**. Fill the plain `<field>` columns: they start as model A's
answer, so you only type where you disagree.

## What not to do

Do not rubber-stamp. The value of this sheet is your corrections; a sheet you
did not really read makes the metrics look better than the model deserves and
the error analysis will not make sense.

Write `UNCLEAR` where you genuinely cannot decide. Put anything the rubric did
not cover in `notes` — those notes are what the next rubric version is written
from.

These rows are reported as `gold_verified` and described as adjudicated, never
as independent annotation.
"""


def build_pool() -> None:
    """Gold rows that are not in the blind set."""
    for path in (config.POOL_RANDOM, config.POOL_ENRICHED):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run `python sample.py` first.")
    blind = guard.blind_ids()
    if not blind:
        raise SystemExit(f"Missing {config.POOL_PILOT}. Run `python pilot.py` first: "
                         f"the blind rows have to be chosen before these.")

    random_pool = pd.read_csv(config.POOL_RANDOM)
    enriched_pool = pd.read_csv(config.POOL_ENRICHED)

    # The `womens` retrieval heuristic pulls in women's-health posts with no
    # cardiac content at all - hysterectomy, pregnancy or menopause threads on
    # Mayo Clinic Connect, which is a general health forum. 89 of its 666 rows
    # (13.4%) have no cardiac term anywhere. They are legitimate corpus rows and
    # the model will meet them at inference time, but spending scarce hand
    # labelling on them is a poor trade, so they are filtered out here rather
    # than by re-running sample.py, which would invalidate the blind draw.
    off_topic = ~enriched_pool["text"].astype(str).str.lower().str.contains(
        CARDIAC_TERMS, regex=True, na=False)
    if off_topic.any():
        print(f"filtering {off_topic.sum()} enriched rows with no cardiac term")
        enriched_pool = enriched_pool[~off_topic]
    # The pools reserved a larger gold set than the one-annotator budget can
    # reach. Take what is needed and leave the surplus unused rather than
    # re-running sample.py, which would invalidate the blind draw.
    wanted = {"random": config.N_GOLD_RANDOM - config.N_PILOT_RANDOM,
              "enriched": config.N_GOLD_ENRICHED - config.N_PILOT_ENRICHED}
    picked = []
    for name, pool, split in (("random", random_pool, "gold_random"),
                              ("enriched", enriched_pool, "gold_enriched")):
        available = pool[(pool["split"] == split) & (~pool["id"].isin(blind))]
        take = min(wanted[name], len(available))
        if take < wanted[name]:
            print(f"note: only {take} {name} gold rows left outside the blind set")
        picked.append(available.sample(take, random_state=config.SEED + 3))

    verify = pd.concat(picked, ignore_index=True)
    verify = verify.sample(frac=1, random_state=config.SEED + 3).reset_index(drop=True)
    verify["split"] = "gold_verified"
    verify.to_csv(config.POOL_VERIFY, index=False, encoding="utf-8")

    print(f"wrote {config.POOL_VERIFY}  ({len(verify)} rows)")
    print(verify["pool"].value_counts().to_string())
    print(f"\nblind rows excluded: {len(blind)}")
    print(f"gold total will be {len(blind)} blind + {len(verify)} verified "
          f"= {len(blind) + len(verify)}")
    print("\nNext, label them with two different models (the prefill must not come\n"
          "from the teacher alone), then run verify.py with both files:\n"
          f"    python teacher.py --provider gemini --pool verify "
          f"--out data/verify_model_a.csv\n"
          f"    python teacher.py --provider anthropic --pool verify "
          f"--model claude-haiku-4-5-20251001 --out data/verify_model_b.csv\n"
          f"    python verify.py data/verify_model_a.csv data/verify_model_b.csv")


def read_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Label the verify pool with a model first.")
    frame = pd.read_csv(path, dtype=str).fillna("")
    return frame.set_index("id")


def build_sheet(a_path: Path, b_path: Path) -> None:
    if not config.POOL_VERIFY.exists():
        raise SystemExit(f"Missing {config.POOL_VERIFY}. Run "
                         f"`python verify.py --build-pool` first.")
    rows = pd.read_csv(config.POOL_VERIFY, dtype=str)
    a, b = read_labels(a_path), read_labels(b_path)

    blind = guard.blind_ids()
    leaked = rows["id"].isin(blind)
    if leaked.any():                      # belt and braces: build_pool excludes them
        print(f"dropping {leaked.sum()} blind rows that should not be here")
        rows = rows[~leaked]

    shared = [i for i in rows["id"] if i in a.index and i in b.index]
    missing = len(rows) - len(shared)
    if missing:
        print(f"note: {missing} rows are missing from one of the model files, skipping")
    rows = rows[rows["id"].isin(shared)].reset_index(drop=True)

    models = {"a": str(a["model"].iloc[0]) if "model" in a else a_path.stem,
              "b": str(b["model"].iloc[0]) if "model" in b else b_path.stem}
    if models["a"] == models["b"]:
        print(f"\nWARNING: both files are from {models['a']}. Two tiers of the same\n"
              f"  family are only weakly independent, and the same model twice is not\n"
              f"  independent at all: the disagreement routing will find almost\n"
              f"  nothing and the prefill carries that model's bias wholesale.\n")

    prepared = [prepare(row) for _, row in rows.iterrows()]
    sheet = pd.DataFrame({
        "id": rows["id"],
        "platform": rows["platform"],
        "type": rows["type"],
        "pool": rows["pool"],
        "parent_context": [parent for _, parent in prepared],
        "post": [post for post, _ in prepared],
    })

    disagreements = pd.Series(0, index=sheet.index)
    for field in schema.FIELDS:
        left = [str(a.at[i, field.name]).strip().lower() for i in rows["id"]]
        right = [str(b.at[i, field.name]).strip().lower() for i in rows["id"]]
        differs = [x != y for x, y in zip(left, right)]
        disagreements += pd.Series(differs, index=sheet.index).astype(int)
        sheet[field.name] = left              # prefill: model A, the one you correct
        sheet[f"a_{field.name}"] = left
        sheet[f"b_{field.name}"] = right

    sheet["n_model_disagreements"] = disagreements
    sheet["needs_attention"] = ["yes" if n else "no" for n in disagreements]
    sheet["notes"] = ""
    sheet = sheet.sort_values(["n_model_disagreements", "id"], ascending=[False, True])

    attention = int((sheet["needs_attention"] == "yes").sum())
    easy = len(sheet) - attention
    sheet.to_csv(config.VERIFY_SHEET, index=False, encoding="utf-8-sig")
    header = config.VERIFY_SHEET.with_suffix(".md")
    header.write_text(
        SHEET_HEADER.format(n=len(sheet), attention=attention, easy=easy),
        encoding="utf-8")

    print(f"\nwrote {config.VERIFY_SHEET}  ({len(sheet)} rows)")
    print(f"wrote {header}")
    print(f"\n  prefill from   {models['a']}  (the column you correct)")
    print(f"  second opinion {models['b']}")
    print(f"  needs_attention  yes {attention}  no {easy}")
    print("\n  per-field disagreement between the two models:")
    for field in schema.FIELDS:
        differs = (sheet[f"a_{field.name}"] != sheet[f"b_{field.name}"]).mean()
        print(f"    {field.name:<18} {differs:5.1%}")
    estimate = attention * 60 + easy * 15
    print(f"\n  rough budget: {attention} x 60s + {easy} x 15s "
          f"= {estimate / 3600:.1f} hours")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-pool", action="store_true",
                        help="choose the rows and write data/pool_verify.csv")
    parser.add_argument("models", nargs="*", type=Path,
                        help="the two model label files, A then B")
    args = parser.parse_args()
    config.ensure_dirs()

    if args.build_pool:
        build_pool()
        return
    if len(args.models) != 2:
        parser.error("give exactly two model label files, or --build-pool")
    build_sheet(*args.models)


if __name__ == "__main__":
    main()
