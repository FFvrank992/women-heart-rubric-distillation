"""The calibration session: resolve two annotators' disagreements together.

    python calibrate.py                       # care_experience + emotion
    python calibrate.py --fields speech_act   # or any subset

Writes `data/gold/calibration.csv`, one row per post where the two annotators
disagree on one of the chosen fields, showing both answers side by side with a
column for the resolved label and a column for the rule that would have
prevented the disagreement.

Why this rather than labelling fresh rows: the 100 rows already labelled
contain the diagnosis. Resolving them together does three jobs at once — it
calibrates the two readers, it produces an adjudicated gold set, and the
`rule_needed` column is the next version of the rubric, written from real
disagreements instead of guesses.

Rows are grouped by the pair of labels that clashed, so one principle settles a
whole block at a time. Deciding "a feeling about care is not an episode" once
clears every `positive` / `not_mentioned` row below it.

**The reliability figure is the kappa from before this session, not after.**
Adjudicated labels are no longer independent of the discussion, so they cannot
be used to measure agreement. They are still independent of the *model*, which
is what makes them usable as gold for evaluation. Report the two separately:
pre-adjudication kappa as reliability, adjudicated labels as gold.
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import cohen_kappa_score

import agreement as ag
import config
import schema

DEFAULT_FIELDS = ("care_experience", "emotion")

HEADER = """\
# Calibration session — {rows} posts, {cells} decisions

Work through this together, both of you looking at the same screen, with
`RUBRIC.md` open. Budget about {minutes} minutes.

## Why

On the blind 100 you two agreed far less than the definitions assumed:

{summary}

The disagreements are *systematic*, not random — each of you applied a
consistent reading that the other did not share. That is a sign the definition
is underspecified, not that either of you was careless. This session fixes the
definition.

## How to work through it

The rows are **grouped by which pair of labels clashed**, so similar cases sit
together. For each block:

1. Read the first post and decide, out loud, which label the rubric supports.
2. Write the agreed label in `resolved_<field>`.
3. **Write the rule that settled it in `rule_needed`** — one sentence, phrased
   so that someone with only the rubric would reach the same answer. This is
   the most valuable column in the file.
4. Apply the same rule down the rest of the block, and only slow down where a
   row genuinely differs.

If you cannot agree on a row after a minute, put `UNCLEAR` in `resolved_` and
say why in `rule_needed`. A row that two people cannot settle with the rubric in
front of them is a row the model should not be scored on either.

## Two rules were already tightened before this session

Both from the same idea — you must be able to **point at the words**:

- `care_experience` is `not_mentioned` unless you can point at what a clinician
  or the system *did or failed to do*. A feeling about care is not an episode.
  And `access_barrier` needs the care to have **not been obtained** — delay or
  expense on the way to care that still happened is something else.
- `emotion` is `none` unless you can point at a word that *states* a feeling.
  "I had a heart attack" is `none`; "I was terrified" is `fear`.

Check whether those two rules already settle a block before debating it. Where
they do, say so in `rule_needed` — confirming a new rule works is as useful as
writing one.
"""


def disagreeing(a: pd.DataFrame, b: pd.DataFrame, field: schema.Field) -> list[str]:
    """Row ids where the two annotators differ on this field, UNCLEAR included."""
    out = []
    for row_id in a.index:
        left = str(a.at[row_id, field.name]).strip().lower()
        right = str(b.at[row_id, field.name]).strip().lower()
        if not left or not right:
            continue
        if field.kind == "multi":
            if {p.strip() for p in left.split("|")} != {p.strip() for p in right.split("|")}:
                out.append(row_id)
        elif left != right:
            out.append(row_id)
    return out


def apply_resolutions(fields: list[schema.Field]) -> None:
    """Turn the finished calibration sheet into `data/gold/gold_blind.csv`.

    Each cell comes from one of two places, and the source is recorded per field
    so the report can say which: `agreed` where the two annotators matched
    independently, `adjudicated` where this session settled it. Adjudicated
    cells are not independent of the discussion and must never be used to
    compute agreement - the pre-session kappa is the reliability figure. They
    are still independent of the model, which is what makes them usable as gold.
    """
    path = config.GOLD / "calibration.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run `python calibrate.py` first.")
    resolved = pd.read_csv(path, dtype=str).fillna("").set_index("id")
    a, b = ag.align(ag.read_sheet(config.PASS1), ag.read_sheet(config.INTER))

    unfinished = []
    for row_id, row in resolved.iterrows():
        for name in str(row["fields_in_dispute"]).split("|"):
            if name and not str(row.get(f"resolved_{name}", "")).strip():
                unfinished.append(f"{row_id} / {name}")
    if unfinished:
        raise SystemExit(
            f"{len(unfinished)} decisions are still blank, e.g. "
            f"{', '.join(unfinished[:3])}.\n"
            f"  Finish the session before applying it, or the gold set silently\n"
            f"  keeps one annotator's answer where you meant to agree one.")

    out, sources = [], []
    for row_id in a.index:
        record = {"id": row_id}
        source = {"id": row_id}
        for field in schema.FIELDS:
            left = str(a.at[row_id, field.name]).strip()
            right = str(b.at[row_id, field.name]).strip()
            same = (({p.strip().lower() for p in left.split("|")}
                     == {p.strip().lower() for p in right.split("|")})
                    if field.kind == "multi" else left.lower() == right.lower())
            column = f"resolved_{field.name}"
            # Only the fields this session covered have a resolved column; the
            # rest keep an agreement or stay unresolved.
            settled = (column in resolved.columns and row_id in resolved.index
                       and str(resolved.at[row_id, column]).strip())
            if same and left.upper() != "UNCLEAR":
                record[field.name], source[field.name] = left, "agreed"
            elif settled:
                record[field.name] = str(resolved.at[row_id, column]).strip()
                source[field.name] = "adjudicated"
            else:
                # Not calibrated this round: an exploratory or already-usable
                # field where the two still differ. Left out of gold rather
                # than resolved by coin flip.
                record[field.name], source[field.name] = "UNCLEAR", "unresolved"
        out.append(record)
        sources.append(source)

    gold = pd.DataFrame(out)
    gold.to_csv(config.GOLD / "gold_blind.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sources).to_csv(config.GOLD / "gold_blind_sources.csv",
                                 index=False, encoding="utf-8-sig")
    print(f"wrote {config.GOLD / 'gold_blind.csv'}  ({len(gold)} rows)")
    print(f"wrote {config.GOLD / 'gold_blind_sources.csv'}\n")
    frame = pd.DataFrame(sources)
    for field in schema.FIELDS:
        counts = frame[field.name].value_counts()
        note = " (exploratory)" if field.name in schema.EXPLORATORY else ""
        print(f"  {field.name:<18} " + "  ".join(
            f"{k} {v}" for k, v in counts.items()) + note)
    print("\n  The reliability figure stays the pre-session kappa from")
    print("  `agreement.py --inter`. Quote these labels as gold, and say in the")
    print("  report that disagreements were resolved by discussion.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", nargs="*", default=list(DEFAULT_FIELDS),
                        help=f"fields to calibrate (default: {' '.join(DEFAULT_FIELDS)})")
    parser.add_argument("--apply", action="store_true",
                        help="turn the finished sheet into data/gold/gold_blind.csv")
    args = parser.parse_args()
    config.ensure_dirs()

    if args.apply:
        apply_resolutions([schema.BY_NAME[n] for n in args.fields
                           if n in schema.BY_NAME])
        return

    unknown = [f for f in args.fields if f not in schema.BY_NAME]
    if unknown:
        raise SystemExit(f"Not active fields: {', '.join(unknown)}. "
                         f"Available: {', '.join(schema.NAMES)}")

    a, b = ag.align(ag.read_sheet(config.PASS1), ag.read_sheet(config.INTER))
    fields = [schema.BY_NAME[name] for name in args.fields]

    wanted: dict[str, list[str]] = {}
    for field in fields:
        for row_id in disagreeing(a, b, field):
            wanted.setdefault(row_id, []).append(field.name)
    if not wanted:
        print("The two sheets already agree on every chosen field.")
        return

    pool = pd.read_csv(config.POOL_PILOT).set_index("id")
    sheet_text = pd.read_csv(config.PASS1, dtype=str).fillna("").set_index("id")

    # Every single-field clash in the set, so a row can be filed under its most
    # common one. Keying blocks on the *combination* of clashes fragments them:
    # 47 blocks with a maximum of 7 rows, which defeats the point of grouping.
    clash_counts: dict[str, int] = {}
    row_clashes: dict[str, list[str]] = {}
    for row_id, clashing in wanted.items():
        for name in clashing:
            left = str(a.at[row_id, name]).strip().lower()
            right = str(b.at[row_id, name]).strip().lower()
            key = f"{name}: {left} / {right}"
            clash_counts[key] = clash_counts.get(key, 0) + 1
            row_clashes.setdefault(row_id, []).append(key)

    rows = []
    for row_id, clashing in wanted.items():
        row = {"id": row_id,
               "platform": pool.at[row_id, "platform"] if row_id in pool.index else "",
               "fields_in_dispute": "|".join(clashing),
               "parent_context": sheet_text.at[row_id, "parent_context"],
               "post": sheet_text.at[row_id, "post"]}
        for field in fields:
            row[f"A_{field.name}"] = str(a.at[row_id, field.name]).strip()
            row[f"B_{field.name}"] = str(b.at[row_id, field.name]).strip()
            row[f"resolved_{field.name}"] = ""
        # File the row under its most frequent clash, so blocks come out large.
        row["_block"] = max(row_clashes[row_id], key=lambda k: clash_counts[k])
        row["rule_needed"] = ""
        rows.append(row)

    sheet = pd.DataFrame(rows)
    sizes = sheet["_block"].value_counts()
    sheet["_size"] = sheet["_block"].map(sizes)
    sheet = (sheet.sort_values(["_size", "_block", "id"], ascending=[False, True, True])
             .drop(columns=["_size"]))

    cells = sum(len(v) for v in wanted.values())
    summary_lines = []
    for field in fields:
        if field.kind == "multi":
            kappa = ag.emotion_kappa(a[field.name], b[field.name])[0]
        else:
            left, right, _ = ag.paired(a[field.name], b[field.name])
            kappa = cohen_kappa_score(left, right)
        n = len(disagreeing(a, b, field))
        summary_lines.append(f"- `{field.name}` — kappa {kappa:.3f}, "
                             f"{n} of {len(a)} rows disagree")
    summary = "\n".join(summary_lines)

    blocks = sheet["_block"].value_counts()
    sheet = sheet.drop(columns=["_block"])
    sheet.to_csv(config.GOLD / "calibration.csv", index=False, encoding="utf-8-sig")
    (config.GOLD / "calibration.md").write_text(
        HEADER.format(rows=len(sheet), cells=cells, minutes=round(cells * 0.75),
                      summary=summary), encoding="utf-8")

    print(f"wrote {config.GOLD / 'calibration.csv'}  ({len(sheet)} rows, {cells} decisions)")
    print(f"wrote {config.GOLD / 'calibration.md'}")
    print(f"\nestimated {cells * 0.75:.0f} minutes at 45s per decision\n")
    print("the blocks to settle, biggest first:")
    for block, n in blocks.head(12).items():
        print(f"  {n:3d}  {block}")
    if len(blocks) > 12:
        print(f"  ... and {len(blocks) - 12} smaller blocks")
    print("\nAfterwards:  python calibrate.py --apply")


if __name__ == "__main__":
    main()
