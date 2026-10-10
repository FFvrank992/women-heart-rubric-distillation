"""The blind gold set and the rubric-debugging slice.

    python pilot.py                 # 100 blind rows + the first pass sheet
    python pilot.py --pass2         # the second pass, reshuffled, 3+ days later
    python pilot.py --debug-slice   # 100 train rows for reading the teacher

This script sets up the sheets that reliability is measured from. Three
measurements are taken over the same 100 rows, and they answer different
questions:

**Blind pass 1 and pass 2.** The same 100 rows labelled twice by one annotator,
at least three days apart, in a different order, without looking at the first
pass. The kappa between them is intra-annotator agreement (test-retest
reliability): it shows the definitions are stable enough for one careful reader
to reproduce. It does **not** show that a different person would agree, and it
is an upper bound, because one annotator shares their own reading of every edge
case. Quote it only alongside the inter-annotator figure, never instead of it.

**A second annotator's blind pass** (`--second-annotator`, scored by
`agreement.py --inter`) is the real measurement: two people, the same written
instructions, no conversation between them. It is the figure the field gates
run on, and it came out far below the test-retest figure — which is the
expected result and the reason the calibration session exists
(`calibrate.py`, `docs/calibration.md`).

**The debug slice.** 100 rows taken from the `train` split, which no human ever
labels. The teacher's three votes on these rows are free to look at: where the
model cannot reproduce its own label across three samples, the definition is
ambiguous. Read that *before* spending hours on the blind passes, so the hand
labelling runs against a rubric the model has already stress-tested. It is the
cheapest of the three and it catches the crudest problems, but it is not a
substitute for the second reader: it found no field over the 25% threshold,
and the second reader then returned kappa 0.136 on `health_claim`.

The blind rows come from the gold splits, so the hand labelling is not thrown
away once the gate passes: they become `gold_blind`.

The mix is 60 random and 40 enriched rows, on purpose. 100 purely random rows
would hold roughly three `dismissed` cases and almost no claims, so the
reliability figure for the fields that matter most would be undefined. The
consequence is that **label frequencies here are not prevalence estimates** —
prevalence comes from the weighted random pool.
"""
from __future__ import annotations

import argparse

import pandas as pd

import config
import guard
import schema
from teacher import prepare

PASS1_HEADER = """\
# Blind pass 1 — 100 posts

Label all 100 rows yourself, from scratch. **Do not open
`data/teacher_labels.csv`, and do not ask any chatbot what these posts are.**
These rows are the only measurement in the project that no model has touched,
and that is the whole of their value. The tools will refuse to show you model
output for them until both passes are done.

Read `RUBRIC.md` end to end first, including the boundary cases: that is where
the hard decisions are, and most of them are there because a real post in this
corpus is genuinely ambiguous.

When this sheet is finished, wait **at least three days**, then run
`python pilot.py --pass2` and label the same rows again in the new order,
without reopening this file. The agreement between the two passes is the
reliability number that goes in the report.
"""

PASS2_HEADER = """\
# Blind pass 2 — the same 100 posts, reshuffled

Same rows, new order. **Do not reopen `pilot_pass1.csv`** — if you copy your
earlier answers the agreement figure becomes meaningless, and a meaningless
reliability figure is worse than none.

Label from the rubric as if for the first time. Disagreeing with your earlier
self is the expected outcome on hard rows; that is the thing being measured.
"""

ANNOTATOR_B_HEADER = """\
# Blind labelling — 100 posts about women's heart health

Thank you for doing this. It takes about an hour and a half, and what it buys is
the one number that says whether this project's labels are trustworthy.

## What this is for

The 100 posts in `pilot_annotator_b.csv` have already been labelled by someone
else. Comparing your answers with theirs measures **inter-annotator
agreement**: whether two people reading the same instructions land in the same
place. If we agree, the definitions are sound. Where we disagree, the
definition is unclear and gets rewritten.

So the measurement only works if you label **independently**. Please:

- **Do not look at anyone else's labels** before you finish.
- **Do not ask what a borderline case "should" be.** If you ask and are told,
  the number stops measuring the instructions and starts measuring the
  conversation. Use your own reading of the rubric and move on.
- **Write down every question you wanted to ask** in the `notes` column. Those
  are the most valuable thing you will produce: each one is a place where the
  instructions were not clear enough.
- Disagreeing with the other annotator is **not a mistake**. On the hard rows it
  is the expected outcome, and it is the thing being measured.

## What to do

1. Read `RUBRIC.md` once, end to end, including the boundary cases. It is the
   only thing you need, and about twenty minutes of reading. The boundary cases
   exist because real posts in this corpus are genuinely ambiguous.
2. Open `pilot_annotator_b.csv` in Excel and fill one row at a time.

One column per field:

{table}
- `emotion` takes one or two values, strongest first, separated by `|`
  (e.g. `fear|frustration`). `none` cannot be combined with anything.
- `womens_specific` is `true` or `false`.
- Write **`UNCLEAR`** where you genuinely cannot decide. Those cells are left
  out of the metrics. An honest `UNCLEAR` is far better than a guess.
- `notes` is free text — see above, please use it.

## What you are reading

`post` is a de-identified public comment or forum post, truncated to 1,200
characters. `parent_context` is filled only for replies and tells you what the
reply is responding to: **label the reply, not the parent.**

Some posts have nothing to do with the heart — the forums cover general health.
Those still get labelled; see rule 8 at the top of the rubric.

The posts are in a different order from the other annotator's sheet. That is
deliberate and nothing to worry about: the comparison is done by post id.
"""

INSTRUCTIONS = """\
# Annotating

Four figures are available over the same 100 rows. They are not
interchangeable, and each one is reported as what it is.

| Figure | What it shows | How |
|---|---|---|
| Teacher self-disagreement | which field definitions are ambiguous | free, no human: `agreement.py --teacher-self --pool debug` |
| **Inter-annotator kappa** | **whether the definitions are shared** | a second annotator's blind pass: `agreement.py --inter <their sheet>` |
| Intra-annotator kappa | whether the definitions are stable for one reader | both blind passes, 3+ days apart |
| Human-model agreement | that the labels track more than one reader | `agreement.py --vs-teacher`, after both passes |

**The inter-annotator figure is the one that gates the fields.** The other three
are weaker and were built first because they cost nothing; none of them caught
what the second reader caught. Getting one person for 90 minutes is worth more
than any amount of work on the substitutes — ask early.

## Order of work

1. `python pilot.py --debug-slice` then label those rows with the teacher and
   read `agreement.py --teacher-self --pool debug`. Fix the rubric where the
   model cannot agree with itself. Bump the rubric version.
2. `python pilot.py` and fill `pilot_pass1.csv` from scratch. ~1.5 hours.
3. `python pilot.py --second-annotator` and give that sheet, plus `RUBRIC.md` and
   `docs/annotator-b-brief.md`, to a second person. Do not discuss rows with
   them while they work.
4. `python agreement.py --inter data/gold/pilot_annotator_b.csv`. This is the
   gate. Expect it to be lower than you hope.
5. For any field it fails, `python calibrate.py` and work the disagreements
   through together, writing the rule that settles each one. Then raise the
   rubric version and rebuild the gold set from the resolved labels.
6. Optionally, three days or more after step 2, `python pilot.py --pass2` and
   `python agreement.py --intra --vs-teacher` for the test-retest figure.
7. Only then `python verify.py` for the prefilled expansion set.

## Filling a sheet

One column per field:

{table}
- `emotion` takes one or two values, strongest first, separated by `|`
  (e.g. `fear|frustration`). `none` cannot be combined with anything.
- `womens_specific` is `true` or `false`.
- Write **`UNCLEAR`** where you genuinely cannot decide. Those cells are
  excluded from the metrics and reported as a coverage figure. An honest
  `UNCLEAR` is far more useful than a guess, and on pass 2 it is a real signal.
- `notes` is free text. Use it whenever the rubric did not cover the case:
  those notes are what the next version of the rubric is written from.

`post` is exactly the text the model sees — truncated to 1,200 characters, with
salutation names replaced by `[NAME]`. `parent_context` is filled only for
replies and is there to tell you what the reply refers to: **annotate the
reply, not the parent.**
"""


def build_sheet_table() -> str:
    rows = ["| Column | Allowed values |", "|---|---|"]
    for field in schema.FIELDS:
        allowed = ("`true` / `false`" if field.kind == "bool"
                   else " / ".join(f"`{v}`" for v in field.values))
        rows.append(f"| `{field.name}` | {allowed} |")
    return "\n".join(rows)


def load_pools() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (config.POOL_RANDOM, config.POOL_ENRICHED):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run `python sample.py` first.")
    return pd.read_csv(config.POOL_RANDOM), pd.read_csv(config.POOL_ENRICHED)


def pick_pilot() -> pd.DataFrame:
    random_pool, enriched_pool = load_pools()
    from_random = (random_pool[random_pool["split"] == "gold_random"]
                   .sample(config.N_PILOT_RANDOM, random_state=config.SEED))
    from_enriched = (enriched_pool[enriched_pool["split"] == "gold_enriched"]
                     .sample(config.N_PILOT_ENRICHED, random_state=config.SEED))
    pilot = pd.concat([from_random, from_enriched], ignore_index=True)
    pilot = pilot.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    pilot["split"] = "gold_blind"
    return pilot


def pick_debug() -> pd.DataFrame:
    """Train rows, never hand-labelled, so model output on them is free to read."""
    random_pool, enriched_pool = load_pools()
    both = pd.concat([random_pool, enriched_pool], ignore_index=True)
    train = both[both["split"] == "train"]
    # A different seed from the blind draw, and blind ids excluded anyway.
    debug = train[~train["id"].isin(guard.blind_ids())].sample(
        config.N_DEBUG, random_state=config.SEED + 1).reset_index(drop=True)
    debug["split"] = "debug"
    return debug


def build_sheet(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per post, showing exactly what the model is shown."""
    prepared = [prepare(row) for _, row in rows.iterrows()]
    sheet = pd.DataFrame({
        "id": rows["id"].values,
        "platform": rows["platform"].values,
        "type": rows["type"].values,
        "pool": rows["pool"].values,
        "parent_context": [parent for _, parent in prepared],
        "post": [post for post, _ in prepared],
    })
    for field in schema.FIELDS:
        sheet[field.name] = ""
    sheet["notes"] = ""
    return sheet


def write_sheet(sheet: pd.DataFrame, path, header: str) -> None:
    # utf-8-sig so Excel on Windows opens the quotes and emoji correctly.
    sheet.to_csv(path, index=False, encoding="utf-8-sig")
    path.with_suffix(".md").write_text(header, encoding="utf-8")
    print(f"wrote {path}  ({len(sheet)} rows)")
    print(f"wrote {path.with_suffix('.md')}")


def refresh_text(path) -> None:
    """Rebuild a sheet's `post` and `parent_context` while keeping its labels.

    Needed whenever a preprocessing bug is fixed after labelling has started:
    regenerating the sheet from scratch would throw the work away, and leaving
    the corrupted text in place means labelling the wrong thing. Rows whose text
    actually moved are listed, so only those have to be looked at again.
    """
    if not path.exists():
        raise SystemExit(f"Missing {path}.")
    if not config.POOL_PILOT.exists():
        raise SystemExit(f"Missing {config.POOL_PILOT}. Run `python pilot.py` first.")

    sheet = pd.read_csv(path, dtype=str).fillna("")
    sheet.columns = [c.lstrip("﻿") for c in sheet.columns]
    pool = pd.read_csv(config.POOL_PILOT).set_index("id")

    # Row ids are content hashes (platform + sha256(url + content)[:16]), so
    # repairing the stored text re-keys the row and the id in a finished sheet
    # stops resolving. Silently skipping those rows would leave hand-labelled
    # gold pointing at stale text while every count still looked right, so this
    # stops instead and sends you to --remap-ids.
    missing = [(i + 1, row["id"]) for i, row in sheet.iterrows()
               if row["id"] not in pool.index]
    if missing:
        listed = "\n".join(f"    row {n:3d}  {row_id}" for n, row_id in missing[:10])
        more = f"\n    ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise SystemExit(
            f"{len(missing)} of {len(sheet)} ids in {path.name} are not in "
            f"{config.POOL_PILOT.name}:\n{listed}{more}\n\n"
            f"  That happens when the delivered corpus is repaired: the id is a\n"
            f"  hash of the content, so fixing the content changes the id. The\n"
            f"  labels are still good, the keys are not.\n\n"
            f"      python pilot.py --remap-ids\n\n"
            f"  finds where those rows went by matching on their text, and\n"
            f"  rewrites the pool and every sheet together. Run it first.")

    label_columns = [f.name for f in schema.FIELDS]
    changed = []
    for i, row in sheet.iterrows():
        post, parent = prepare(pool.loc[row["id"]])
        if post != row["post"] or parent != row["parent_context"]:
            labelled = any(str(row[c]).strip() for c in label_columns)
            changed.append((i + 1, row["id"], labelled))
            sheet.at[i, "post"] = post
            sheet.at[i, "parent_context"] = parent

    if not changed:
        print(f"{path.name}: text already current, nothing to do")
        return

    backup = path.with_suffix(".csv.bak")
    path.replace(backup)
    sheet.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"previous sheet kept at {backup.name}")
    print(f"refreshed {path.name}: {len(changed)} rows had their text corrected\n")

    done = [(n, i) for n, i, labelled in changed if labelled]
    todo = [(n, i) for n, i, labelled in changed if not labelled]
    if done:
        print(f"  ALREADY LABELLED - check these {len(done)} again, the text moved:")
        for n, row_id in done:
            print(f"    row {n:3d}  {row_id}")
    if todo:
        print(f"  not yet labelled, nothing to redo ({len(todo)}): "
              f"{', '.join(str(n) for n, _ in todo)}")


def corpus_text_index() -> dict[tuple[str, str], list[str]]:
    """(platform, prepared post) -> ids, for finding a row whose id has changed.

    The prepared post is exactly what the gold sheets hold, which makes it the
    natural key: it survives a re-keying, and it is already the thing both the
    annotators and the model were shown.
    """
    config.require_corpus()
    corpus = pd.read_csv(config.CORPUS)
    corpus.columns = [c.lstrip("﻿") for c in corpus.columns]
    full = pd.read_csv(config.CORPUS_FULL, usecols=["id", "is_reply", "parent_text"])
    full.columns = [c.lstrip("﻿") for c in full.columns]
    corpus = corpus.merge(full, on="id", how="left")

    index: dict[tuple[str, str], list[str]] = {}
    for _, row in corpus.iterrows():
        post, _ = prepare(row)
        index.setdefault((str(row["platform"]), post), []).append(str(row["id"]))
    return index


def remap_ids() -> None:
    """Re-key the pool and every sheet after the corpus has been repaired.

    Matches on (platform, prepared post) and requires a unique hit. Repairing
    the text makes some short comments collide — the delivered YouTube rows
    carry their video's URL rather than their own, so identical short comments
    from different authors become indistinguishable — and a collision is
    reported rather than guessed at.
    """
    if not config.POOL_PILOT.exists():
        raise SystemExit(f"Missing {config.POOL_PILOT}. Run `python pilot.py` first.")
    config.require_corpus()
    pool = pd.read_csv(config.POOL_PILOT)
    corpus_ids = set(pd.read_csv(config.CORPUS, usecols=["id"])["id"].astype(str))

    stale = pool[~pool["id"].astype(str).isin(corpus_ids)]
    if stale.empty:
        print(f"every id in {config.POOL_PILOT.name} still resolves - nothing to remap")
        return

    print(f"{len(stale)} of {len(pool)} pool ids no longer exist in the corpus")
    print("building a text index over the corpus ...", flush=True)
    index = corpus_text_index()

    remap: dict[str, str] = {}
    ambiguous, lost = [], []
    for _, row in stale.iterrows():
        post, _ = prepare(row)
        matches = index.get((str(row["platform"]), post), [])
        if len(matches) == 1:
            remap[str(row["id"])] = matches[0]
        elif matches:
            ambiguous.append((str(row["id"]), matches))
        else:
            lost.append(str(row["id"]))

    print(f"  matched uniquely : {len(remap)}")
    print(f"  ambiguous        : {len(ambiguous)}")
    print(f"  not found        : {len(lost)}")

    if ambiguous or lost:
        for old, matches in ambiguous[:5]:
            print(f"    ambiguous {old} -> {len(matches)} rows share its text")
        for old in lost[:5]:
            print(f"    not found {old}")
        raise SystemExit(
            "\n  Nothing was written. An ambiguous row means the repaired text no\n"
            "  longer identifies it; a missing one means its text changed in a way\n"
            "  `prepare` does not reproduce. Both need a human decision: resolve\n"
            "  them by hand in data/id_remap.csv, or drop those rows from the gold\n"
            "  set and say so in the report.")

    table = pd.DataFrame({"old_id": list(remap), "new_id": list(remap.values())})
    table.to_csv(config.DATA / "id_remap.csv", index=False, encoding="utf-8")
    print(f"\nwrote {config.DATA / 'id_remap.csv'}")

    targets = [config.POOL_PILOT, config.POOL_RANDOM, config.POOL_ENRICHED,
               config.POOL_VERIFY, config.POOL_DEBUG, config.PASS1, config.PASS2,
               config.INTER, config.GOLD / "calibration.csv",
               config.GOLD / "gold_blind.csv"]
    for path in targets:
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype=str)
        frame.columns = [c.lstrip("﻿") for c in frame.columns]
        if "id" not in frame.columns:
            continue
        hits = frame["id"].isin(remap).sum()
        if not hits:
            continue
        path.replace(path.with_suffix(".csv.bak"))
        frame["id"] = frame["id"].map(lambda v: remap.get(v, v))
        encoding = "utf-8-sig" if path.parent == config.GOLD else "utf-8"
        frame.to_csv(path, index=False, encoding=encoding)
        print(f"  {path.name}: {hits} ids rewritten (previous kept as .csv.bak)")

    print("\n  Teacher labels are keyed on the old ids and on the old prompt text,")
    print("  so re-run them rather than remapping:")
    print("      python teacher.py --provider gemini --pool pilot")
    print("  Then `python pilot.py --refresh-text` to pick up the repaired text.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass2", action="store_true",
                        help="write the reshuffled second pass (needs pass 1 finished)")
    parser.add_argument("--debug-slice", action="store_true",
                        help="write 100 train rows for reading the teacher's votes")
    parser.add_argument("--second-annotator", action="store_true",
                        help="blank sheet for a different person: real inter-annotator kappa")
    parser.add_argument("--refresh-text", action="store_true",
                        help="rebuild the sheets' text, keeping any labels already filled in")
    parser.add_argument("--remap-ids", action="store_true",
                        help="re-key the pool and sheets after the corpus was repaired")
    args = parser.parse_args()
    config.ensure_dirs()

    if args.remap_ids:
        remap_ids()
        return

    if args.second_annotator:
        filled, why = guard.sheet_filled(config.PASS1)
        if not filled:
            print(f"note: pass 1 is not finished ({why}).\n"
                  f"  That is fine - the two can be labelled in any order, as long as\n"
                  f"  neither annotator sees the other's sheet.\n")
        if not config.POOL_PILOT.exists():
            raise SystemExit(f"Missing {config.POOL_PILOT}. Run `python pilot.py` first.")
        pilot = pd.read_csv(config.POOL_PILOT)
        # A different order again, so nothing about position can carry over if
        # the two sheets are ever open side by side. Alignment is by id.
        reshuffled = pilot.sample(frac=1, random_state=config.SEED + 4).reset_index(drop=True)
        write_sheet(build_sheet(reshuffled), config.INTER, ANNOTATOR_B_HEADER.format(
            table=build_sheet_table()))
        print(f"\nGive them {config.INTER.name} and RUBRIC.md, and nothing else.\n"
              f"Do not answer rubric questions until they are finished: an answered\n"
              f"question makes the agreement measure the conversation instead of the\n"
              f"instructions. Ask them to put the questions in the notes column.\n"
              f"\nThen:  python agreement.py --inter --vs-teacher")
        return

    if args.refresh_text:
        for path in (config.PASS1, config.PASS2):
            if path.exists():
                refresh_text(path)
        return

    if args.debug_slice:
        debug = pick_debug()
        debug.to_csv(config.POOL_DEBUG, index=False, encoding="utf-8")
        print(f"wrote {config.POOL_DEBUG}  ({len(debug)} rows from the train split)")
        print(debug["pool"].value_counts().to_string())
        print("\nThese rows are never hand-labelled, so the teacher's answers on them\n"
              "are free to read. Next:\n"
              "    python teacher.py --provider gemini --pool debug\n"
              "    python agreement.py --teacher-self --pool debug")
        return

    if args.pass2:
        filled, why = guard.sheet_filled(config.PASS1)
        if not filled:
            raise SystemExit(
                f"Pass 1 is not finished ({why}).\n\n"
                f"  Fill {config.PASS1.name} completely before starting pass 2, or the\n"
                f"  two passes are not independent and the agreement figure means\n"
                f"  nothing.")
        if not config.POOL_PILOT.exists():
            raise SystemExit(f"Missing {config.POOL_PILOT}. Run `python pilot.py` first.")
        pilot = pd.read_csv(config.POOL_PILOT)
        # A different seed: same rows, different order, so position cannot cue
        # the earlier answer.
        reshuffled = pilot.sample(frac=1, random_state=config.SEED + 2).reset_index(drop=True)
        write_sheet(build_sheet(reshuffled), config.PASS2, PASS2_HEADER)
        print("\nLabel these from the rubric without reopening pass 1.")
        return

    pilot = pick_pilot()
    pilot.to_csv(config.POOL_PILOT, index=False, encoding="utf-8")
    print(f"wrote {config.POOL_PILOT}  ({len(pilot)} rows)")
    print(pilot["pool"].value_counts().to_string())
    print(pilot["platform"].value_counts().to_string())
    if "retrieved_by" in pilot and pilot["retrieved_by"].notna().any():
        print("\nenriched rows by retrieval heuristic:")
        print(pilot["retrieved_by"].value_counts().to_string())

    write_sheet(build_sheet(pilot), config.PASS1, PASS1_HEADER)
    notes = config.GOLD / "ANNOTATING.md"
    notes.write_text(INSTRUCTIONS.format(table=build_sheet_table()), encoding="utf-8")
    print(f"wrote {notes}")
    print("\nDebug the rubric first if you have not:  python pilot.py --debug-slice")


if __name__ == "__main__":
    main()
