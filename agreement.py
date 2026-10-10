"""Four measurements of annotation reliability, each reported as what it is.

    python agreement.py --inter data/gold/pilot_annotator_b.csv   # the real gate
    python agreement.py --teacher-self --pool debug   # free, no human labels needed
    python agreement.py --intra                       # pass 1 vs pass 2
    python agreement.py --intra --vs-teacher          # test-retest + the teacher

**--inter** is Cohen's kappa between two people who labelled the same 100 rows
from the same written instructions, without discussing them. This is the figure
that gates the fields, and it is the only one of the four that measures whether
the definitions are *shared* rather than merely self-consistent. On this project
it landed well below the other three and removed one field outright — see the
results table in README.md.

**--teacher-self** needs no human labels at all. Every row was annotated three
times at a non-zero temperature; where the model cannot reproduce its own label
across three samples, the *definition* is ambiguous, not the model. Run this on
the debug slice before hand labelling, and fix the rubric from what it shows.
It also flags values that never appear, which cannot be evaluated later. Cheap
and worth doing, but do not mistake it for the gate: it passed every field at
the 25% threshold, and `--inter` then returned 0.136 on `health_claim`.

**--intra** is the kappa between two blind hand passes by the same person:
intra-annotator agreement, or test-retest reliability. It shows the definitions
are stable enough for one reader to reproduce. It does **not** show a different
person would agree, and it is an upper bound, because one annotator shares their
own reading of every edge case. Say so wherever it is quoted, and quote it
beside the inter figure rather than in place of it.

**--vs-teacher** is human-model agreement on the blind rows: convergent
validity, plus the teacher accuracy that caps what the student can learn. It is
blocked until both blind passes are finished, because seeing the teacher's
answer for a row you have not labelled yet anchors your label to it.

Thresholds (RUBRIC.md section 9): >= 0.75 proceed; 0.60-0.75 proceed and report
the figure as a ceiling; < 0.60 rewrite the definition or drop the field.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

import config
import guard
import schema

UNCLEAR = "UNCLEAR"
AGREEMENT_FLOOR = 0.67          # below this, the three votes did not have a majority
AMBIGUOUS_SHARE = 0.25          # a field this often split is a definition problem


def verdict(kappa: float) -> str:
    if kappa != kappa:
        return "n/a     not enough variation"
    if kappa >= 0.75:
        return "OK      proceed"
    if kappa >= 0.60:
        return "MARGINAL report as a ceiling"
    return "STOP    rewrite or drop"


# ---------- Sheets ----------

def read_sheet(path: Path) -> pd.DataFrame:
    filled, why = guard.sheet_filled(path)
    if not filled:
        raise SystemExit(f"{why}.\n\n  Both blind passes must be complete before the "
                         f"agreement between them means anything.")
    frame = pd.read_csv(path, dtype=str).fillna("")
    frame.columns = [c.lstrip("﻿") for c in frame.columns]
    return frame.set_index("id")


def align(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = a.index.intersection(b.index)
    if len(shared) < len(a):
        print(f"note: {len(a) - len(shared)} rows are not in both sheets, "
              f"using the {len(shared)} shared ones\n")
    return a.loc[shared], b.loc[shared]


def check_values(sheets: dict[str, pd.DataFrame]) -> bool:
    """Report cells that are not an allowed value, so the sheet can be fixed."""
    problems = []
    for field in schema.FIELDS:
        allowed = {"true", "false"} if field.kind == "bool" else set(field.values)
        for who, frame in sheets.items():
            for row_id, raw in frame[field.name].items():
                value = str(raw).strip()
                if not value or value == UNCLEAR:
                    continue
                parts = ([p.strip().lower() for p in value.split("|")]
                         if field.kind == "multi" else [value.lower()])
                bad = [p for p in parts if p not in allowed]
                if bad:
                    problems.append(f"  {who}  {row_id}  {field.name}: {bad}")
                elif field.kind == "multi":
                    if len(parts) > field.max_values:
                        problems.append(f"  {who}  {row_id}  {field.name}: "
                                        f"{len(parts)} values, max {field.max_values}")
                    elif "none" in parts and len(parts) > 1:
                        problems.append(f"  {who}  {row_id}  {field.name}: "
                                        f"none combined with {parts}")
    if problems:
        print(f"INVALID CELLS ({len(problems)}) - fix these first, they are excluded now:")
        print("\n".join(problems[:40]))
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        print()
    return not problems


# ---------- Pairing ----------

def paired(a: pd.Series, b: pd.Series) -> tuple[list[str], list[str], int]:
    """Aligned label pairs with UNCLEAR and blank rows dropped."""
    left, right, dropped = [], [], 0
    for x, y in zip(a, b):
        x, y = str(x).strip(), str(y).strip()
        if not x or not y or x == UNCLEAR or y == UNCLEAR:
            dropped += 1
            continue
        left.append(x.lower())
        right.append(y.lower())
    return left, right, dropped


def emotion_sets(series: pd.Series) -> list[set[str] | None]:
    out = []
    for raw in series:
        value = str(raw).strip()
        out.append(None if not value or value == UNCLEAR
                   else {p.strip().lower() for p in value.split("|")})
    return out


def emotion_kappa(a: pd.Series, b: pd.Series) -> tuple[float, str]:
    """Multi-label: mean of the per-value binary kappas, plus the detail line."""
    pairs = [(x, y) for x, y in zip(emotion_sets(a), emotion_sets(b))
             if x is not None and y is not None]
    if not pairs:
        return float("nan"), "no usable pairs"
    exact = sum(x == y for x, y in pairs) / len(pairs)
    jaccard = sum(len(x & y) / len(x | y) for x, y in pairs) / len(pairs)
    kappas = []
    for value in schema.BY_NAME["emotion"].values:
        xs = [int(value in x) for x, _ in pairs]
        ys = [int(value in y) for _, y in pairs]
        if set(xs) == set(ys) == {0}:
            continue                              # never used by either pass
        kappas.append(cohen_kappa_score(xs, ys))
    mean = sum(kappas) / len(kappas) if kappas else float("nan")
    detail = (f"exact set match {exact:.1%}, mean Jaccard {jaccard:.3f}, "
              f"{len(kappas)} values used, {len(a) - len(pairs)} dropped")
    return mean, detail


def report_kappa(a: pd.DataFrame, b: pd.DataFrame, title: str) -> dict[str, float]:
    print(f"{title}  ({len(a)} shared rows)\n")
    results = {}
    for field in schema.FIELDS:
        if field.kind == "multi":
            kappa, detail = emotion_kappa(a[field.name], b[field.name])
            results[field.name] = kappa
            print(f"  {field.name:<18} kappa {kappa:5.3f}  {verdict(kappa)}"
                  f"   (tier {field.tier})")
            print(f"  {'':<18} {detail}")
            continue
        left, right, dropped = paired(a[field.name], b[field.name])
        if len(left) < 2 or len(set(left) | set(right)) < 2:
            results[field.name] = float("nan")
            print(f"  {field.name:<18} {verdict(float('nan'))}   ({len(left)} pairs)")
            continue
        kappa = cohen_kappa_score(left, right)
        results[field.name] = kappa
        note = f"tier {field.tier}"
        if dropped:
            note += f", {dropped} UNCLEAR dropped"
        print(f"  {field.name:<18} kappa {kappa:5.3f}  {verdict(kappa)}   ({note})")
    return results


# ---------- Teacher self-disagreement ----------

def load_teacher(pool: str) -> pd.DataFrame:
    path = (config.TEACHER_LABELS if pool == "all" else
            config.TEACHER_LABELS.with_name(f"teacher_labels_{pool}.csv"))
    if not path.exists() and config.TEACHER_LABELS.exists() and pool == "pilot":
        path = config.TEACHER_LABELS          # written before the per-pool split
    if not path.exists():
        raise SystemExit(
            f"Missing {path}.\n\n"
            f"  Run:  python teacher.py --provider gemini --pool {pool}")
    return pd.read_csv(path, dtype=str)


def report_teacher_self(pool: str, examples: int) -> None:
    teacher = load_teacher(pool)
    print(f"TEACHER SELF-DISAGREEMENT  ({len(teacher)} rows, "
          f"{teacher['votes'].astype(int).max()} votes each, pool '{pool}')\n")
    print("  Where the model cannot reproduce its own label across three samples,")
    print("  the definition is ambiguous. This needs no human labels.\n")

    ambiguous = []
    for field in schema.FIELDS:
        agree = teacher[f"agree_{field.name}"].astype(float)
        unanimous = (agree >= 0.999).mean()
        split = (agree < AGREEMENT_FLOOR).mean()
        flag = "  <- definition problem" if split > AMBIGUOUS_SHARE else ""
        if split > AMBIGUOUS_SHARE:
            ambiguous.append(field.name)
        print(f"  {field.name:<18} unanimous {unanimous:5.1%}   "
              f"no majority {split:5.1%}   mean {agree.mean():.3f}"
              f"   (tier {field.tier}){flag}")

    print("\n  LABEL DISTRIBUTION (a value never used cannot be evaluated later)")
    for field in schema.FIELDS:
        if field.kind == "multi":
            counts = (teacher[field.name].fillna("").str.split("|")
                      .explode().str.strip().value_counts())
            universe = field.values
        else:
            counts = teacher[field.name].str.lower().value_counts()
            universe = ("true", "false") if field.kind == "bool" else field.values
        used = ", ".join(f"{v}={counts.get(v, 0)}" for v in universe)
        unused = [v for v in universe if counts.get(v, 0) == 0]
        mark = f"   <- never used: {', '.join(unused)}" if unused else ""
        print(f"  {field.name:<18} {used}{mark}")

    # parse_ok is false whenever *anything* had to be corrected, which is mostly
    # schema-rule violations rather than bad JSON. Saying "JSON parsed cleanly"
    # here would misreport it, and this number is headed for a report.
    clean = teacher["parse_ok"].astype(str).str.lower().eq("true").mean()
    print(f"\n  Answers needing no correction: {clean:.1%} of rows")
    problems = teacher.get("problems")
    if problems is not None:
        reasons = (problems.fillna("").str.split("; ").explode()
                   .str.replace(r"^vote \d+: ", "", regex=True))
        reasons = reasons[reasons.str.strip() != ""].value_counts()
        for reason, count in reasons.head(5).items():
            print(f"    {count:4d}  {reason}")

    if examples:
        show_examples(teacher, pool, examples)

    print()
    if ambiguous:
        print(f"ACTION: {', '.join(ambiguous)} split the vote on more than "
              f"{AMBIGUOUS_SHARE:.0%} of rows.")
        print("  Rewrite those definitions in RUBRIC.md and bump its version before")
        print("  hand labelling. Cut order if a definition will not come clean:")
        print("  clinical_stage (tier 3), then the tier 2 fields.")
    else:
        print(f"ACTION: no field splits the vote on more than {AMBIGUOUS_SHARE:.0%} "
              f"of rows. The definitions are usable; go to the blind passes.")


def show_examples(teacher: pd.DataFrame, pool: str, count: int) -> None:
    """The rows the teacher was least sure about, with their text.

    Per-row model output for blind rows is withheld: reading it before labelling
    those rows by hand would anchor the only unanchored measurement there is.
    """
    pool_path = {"debug": config.POOL_DEBUG, "pilot": config.POOL_PILOT}.get(pool)
    if pool_path is None or not pool_path.exists():
        return
    safe = guard.drop_blind(teacher)
    if safe.empty:
        locked, why = guard.blind_locked()
        if locked:
            print(f"\n  (Example rows withheld: every row here is in the blind set and "
                  f"{why}.\n   Use `--pool debug` to debug the rubric safely.)")
        return

    texts = pd.read_csv(pool_path, dtype=str).set_index("id")["text"]
    agree_columns = [f"agree_{f.name}" for f in schema.FIELDS]
    safe = safe.assign(_mean=safe[agree_columns].astype(float).mean(axis=1))
    worst = safe.nsmallest(count, "_mean")

    print(f"\n  {len(worst)} ROWS THE TEACHER WAS LEAST SURE ABOUT")
    print("  Read these against the rubric: they are where the definitions bend.\n")
    for _, row in worst.iterrows():
        split = [f.name for f in schema.FIELDS
                 if float(row[f"agree_{f.name}"]) < AGREEMENT_FLOOR]
        text = " ".join(str(texts.get(row["id"], "")).split())[:260]
        print(f"  --- {row['id']}  split on: {', '.join(split) or 'none'}")
        print(f"      {text}")


# ---------- Human vs teacher ----------

def second_labelling() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """The first pass and whichever second labelling of the same rows exists.

    A different person is the better measurement, so it wins when both are
    there; the annotator's own second pass is the fallback.
    """
    for path, source in ((config.INTER, "two annotators"),
                         (config.PASS2, "two passes by one annotator")):
        if guard.sheet_filled(path)[0]:
            first, second = align(read_sheet(config.PASS1), read_sheet(path))
            return first, second, source
    guard.require_unlocked("score the teacher against the hand labels")
    raise SystemExit("No second labelling of the blind rows exists yet.")


def score(truth: list[str], predicted: list[str]) -> tuple[float, float]:
    accuracy = sum(t == p for t, p in zip(truth, predicted)) / len(truth)
    return accuracy, f1_score(truth, predicted, average="macro", zero_division=0)


def report_vs_teacher(pool: str) -> None:
    """The teacher against the gold set, split by how each cell was produced.

    `agreed` cells are independent of any discussion; `adjudicated` ones were
    settled in the calibration session and are by construction the contested
    rows. Scoring them separately says how much harder the contested rows are
    for the model, and keeps the independent figure quotable on its own.
    """
    guard.require_unlocked("score the teacher against the hand labels")
    gold_path = config.GOLD / "gold_blind.csv"
    if not gold_path.exists():
        raise SystemExit(f"Missing {gold_path}. Run `python calibrate.py --apply` first.")
    gold = pd.read_csv(gold_path, dtype=str).fillna("")
    gold.columns = [c.lstrip("﻿") for c in gold.columns]
    gold = gold.set_index("id")
    sources = pd.read_csv(config.GOLD / "gold_blind_sources.csv", dtype=str).fillna("")
    sources.columns = [c.lstrip("﻿") for c in sources.columns]
    sources = sources.set_index("id")

    teacher = load_teacher(pool).set_index("id")
    shared = gold.index.intersection(teacher.index)
    if shared.empty:
        raise SystemExit("The teacher labels and the gold set share no rows.")

    print(f"\n\nTEACHER vs GOLD  ({len(shared)} rows)\n")
    print("  The teacher's accuracy here caps what the student model can learn.")
    print("  'independent' = both annotators reached it separately; 'contested' =")
    print("  settled in the calibration session, so the harder rows by construction.\n")
    print(f"  {'field':<18} {'independent':>22}   {'contested':>20}   all")
    caveats = []
    for field in schema.FIELDS:
        buckets: dict[str, tuple[list[str], list[str]]] = {
            "agreed": ([], []), "adjudicated": ([], [])}
        excluded = 0
        for row_id in shared:
            origin = str(sources.at[row_id, field.name]).strip()
            value = str(gold.at[row_id, field.name]).strip().lower()
            if origin not in buckets or not value or value == UNCLEAR.lower():
                excluded += 1
                continue
            buckets[origin][0].append(value)
            buckets[origin][1].append(str(teacher.at[row_id, field.name]).strip().lower())

        cells = []
        for name in ("agreed", "adjudicated"):
            truth, predicted = buckets[name]
            if len(truth) < 5:
                cells.append(f"{'n/a':>12} (n={len(truth)})")
                continue
            accuracy, macro = score(truth, predicted)
            cells.append(f"{accuracy:5.1%} F1 {macro:.2f} (n={len(truth)})")

        combined = ([*buckets["agreed"][0], *buckets["adjudicated"][0]],
                    [*buckets["agreed"][1], *buckets["adjudicated"][1]])
        overall = (f"{score(*combined)[0]:5.1%}" if len(combined[0]) >= 5 else "n/a")
        note = " (exploratory)" if field.name in schema.EXPLORATORY else ""
        print(f"  {field.name:<18} {cells[0]:>22}   {cells[1]:>20}   {overall}{note}")

        if excluded >= 0.2 * len(shared) and not buckets["adjudicated"][0]:
            caveats.append((field.name, excluded, len(shared)))

    if caveats:
        print("\n  OPTIMISM CAVEAT - quote this wherever these fields are reported:")
        for name, excluded, total in caveats:
            print(f"    {name}: scored on {total - excluded} of {total} rows. The "
                  f"{excluded} excluded\n      are the ones the two annotators "
                  f"disagreed on and no session settled,\n      so by construction "
                  f"they are the harder rows and this figure is optimistic.")


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-self", action="store_true",
                        help="the model's own vote split: needs no human labels")
    parser.add_argument("--intra", action="store_true",
                        help="kappa between the two blind hand passes")
    parser.add_argument("--vs-teacher", action="store_true",
                        help="human-model agreement, after both passes")
    parser.add_argument("--inter", nargs="?", const="", metavar="SHEET",
                        help="a second annotator's sheet (default: data/gold/"
                             "pilot_annotator_b.csv) for a real inter-annotator kappa")
    parser.add_argument("--pool", default="debug",
                        choices=("debug", "pilot", "random", "enriched", "all"),
                        help="which teacher labels to read (default: debug)")
    parser.add_argument("--examples", type=int, default=6,
                        help="rows to print from the least certain (0 for none)")
    args = parser.parse_args()

    if not any((args.teacher_self, args.intra, args.vs_teacher,
                args.inter is not None)):
        parser.error("pick at least one of --teacher-self, --intra, --vs-teacher, --inter")

    if args.teacher_self:
        report_teacher_self(args.pool, args.examples)

    results: dict[str, float] = {}
    if args.intra:
        first, second = align(read_sheet(config.PASS1), read_sheet(config.PASS2))
        check_values({"pass1": first, "pass2": second})
        if args.teacher_self:
            print("\n")
        results = report_kappa(first, second,
                               "INTRA-ANNOTATOR AGREEMENT (test-retest)")
        print("\n  Report this as intra-annotator agreement and as an upper bound:")
        print("  one annotator shares their own reading of every edge case.")

    if args.inter is not None:
        other = read_sheet(Path(args.inter) if args.inter else config.INTER)
        first, other = align(read_sheet(config.PASS1), other)
        check_values({"pass1": first, "other": other})
        print("\n")
        inter = report_kappa(first, other, "INTER-ANNOTATOR AGREEMENT (two people)")
        print("\n  This is the real thing. Quote it in preference to the intra figure,")
        print(f"  noting it covers {len(first)} rows.")
        results = inter or results

    if results:
        stop = [name for name, value in results.items()
                if value == value and value < 0.60]
        print()
        if stop:
            print(f"GATE: below 0.60 -> {', '.join(stop)}")
            print("  Rewrite the definition and re-pilot, or drop the field. Cut order:")
            print("  clinical_stage (tier 3), then the tier 2 fields. A tier 1 field")
            print("  here means the task design needs revisiting, not the field.")
        else:
            print("GATE: every field at or above 0.60 -> proceed to full labelling")

    if args.vs_teacher:
        second_labelling()          # enforces the blind guard before anything shows
        report_vs_teacher("pilot")


if __name__ == "__main__":
    main()
