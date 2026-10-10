"""Keeps the blind gold set blind.

The pilot rows are the only unanchored measurement in this project: labelled by
hand with no model output in sight, independently, by two people. Every other
number — teacher accuracy, student accuracy, the verified gold set — is either
derived from a model or corrected against one.

That measurement is also trivially easy to destroy by accident, because the
teacher's labels for those same rows are already sitting in
`data/teacher_labels.csv`. One glance is enough. So the tools refuse to show
model output for blind rows until both hand passes exist and are complete, and
say why.

Aggregate statistics over the blind rows (a disagreement rate, a label
distribution) do not anchor an individual judgement and are not blocked. What is
blocked is seeing the model's label for a specific row you have yet to label.
"""
from __future__ import annotations

import pandas as pd

import config
import schema


def sheet_filled(path) -> tuple[bool, str]:
    """Is this annotation sheet complete? Returns (filled, why not)."""
    if not path.exists():
        return False, f"{path.name} does not exist yet"
    frame = pd.read_csv(path, dtype=str).fillna("")
    frame.columns = [c.lstrip("﻿") for c in frame.columns]
    missing = [f.name for f in schema.FIELDS if f.name not in frame.columns]
    if missing:
        return False, f"{path.name} is missing the columns {', '.join(missing)}"
    blank_rows = frame[[f.name for f in schema.FIELDS]].apply(
        lambda col: col.str.strip() == "").any(axis=1).sum()
    if blank_rows:
        return False, f"{path.name} has {blank_rows} of {len(frame)} rows unfinished"
    return True, ""


def blind_locked() -> tuple[bool, str]:
    """True while model output for the blind rows must stay hidden.

    Unlocks on the first pass plus any second independent labelling of the same
    rows: a different person (the better measurement, inter-annotator) or the
    same person again after a gap (intra-annotator). Either one is enough, and
    until one of them exists there is nothing to protect but also nothing to
    compare against.
    """
    filled, why = sheet_filled(config.PASS1)
    if not filled:
        return True, why

    second = [(path, sheet_filled(path)) for path in (config.INTER, config.PASS2)]
    if any(ok for _, (ok, _) in second):
        return False, ""
    return True, ("no second labelling of the blind rows yet - "
                  + "; ".join(why for _, (_, why) in second))


def blind_ids() -> set[str]:
    """The ids reserved for the blind set, whether or not they are labelled."""
    if not config.POOL_PILOT.exists():
        return set()
    return set(pd.read_csv(config.POOL_PILOT, dtype=str)["id"])


def require_unlocked(what: str) -> None:
    """Stop with an explanation rather than leaking a blind row's labels."""
    locked, why = blind_locked()
    if not locked:
        return
    raise SystemExit(
        f"Refusing to {what}: the blind gold set is not finished ({why}).\n"
        f"\n"
        f"  Those {len(blind_ids())} rows are the only measurement in this project\n"
        f"  that no model has touched, and seeing a model's answer for a row before\n"
        f"  you label it anchors your own label to it.\n"
        f"\n"
        f"  Either a second person labelling the same rows, which is the better\n"
        f"  measurement, or your own second pass after a gap, unlocks this:\n"
        f"\n"
        f"      python pilot.py --second-annotator   # writes {config.INTER.name}\n"
        f"      python pilot.py --pass2              # or {config.PASS2.name}\n"
        f"\n"
        f"  To debug the rubric in the meantime, use rows that are never\n"
        f"  hand-labelled and so cannot be contaminated:\n"
        f"\n"
        f"      python pilot.py --debug-slice\n"
        f"      python teacher.py --provider gemini --pool debug\n"
        f"      python agreement.py --teacher-self --pool debug\n")


def drop_blind(frame: pd.DataFrame, id_column: str = "id") -> pd.DataFrame:
    """Remove blind rows from anything that shows per-row model output."""
    ids = blind_ids()
    if not ids:
        return frame
    return frame[~frame[id_column].astype(str).isin(ids)]
