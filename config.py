"""Paths, constants and run folders for the annotation pipeline.

Everything the pipeline writes goes under this directory. The one thing it
needs from outside is the corpus, which is **not** in this repository: it is
public forum text about people's own health and only partially de-identified,
so it is mounted rather than committed (see the de-identification note in
`README.md`).

Point `CORPUS_DIR` at it either by setting the environment variable

    RUBRIC_CORPUS_DIR=/path/to/corpus

or by dropping the three files below into `data/corpus/`. With neither, the
sampling and teacher stages will say what is missing and stop; every stage
downstream of `data/sft/` runs from the committed metrics and the synthetic
corpus in `data/examples/`.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------- Inputs (read-only) ----------
# A cleaned corpus of posts, one row per post or comment, with at least:
#   id, date, platform, type, text        and, in the *_full variant,
#   id, is_reply, parent_text
# `data/examples/README.md` documents the full expected schema and ships a
# synthetic ten-row sample in the same shape.
CORPUS_DIR = Path(os.environ.get("RUBRIC_CORPUS_DIR") or ROOT / "data" / "corpus")
CORPUS = CORPUS_DIR / "corpus.csv"
# The same rows with the cleaning columns, used here only for `parent_text`.
CORPUS_FULL = CORPUS_DIR / "corpus_full.csv"
# Optional. Topic assignments from an upstream topic model, used only to spread
# the sample across topics; sampling falls back to a single topic without it.
DOC_TOPICS = CORPUS_DIR / "doc_topics.csv"

# ---------- Outputs ----------
DATA = ROOT / "data"
POOL_RANDOM = DATA / "pool_random.csv"
POOL_ENRICHED = DATA / "pool_enriched.csv"
POOL_PILOT = DATA / "pool_pilot.csv"       # the blind rows, labelled by hand
POOL_DEBUG = DATA / "pool_debug.csv"       # train rows used to debug the rubric
POOL_VERIFY = DATA / "pool_verify.csv"     # gold rows that get a prefilled sheet
SAMPLE_MANIFEST = DATA / "sample_manifest.json"
TEACHER_LABELS = DATA / "teacher_labels.csv"
TEACHER_CACHE = DATA / "teacher_cache.json"
GOLD = DATA / "gold"                        # human annotations go here
PASS1 = GOLD / "pilot_pass1.csv"            # annotator A, first blind pass
PASS2 = GOLD / "pilot_pass2.csv"            # annotator A, second blind pass
INTER = GOLD / "pilot_annotator_b.csv"      # annotator B, one blind pass
VERIFY_SHEET = GOLD / "gold_verified.csv"   # prefilled, corrected by hand
PROMPTS = ROOT / "prompts"                  # `--provider none` writes here
RUNS = ROOT / "runs"

# ---------- API ----------
ENV_FILE = ROOT / ".env"                    # gitignored; holds GEMINI_API_KEY
KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_API_KEY")
DEFAULT_MODEL = "gemini-3.1-flash-lite"     # 2.5-flash-lite retires 2026-10-16
DEFAULT_VOTES = 3
DEFAULT_TEMPERATURE = 0.6                   # > 0 so the three votes can differ
RPM_PAUSE = 0.25                            # seconds between calls

# ---------- Truncation ----------
# Both the teacher prompt and the student input use these, so training and
# inference see a document the same way. Mean cost at 1200 chars: ~88 tokens.
DOC_CHARS = 1200
PARENT_CHARS = 200

# ---------- Sampling ----------
# Frozen. Every pool, the pilot draw and the committed gold ids come from this
# value; changing it re-draws all of them and the ids in
# `data/gold/gold_blind.csv` will no longer match any row.
SEED = 5703
N_RANDOM = 7000                             # unbiased stratified pool
N_ENRICHED = 2000                           # retrieved for the rare fields
# The gold set is split by how it was produced, and the two halves are never
# pooled for a headline number:
#   gold_blind    labelled from scratch with no model output visible, by two
#                 annotators independently. The only unanchored measurement.
#   gold_verified a second model's labels, corrected by hand. Reported
#                 separately and always described as adjudicated, not
#                 independent.
N_GOLD_RANDOM = 250                         # human-labelled, unbiased metrics
N_GOLD_ENRICHED = 150                       # human-labelled, rare-field metrics
N_VAL = 1000
# The pilot deliberately over-weights the enriched pool: 100 purely random rows
# would hold ~3 `dismissed` cases, far too few for a kappa on care_experience.
N_PILOT_RANDOM = 60
N_PILOT_ENRICHED = 40
# Rows drawn from the train split purely to read the teacher's self-disagreement
# and debug the rubric. They are never hand-labelled, so looking at model output
# on them cannot contaminate anything.
N_DEBUG = 100
MAX_PLATFORM_SHARE = 0.40                   # the largest platform is 65% of rows
MIN_PER_STRATUM = 25
LENGTH_BINS = (25, 100)                     # words: short / medium / long


def ensure_dirs() -> None:
    for path in (DATA, GOLD, PROMPTS, RUNS):
        path.mkdir(parents=True, exist_ok=True)


def require_corpus() -> None:
    """Fail with instructions rather than a bare FileNotFoundError."""
    missing = [p for p in (CORPUS, CORPUS_FULL) if not p.exists()]
    if not missing:
        return
    raise SystemExit(
        "The corpus is not in this repository and was not found at\n"
        f"  {CORPUS_DIR}\n"
        "Missing: " + ", ".join(p.name for p in missing) + "\n\n"
        "Set RUBRIC_CORPUS_DIR to the directory holding corpus.csv and\n"
        "corpus_full.csv, or copy them into data/corpus/. See\n"
        "data/examples/README.md for the expected columns and a synthetic\n"
        "sample in the same shape."
    )


def new_run(tag: str) -> Path:
    run = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_{tag}"
    run.mkdir(parents=True, exist_ok=True)
    return run
