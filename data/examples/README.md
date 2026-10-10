# What the corpus has to look like

The corpus is **not in this repository**. It is public forum and video-comment
text about people's own health, only partially de-identified, so it is mounted
rather than committed — see the de-identification note in the top-level
`README.md`.

Point the pipeline at a real corpus with either

```bash
export RUBRIC_CORPUS_DIR=/path/to/corpus      # PowerShell: $env:RUBRIC_CORPUS_DIR=...
```

or by copying the files below into `data/corpus/`.

## `corpus.csv` — one row per post or comment

| Column | Type | Required | Notes |
|---|---|---|---|
| `id` | str | yes | Stable and unique. `<platform>_<hash>` in this project, but any unique string works. |
| `date` | ISO date | yes | Carried through to the pools for the time-series breakdowns. |
| `platform` | str | yes | Source site. One of the stratification axes, and `MAX_PLATFORM_SHARE` caps any single value's share of the random pool. |
| `type` | str | yes | `post` / `comment` / `blog` or similar. The second stratification axis. |
| `text` | str | yes | The post body. Truncated to `DOC_CHARS` downstream; nothing here is modified in place. |

## `corpus_full.csv` — the same rows, with reply structure

| Column | Type | Required | Notes |
|---|---|---|---|
| `id` | str | yes | Joins to `corpus.csv`. |
| `is_reply` | bool | yes | Missing is read as `False`. |
| `parent_text` | str | yes | Blank for non-replies. The first `PARENT_CHARS` characters are shown to annotators and to the model as context — **the reply is what gets labelled, never the parent.** |

Two files rather than one because in this project they came from different
stages of an upstream cleaning pipeline. If you have a single file with all the
columns, point both `CORPUS` and `CORPUS_FULL` at it.

## `doc_topics.csv` — optional

| Column | Type | Notes |
|---|---|---|
| `id` | str | Joins to `corpus.csv`. |
| `topic` | int | Topic id from any upstream topic model; `-1` for outliers. |

Used for one thing only: spreading the draw *within* a stratum so a single
topic cannot dominate a cell. Without the file every row is assigned topic
`-1`, the spread degrades to a plain shuffle, and `sample.py` says so. No other
stage reads it.

## `corpus_sample.csv` / `corpus_sample_full.csv`

Ten synthetic rows in exactly that shape, written for this repository — they
are invented, not drawn from any corpus. Enough to exercise the schema
validation, the preprocessing in `teacher.prepare` (including the quoted-parent
split and the salutation redaction) and the sheet builders, and small enough to
read. Not enough to sample or train from.

```bash
RUBRIC_CORPUS_DIR=data/examples python teacher.py --provider none --pool pilot
```

The example files are named `corpus_sample*.csv` on purpose, so that pointing
`RUBRIC_CORPUS_DIR` here does **not** silently pick them up as a real corpus;
copy or rename them to `corpus.csv` / `corpus_full.csv` if that is what you
want.
