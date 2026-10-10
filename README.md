# rubric-distill

Distilling a 2,637-token annotation rubric into the weights of a 4B model, on
one 8 GB consumer GPU — and building the reliability measurement that says
whether the labels it learned were worth learning.

The task: annotate public posts about women's heart health (patient forums,
video comments, health blogs) against a six-field schema — what the post *does*,
the affect it expresses, whether it describes a care experience, whose health it
concerns, whether it makes a generalisable health claim, and whether it is
women-specific. Sentiment is derived from the emotion field rather than
predicted, so it cannot contradict it.

**Three results, in the order they matter:**

| | |
|---|---|
| **Prompt cost** | System prompt **2,637 → 142 tokens (−94.6%)** at a *higher* mean score. The rubric moves out of the context window and into the weights. |
| **Where the ceiling is** | On `care_experience`, macro-F1 is **0.84 on rows two annotators independently agreed on, 0.26 on rows that needed adjudication** (emotion: 0.27 vs 0.08). The model's hard rows are the humans' hard rows. Ambiguity, not model capacity, is the limit — and only a gold set split by provenance makes that visible at all. |
| **Score** | Mean macro-F1 over six fields: zero-shot short prompt 0.547 → zero-shot full rubric 0.595 → **fine-tuned 0.617**. A real but modest gain; the cost figure is the headline, not this. |

Trained on an RTX 4060 Laptop 8 GB: QLoRA over Qwen3-4B-Instruct-2507, **6.16 GiB
peak, 106 minutes**.

---

## Why this is built the way it is

Most of the work here is not the fine-tune. It is the measurement around it.
Four things are worth reading the code for.

### 1. The reliability figure that mattered was the one that cost the most

Four measurements of annotation reliability were available over the same 100
rows. Three are cheap and one requires a second human for ninety minutes:

| Figure | What it actually shows | Cost |
|---|---|---|
| Teacher self-disagreement | which definitions are *stated* ambiguously | free |
| Intra-annotator κ (test–retest) | the definitions are *stable* for one reader | 2 × 1.5 h |
| Human–model agreement | the labels track more than one reader | free, once the passes exist |
| **Inter-annotator κ** | **the definitions are *shared*** | 1.5 h of someone else's time |

**The three cheap figures all passed. The real one failed on three fields.**
Teacher self-disagreement called `care_experience` 98% unanimous and
`health_claim` 94%; two humans reading the same rubric then returned κ **0.293**
and **0.136** on those same fields.

```
speech_act        0.646   MARGINAL  report as a ceiling
emotion           0.547   STOP      rewrite or drop
care_experience   0.293   STOP      rewrite or drop
subject           0.750   MARGINAL  report as a ceiling
health_claim      0.136   STOP      rewrite or drop
womens_specific   0.709   MARGINAL  report as a ceiling
```

A model agreeing with itself at temperature 0.6 shows a definition can be
*followed* consistently, not that it is the definition a second reader would
independently arrive at. That gap is the single most useful thing this project
found, and nothing cheaper than a second annotator would have found it.
`agreement.py` carries the κ gate and the documented field cut order.

The independence is protected by the brief, not by good intentions
(`docs/annotator-b-brief.md`): different row order compared by post id, every
unasked question recorded in `notes`, and a hard rule that a borderline case is
never asked about — *you ask, you are told, and the number stops measuring the
instructions and starts measuring the conversation.*

### 2. Disagreement was diagnosed before it was fixed

The disagreements were read first, and they were **systematic**: each annotator
had applied a consistent reading the other did not share. That is an
underspecified definition, not a careless annotator — so the definitions got
fixed, not the people.

One 66-minute session over 61 posts and 88 decisions, both annotators at one
screen with the rubric open (`calibrate.py`, `docs/calibration.md`). Rows are
grouped by *which pair of labels clashed*, so one principle settles a whole
block at a time. The design that makes the session worth more than its output
is the **`rule_needed` column**: every row records the rule that settled it,
worded so that someone holding only the rubric reaches the same answer. Filled
61/61, and rubric versions 1.3 and 1.4 were written from it rather than from
guesses. Mid-session the annotator found that v1.2's `care_experience`
definition contradicted its own boundary case — which is the format working.

Outcome: `emotion` and `care_experience` fully resolved (44 adjudicated each).
`clinical_stage` dropped on κ 0.319. `health_claim` kept but marked
**exploratory**, because 65 of its 100 rows could not be settled with the rubric
open.

And the principle that fell out of it: **a row two people cannot settle with
the rubric in front of them is a row the model should not be scored on either.**

### 3. The gold set records how every cell was produced

`data/gold/gold_blind_sources.csv` marks each cell `agreed`, `adjudicated` or
`unresolved`, and `evaluate.py` reports every figure split that way. The two
halves are never pooled into a headline number.

This is what surfaces the real finding. `care_experience` macro-F1 is 0.84 on
agreed cells and 0.26 on adjudicated ones; `emotion` 0.27 and 0.08. A single
pooled number (0.57 and 0.12) would have hidden it completely and invited the
wrong conclusion — that the model needs to be bigger, when what the *labels*
need is a definition two people can share.

### 4. Contamination is prevented in code, not by discipline

The blind rows are the only unanchored measurement in the project, and the
teacher's labels for those same rows sit on disk the whole time. One glance
destroys the measurement permanently. So `guard.py` makes every tool refuse to
show per-row model output for blind rows until both hand passes are complete,
and point at the debug slice instead. Aggregate statistics over blind rows are
*not* blocked, because a disagreement rate does not anchor an individual
judgement.

Two further separations of the same kind:

- **Two pools, kept apart.** `pool_random` is stratified over
  platform × type × length with a 40% cap on any one platform (the largest is
  65% of the corpus and would otherwise swamp the sample). Because the strata
  are not proportional every row carries a `weight`: corpus-level statements
  must use it, model-quality metrics must not. `pool_enriched` is
  keyword-retrieved for the three rare fields. The retrieval heuristics decide
  only what gets *looked at* and never become labels, and enriched rows never
  enter the random pool or its gold set, so the unbiased estimates stay
  unbiased.
- **Gold rows never enter training**, and the student's prompt is not the
  teacher's. `dataset.py` enforces both.

---

## Results

### Three settings over the same base weights

That is what makes the comparison clean — nothing changes but where the rubric
lives.

The per-run JSON behind every figure below is committed under `runs/eval/`, so
`python evaluate.py --compare` prints the comparison in a fresh clone with no
corpus, no weights and no GPU.

| Setting | Input tokens | Mean macro-F1 | s/post |
|---|---|---|---|
| `zero-short` — base model, minimal prompt | 142 + doc | 0.547 | 0.89 |
| `zero-rubric` — base model, whole rubric in context | 2,637 + doc | 0.595 | 0.96 |
| **`adapter`** — rubric distilled into the weights | **142 + doc** | **0.617** | 5.19 |

Per field, macro-F1 over all scorable cells:

| Field | zero-short | zero-rubric | adapter | agreed cells | adjudicated cells |
|---|---|---|---|---|---|
| `speech_act` | 0.607 | 0.736 | **0.825** | 0.825 | — |
| `emotion` | 0.066 | **0.137** | 0.120 | 0.269 | 0.083 |
| `care_experience` | 0.501 | 0.498 | **0.570** | 0.843 | 0.264 |
| `subject` | 0.787 | **0.938** | 0.930 | 0.930 | — |
| `health_claim` | 0.552 | 0.569 | **0.600** | 0.600 | — |
| `womens_specific` | **0.767** | 0.695 | 0.656 | 0.656 | — |

The adapter wins on `speech_act` (+0.089) and `care_experience` (+0.073) and
loses on `emotion` (−0.017) and `womens_specific` (−0.111 against the best
baseline). `health_claim` is scored on 35 of 100 rows and is reported as
exploratory throughout — at κ 0.136 the number is not evidence of anything.

### Caveats that belong next to those numbers

- **5.19 s/post against 0.89 is not a model property.** The baselines run an
  optimised 4-bit `qwen3:4b` through Ollama; the adapter runs unoptimised HF
  transformers + PEFT with the LoRA unmerged. Merging the adapter and serving
  it through vLLM is not done, so this is an artefact of the serving stack, not
  a fair latency comparison.
- **The final checkpoint is not the best one.** `val_loss` bottomed at 0.0385
  at step 2000 and rose to 0.0406 by step 2877 — a mild overfit. No best-model
  callback was configured, so the reported adapter is the last step.
- `emotion` is weak in every setting. At inter-annotator κ 0.547 it is near the
  limit of what the labels can support, and the adjudicated-cell score of 0.083
  says the remaining errors are concentrated exactly where the humans could not
  agree either.
- 100 blind rows is a small test set. Every figure above carries wide intervals
  and none of the between-setting differences is individually significant; the
  token-cost result does not depend on that.

### Training run

`runs/20261008-041627_sft2k/metrics.json`

| | |
|---|---|
| Base | Qwen3-4B-Instruct-2507, 4-bit NF4 |
| LoRA | r=16, α=32, dropout 0.05, all seven projection layers |
| Data | 1,919 train / 292 val, teacher-labelled |
| Schedule | 3 epochs, 2,877 steps, lr 2e-4, batch 2 × grad-accum 8, seq 1024 |
| Hardware | RTX 4060 Laptop 8 GB — **6.16 GiB peak, 105.9 minutes** |

Fitting this in 8 GB came down to three things, all recorded in
`requirements-train.txt`: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(the allocator otherwise fragments and OOMs well below the real limit),
disabling the NVIDIA driver's CUDA system-memory fallback (with it on,
exceeding VRAM silently spills to host RAM and training gets ~20× slower
instead of raising `OutOfMemoryError`, which is far harder to diagnose), and
checking that `bitsandbytes` 4-bit *kernels* actually run rather than merely
import — a CPU-only Windows wheel imports cleanly and then fails at the first
forward pass. `smoke_train.py` tests the kernels for that reason.

---

## Layout

```
RUBRIC.md            the human rubric: definitions, boundary cases, priority
                     rules, the reliability gates and a change log
schema.py            fields, allowed values, validation, derived sentiment
fewshot.py           seven hand-labelled posts that anchor the teacher prompt
config.py            paths, truncation, sampling and API constants

sample.py            builds the random and enriched pools from the corpus
pilot.py             the blind sheets (both passes, and annotator B) and the
                     rubric debug slice
teacher.py           model annotation; --provider none is the review gate
verify.py            the prefilled half of the gold set, routed by model
                     disagreement
agreement.py         the four reliability figures, with the kappa gate
calibrate.py         the calibration session: resolve disagreements together

dataset.py           teacher labels -> SFT train/val jsonl
train.py             QLoRA fine-tune
smoke_train.py       proves the 4-bit kernels work before the real run
evaluate.py          scores a setting against the blind gold set, split by
                     how each gold cell was produced

docs/                annotator briefs and the calibration session notes
data/examples/       a synthetic corpus so the pipeline runs without real data
data/gold/           the label-only blind gold set and its per-cell provenance
prompts/system.txt   the teacher prompt of record (2,637 tokens)
runs/                metrics and eval results, one folder per run
```

`schema.py` is authoritative for field names and values; `RUBRIC.md` is
authoritative for how to decide between them. Changing a value means
re-labelling, so changes go through the rubric's change log.

---

## Running it

```bash
pip install -r requirements.txt            # add requirements-train.txt to train
```

The corpus is not in this repository (see the next section). Point the pipeline
at one:

```bash
export RUBRIC_CORPUS_DIR=/path/to/corpus   # PowerShell: $env:RUBRIC_CORPUS_DIR=...
```

`data/examples/README.md` documents the expected columns and ships ten
synthetic rows in the same shape, which are enough to exercise the schema, the
preprocessing and the sheet builders.

```bash
python sample.py --dry-run         # print the allocation, write nothing
python sample.py                   # write the two pools
python teacher.py --provider none  # build prompts, send nothing  <- review this
```

**`teacher.py --provider none` sends nothing.** It writes the exact system
instruction, a sample of the exact user messages, and a cost and data-exposure
report. Read `prompts/exposure_report.txt` before any paid or outbound run.

```bash
# 1. debug the rubric with no human labels at all
python pilot.py --debug-slice
python teacher.py  --provider gemini --pool debug
python agreement.py --teacher-self --pool debug

# 2. the blind gold set
python pilot.py                               # annotator A, pass 1
python pilot.py --second-annotator            # annotator B, same rows, reshuffled
python agreement.py --inter data/gold/pilot_annotator_b.csv    # the gate
python calibrate.py                           # for any field below 0.60
python pilot.py --pass2                       # optional test-retest, 3+ days later
python agreement.py --intra --vs-teacher

# 3. the prefilled half, only after the gate passes
python verify.py --build-pool
python teacher.py --provider gemini    --pool verify --out data/verify_model_a.csv
python teacher.py --provider anthropic --pool verify --out data/verify_model_b.csv \
      --model claude-haiku-4-5-20251001
python verify.py data/verify_model_a.csv data/verify_model_b.csv

# 4. the full teacher run, then train and score
python teacher.py --provider gemini
python dataset.py
python smoke_train.py              # proves the 4-bit kernels run
python train.py
python evaluate.py --mode adapter --adapter runs/<run>/adapter
python evaluate.py --compare
```

Teacher answers are cached by prompt hash in `data/teacher_cache.json` and
saved after each call, so a stopped run resumes, and a re-run costs nothing and
returns the same labels.

One preprocessing function is used everywhere. `teacher.prepare` does the quote
splitting, the redaction and the truncation (1,200 characters of post, 200 of
parent context), and the teacher prompt, the annotation sheets and the verify
sheet all call it, so every stage sees byte-identical text. Changing anything in
it means re-labelling; `pilot.py --refresh-text` rebuilds a part-filled sheet's
text while keeping the labels and lists the rows that moved.

---

## Data, and what is not in this repository

**No post text is committed here.** The corpus is public forum and
video-comment text in which people describe their own cardiac symptoms,
diagnoses and treatment, and it is only partially de-identified. Sample pools,
SFT files, the teacher cache and every annotation sheet carry that text
verbatim, so none of them are tracked — `.gitignore` says exactly what is
excluded and why.

What *is* here: the code, the rubric, the label-only blind gold set
(`id` + labels + per-cell provenance, no text), the run metrics, and ten
synthetic example rows.

Two exceptions, both deliberate:

- `fewshot.py` carries seven short excerpts of real public posts, because the
  teacher prompt is not reproducible without them. The same seven appear inside
  `prompts/system.txt`, which is that prompt. They are excluded from the sample
  pools, so they are in neither training nor evaluation.
- `data/corpus_lowercase_words.json` is a 14,967-word lowercase vocabulary
  derived from the corpus. It is a word list rather than text, and it is what
  the redactor needs in order to run at all.

The short quoted phrases in `RUBRIC.md` are illustrative boundary cases, not
excerpts.

### De-identification

This pipeline applies its own **salutation-only** redaction, touching about 2%
of rows: the handle of the person being *addressed* is the de-identification
risk worth removing, and a salutation is a precise place to remove it. A
general Title-Case sweep would also eat organisation names and clinical terms
and cost label quality, and clinician or public-figure names in the body are
left in deliberately — they are not the subject's personal data and removing
them takes clinical meaning with them.

The stoplist is derived from the corpus rather than hand-written. The first
version used a hand-written list and replaced "Thank" 93 times, "Welcome" 41,
"Please" 29, "Sorry" 16 and — worst — "SCAD" 5 times, which is a clinical term
the `womens_specific` field depends on: 881 replacements corpus-wide, a small
minority of them actual handles. It now treats a token the corpus uses in lower
case at least five times as a word, not a name. 336 replacements, and what
remains is handles. All-caps tokens and anything under three characters are
left alone.

**This reduces name exposure; it does not eliminate it.** Real handles survive
in salutations the pattern does not match, and clinician names survive by
design. Treat the corpus as unredacted for names regardless of what any
upstream pipeline reports, and verify that against the file rather than against
a claim — the corpus this was built on carried **zero** redaction placeholders
in 53,864 rows while an upstream column claimed otherwise on 14,419 of them.

### Sending text to a third party

Teacher labelling sends roughly 9,000 de-identified post excerpts to a
commercial LLM API. That is a real change in third-party data exposure and it
belongs in a project's data-handling documentation explicitly, with the
residual-name caveat above stated plainly rather than buried.
`teacher.py --provider none` exists so that the exact text, the exact cost and
a residual-PII scan can be reviewed *before* anything leaves the machine.

### Known upstream bug, patched here

Replies on one forum arrive with the quoted parent comment prepended and the
site's own "Jump to this post" link left in the middle, so the stored text is
parent + UI chrome + reply run together: 642 rows, 20.7% of that platform's
rows. The quoted parent routinely contains questions, which drags `speech_act`
towards `asks_question` when the reply is doing something else.
`teacher.split_quoted_parent` cuts on the marker, which never appears twice in
a row, and the recovered quote fills `parent_text` where none was stored — a
net gain, since 3.5% of replies have no parent recorded. **This wants fixing in
the scraper too**; patching it downstream only stops it corrupting the labels.

---

## What is not done

- **Serving.** The adapter is not merged and there is no vLLM or batch
  inference path, which is why the latency figure above is what it is.
- **A best-checkpoint callback**, so the reported adapter overfits slightly.
- **`health_claim` needs re-defining, not re-training.** At κ 0.136 with 65 of
  100 rows unresolved, no amount of modelling helps; the definition has to be
  split into something two people can apply.
- **A larger blind set.** 100 rows is enough to find the agreed/adjudicated
  split but not to put intervals on it.
