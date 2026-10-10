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

| Column | Allowed values |
|---|---|
| `speech_act` | `asks_question` / `shares_experience` / `gives_advice` / `offers_support` / `shares_information` |
| `emotion` | `fear` / `frustration` / `grief` / `hope` / `wellbeing` / `gratitude` / `none` |
| `care_experience` | `not_mentioned` / `positive` / `dismissed` / `delayed_diagnosis` / `access_barrier` |
| `subject` | `self` / `family_member` / `general` |
| `health_claim` | `no_claim` / `personal_experience_claim` / `general_health_claim` / `causal_or_cure_claim` |
| `womens_specific` | `true` / `false` |
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
