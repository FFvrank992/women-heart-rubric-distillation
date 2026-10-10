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

| Column | Allowed values |
|---|---|
| `speech_act` | `asks_question` / `shares_experience` / `gives_advice` / `offers_support` / `shares_information` |
| `clinical_stage` | `presymptomatic_worry` / `acute_event` / `diagnosis_workup` / `treatment` / `recovery_longterm` / `not_applicable` |
| `emotion` | `fear` / `frustration` / `grief` / `hope` / `wellbeing` / `gratitude` / `none` |
| `care_experience` | `not_mentioned` / `positive` / `dismissed` / `delayed_diagnosis` / `access_barrier` |
| `subject` | `self` / `family_member` / `general` |
| `health_claim` | `no_claim` / `personal_experience_claim` / `general_health_claim` / `causal_or_cure_claim` |
| `womens_specific` | `true` / `false` |
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
