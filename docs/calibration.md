# Calibration session — 61 posts, 88 decisions

Work through this together, both of you looking at the same screen, with
`RUBRIC.md` open. Budget about 66 minutes.

## Why

On the blind 100 you two agreed far less than the definitions assumed:

- `care_experience` — kappa 0.293, 44 of 100 rows disagree
- `emotion` — kappa 0.547, 44 of 100 rows disagree

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
