# Annotation rubric — women's heart health post annotator

**Version 1.4 · 2026-10-08 · status: calibrated; gold_blind built**

This document defines every label. It is used three ways, and they must stay in
step: it is the instruction sheet for the human annotators, the source of the
field definitions in the teacher prompt, and an appendix to the report.

`schema.py` holds the machine-readable version. **If the two ever disagree,
`schema.py` is authoritative for field names and allowed values, and this file
is authoritative for how to decide between them.** Changing a value here means
re-labelling, so changes go through the change log at the bottom.

---

## 0. Scope and global rules

**The unit of annotation is one post or comment**, as it appears in
the cleaned corpus: 53,864 rows of public posts and comments across six
platforms (a video platform at 65% of rows, then two patient forums at 23% and
7%, and three smaller sources), 2016-01 to 2026-09.

1. **Label only from the text in front of you.** For a reply (31.8% of rows) the
   first 200 characters of the parent are shown. The parent is there to
   disambiguate — *"Yes, same here"* is unlabelable alone — but **you annotate
   the reply, never the parent**.
2. **Label what is written, not what you infer.** This is the rule that is
   broken most often. If a post recounts a heart attack in a flat, factual
   register, the emotion is `none`, however frightening the event was. Do not
   supply the feelings you think the author must have had.
3. **No outside medical knowledge is needed for any field.** If you find
   yourself wanting to look up whether a claim is true, you are on the wrong
   field — see `health_claim`, which deliberately does not ask about truth. If a
   field genuinely cannot be decided without clinical expertise, the rubric is
   wrong: flag it rather than guessing.
4. **PII placeholders** left by the redaction engine — `[URL]` `[EMAIL]`
   `[EMAIL:ORG]` `[PHONE]` `[USER]` `[NAME]` `[ADDRESS]` `[ID]` `[DOB]` — stand
   for a removed span and **never change a label**. Read *"[NAME] told me"* as
   *"a person told me"*.
5. **Text is truncated at 1,200 characters** (≈ 99% of rows are shorter; mean
   cost 88 tokens). Judge the truncated text, which is also what the model will
   see at training and inference time.
6. **Human annotators may write `UNCLEAR`** in any cell. Those cells are
   excluded from the metrics and reported as a coverage figure. The model has no
   `UNCLEAR` option: it must commit, and a `parse_ok = False` row is dropped
   from training instead.
7. **Every field is independent.** Do not let one label drive another except
   where a priority rule below says so explicitly.
8. **Posts with no cardiac content still get labelled.** Mayo Clinic Connect is
   a general health forum and Heart Sisters covers wider ground, so the corpus
   carries threads about osteoporosis, neuropathy, chemotherapy, hysterectomy
   and so on. These are real rows that the model will meet on all 53,864, so
   they are annotated rather than skipped:
   - `clinical_stage` = **`not_applicable`** is the off-topic marker. There is
     no cardiac journey, so there is no stage.
   - `womens_specific` = **`false`** unless the post links a women-specific
     matter to *heart* health. A hysterectomy thread with no cardiac content is
     women's health but not women's *heart* health.
   - Every other field still applies normally: a post can describe a care
     experience, express emotion and make claims without mentioning the heart.

   Filtering `clinical_stage == "not_applicable"` recovers this slice later, so
   no separate off-topic field is needed.

---

## 1. `speech_act` — what the post does  · Tier 1

The author's dominant communicative purpose.

| Value | Means |
|---|---|
| `asks_question` | Seeks information, advice or reassurance from the readers |
| `shares_experience` | Recounts health events or history, their own or a specific person's |
| `gives_advice` | Tells someone what to do, or recommends a treatment, test or clinician |
| `offers_support` | Sympathy, encouragement, prayers or thanks **with no substantive content of its own** |
| `shares_information` | Conveys general or factual content not framed as the author's own experience |

**Priority when several apply:**
`asks_question` > `gives_advice` > `shares_information` > `shares_experience` > `offers_support`

### Boundary cases

1. **Story that ends in a question → `asks_question`.** 25.6% of the corpus
   contains a question mark, and most symptom narratives close with *"is this
   normal?"*. Getting an answer is why the post was written, so the question
   wins. *(few-shot example 2)*
2. **Rhetorical question → not a question.** *"Why do doctors never listen?!"*
   expects no answer. Judge the rest of the post.
3. **Thanks plus content → `shares_experience`** (or whatever the content is).
   `offers_support` is only for posts whose *whole* content is courtesy.
   *(few-shot 3 opens with "Thanks doctor" and is still `shares_experience`;
   few-shot 7 is pure courtesy and is `offers_support`.)*
4. **Advice appended to a story → `gives_advice`**, when the advice is aimed at
   a reader rather than being part of the narrative.
5. **Low-content noise** (*"Great video!"*, emoji only) → `offers_support`.
   Rows the upstream cleaner marked `is_fragment` are already out of the pool,
   so this should be rare.

---

## 2. `clinical_stage` — which phase of the journey  · Tier 3

What phase of the cardiac journey the post is **mainly** about.

| Value | Means |
|---|---|
| `presymptomatic_worry` | No diagnosis and no event: worry about risk, prevention, or ambiguous sensations |
| `acute_event` | A heart attack, cardiac arrest or emergency — in progress or recounted as an event |
| `diagnosis_workup` | Tests, results, getting or awaiting a diagnosis, referrals |
| `treatment` | Procedures, devices, surgery, and ongoing medication management |
| `recovery_longterm` | Rehab, and living with the condition afterwards |
| `not_applicable` | No identifiable stage |

**Tie-break for timeline-spanning posts:** pick the stage the post *dwells on*;
if attention is genuinely even, pick the **latest** stage mentioned.

### Boundary cases

1. **Full-timeline narrative** (*"symptoms → heart attack → CABG → six years
   later"*) → apply the tie-break above. Do not label by the most dramatic
   event.
2. **`treatment` vs `recovery_longterm`** — the commonest confusion on this
   field (4 of its 8 split rows in the pilot). Use `treatment` when the post is
   about **the drugs or the procedure themselves** — doses, side effects, a
   device, an operation. Use `recovery_longterm` when it is about **the life
   being lived around them** — rehab, work, exercise, how things are years on.
   A chronic condition managed with daily medication is genuinely both, so the
   test is what the post is *talking about*, not how long ago the event was.
   Medication side effects are therefore `treatment`, even a decade later.
3. **Awaiting results → `diagnosis_workup`**, even when the post is frightened
   about what the result might be.
4. **A diagnostic catheterisation → `diagnosis_workup`**, even though it is a
   procedure. *(few-shot 1)*
5. **Worry held by someone who already has a diagnosis** is not
   `presymptomatic_worry`. Use the stage the worry is about.

> This is the field most likely to fail the pilot. It is Tier 3 for that
> reason: if its κ comes in under 0.60, drop it rather than redefine it.

---

## 3. `emotion` — affect the author expresses  · Tier 1

Up to **two** values, strongest first.

| Value | Means |
|---|---|
| `fear` | Fear, anxiety, panic, dread about health |
| `frustration` | Anger, exasperation, feeling wronged — often at clinicians or the system |
| `grief` | Loss, mourning, sadness over a death or an irreversible loss |
| `hope` | Forward-looking optimism, encouragement about what is to come |
| `wellbeing` | Reports feeling well, improved, or a worry resolved |
| `gratitude` | Thanks or appreciation expressed by the author |
| `none` | No affect expressed; factual or neutral register |

**`none` is exclusive** — never combine it with another value.

**When to give a second emotion.** Give one only when the second **has its own
clause**. Where it is merely implied by the first, or you are unsure whether it
applies, **give one emotion only.** *(The "own clause" test came out of the
calibration session: it was the thing that actually settled these rows.)*

**A phrase that performs the feeling counts, even without naming it.** *"Hang in
there"* and *"You will make it"* are `hope`. The test is that you can point at
words carrying the feeling — not that the feeling is named.

**The emotion is the author's own.** Another person's improvement is not the
author's `wellbeing`; saying that *other* people gave up hope is not the
author's despair. Where a post reports someone else's state, ask what the
*author* expresses about it.

**`wellbeing` vs `hope`.** *"I feel reassured"* is a worry resolved →
`wellbeing`. `hope` is forward-looking: about what is still to come.

**Order is not used.** The values are compared as a set, so `fear|hope` and
`hope|fear` are the same label. Write them in whatever order is natural; if a
primary emotion is ever wanted, it needs its own field.

> Added in v1.1. On the pilot, 21 of 100 rows split the teacher's three votes on
> this field — the worst of the seven — and the split was almost never about
> *which* emotion: it was `("fear","frustration")` against `("fear",)`,
> `("hope","wellbeing")` against `("hope",)`. The old wording, "the two most
> strongly expressed", gave no test for when to stop at one. This rule is that
> test.

### Boundary cases

1. **Flat clinical recounting → `none`.** The single most important rule in this
   document. *(few-shot 1 and 4.)*
2. **Offering prayers for someone else is not `gratitude`.** It is an act
   directed outward, not the author's own affect. *"Praying for you"* alone →
   `none`. *"Thank you for sharing this"* → `gratitude`. *(few-shot 5: "Best
   wishes to you" → `none`.)*
3. **Punctuation and capitals are not evidence.** Exclamation marks or caps
   without affective wording do not establish an emotion.
4. **Mixed affect is normal and expected.** *"I'm scared but hopeful"* →
   `["fear", "hope"]`, which derives to `sentiment = mixed`.
5. **More than two apply** → keep the two most strongly expressed. *(few-shot 3
   keeps `wellbeing` and `gratitude`.)*
6. **Describing anxiety as a symptom or as someone else's diagnosis is not
   `fear`.** *"They blamed anxiety"* is about a misdiagnosis, not about the
   author being afraid.

---

## 4. `care_experience` — a described episode of care  · Tier 1

**The focal field of this project.** **How the care went** for a specific person
whose case the post describes — whoever that is.

The test is: **can you point at words showing whether the care was good or
bad?** If not, it is `not_mentioned`, however much medical detail the post
carries.

| Value | Means |
|---|---|
| `not_mentioned` | Nothing in the post says how the care went |
| `positive` | The care is conveyed as good: clinicians listened, believed the author, acted promptly, were competent or kind |
| `dismissed` | Symptoms minimised, disbelieved, or put down to anxiety, stress or weight without a workup |
| `delayed_diagnosis` | The right answer came late: repeat visits, sent home, or a misdiagnosis before it was found |
| `access_barrier` | Care could not be obtained at all — cost, insurance, waiting lists, distance, no specialist |

**Priority when several apply:**
`dismissed` > `delayed_diagnosis` > `access_barrier` > `positive` > `not_mentioned`

*Rationale for that order: dismissal is the phenomenon the project exists to
measure and is the hardest to retrieve any other way; within one post a negative
episode carries more information than a positive one.*

### Boundary cases

1. **`dismissed` vs `delayed_diagnosis`.** `dismissed` is about *being
   minimised or not believed*; `delayed_diagnosis` is about *time and
   sequence*. *"Two doctors sent me home before the third found the
   blockage"*, with no suggestion of disbelief → `delayed_diagnosis`. *"They
   told me it was just anxiety"* → `dismissed`. Both present → `dismissed`.
2. **A general opinion is not an episode.** *"Doctors never listen to women"*
   → `not_mentioned`. The field requires a described event, or it degenerates
   into an opinion detector. *(few-shot 5: "some know more than others" →
   `not_mentioned`.)*
3. **A feared or counterfactual dismissal → `not_mentioned`.** *"My ECG showed
   the LBBB, otherwise I think they would say it's all in my head"* describes a
   dismissal that did **not** happen.
4. **Third-party episodes count.** *"I've had readers whose cardiac symptoms
   were dismissed as just asthma"* → `dismissed`, with `subject = general`.
   The two fields record different things. *(few-shot 6.)*
5. **Thanking a YouTube doctor or blogger is not a care experience** →
   `not_mentioned`; that is `gratitude` under `emotion`.
6. **Cost and insurance complaints → `access_barrier`.** Note that geography is
   unknown for ~91% of rows, so this value mixes health systems; the report must
   say so rather than reading it as a US finding.
7. **A clinical event with no sign of how it went → `not_mentioned`.** Naming a
   doctor, a test, a drug or a hospital says nothing about how the care went.
   *"I had a cath on Tuesday"*, *"they started me on metoprolol"*, *"I saw the
   cardiologist last week"* → `not_mentioned`. A post can be dense with medical
   detail and still say nothing about the experience of being cared for.
   *(few-shot 1.)*
8. **What a clinician advised is not how the advising went.** Recommendations,
   referrals and explanations reported on their own → `not_mentioned`. *"He
   recommended I see an electrophysiologist"*, *"she explained what the EF
   number meant"* say what was said, not whether the author was well treated.
   Nine of the calibration session's sixty-one rules were some form of this
   one, which makes it the commonest trap in the field.
9. **An evaluation needs no action behind it.** *"My cardiologist is great"* and
   *"the nurses were wonderful"* → `positive`. They are about the author's own
   care and they say how it went, which is all this field asks for. Contrast
   with case 2: *"doctors are wonderful these days"* is a general opinion, so
   `not_mentioned`. The line is **specific case vs general opinion**, never
   action vs feeling.
10. **The clinical outcome is not the care experience.** *"The surgery was a
   success"*, *"the cath went well"*, *"my numbers are good now"* describe what
   happened to the body, not how the clinicians treated anyone →
   `not_mentioned`. This distinction matters a lot in this corpus, where
   outcome reports are everywhere. *(few-shot 1 again: the cath "went well" and
   the field is still `not_mentioned`.)*

---

## 5. `subject` — whose health  · Tier 2

| Value | Means |
|---|---|
| `self` | The author's own health |
| `family_member` | A specific identified person close to them — spouse, parent, child, sibling, close friend |
| `general` | No specific individual, or a person the author only addresses without describing their case |

### Boundary cases

1. **Whose health, not whose feelings.** *"My husband had a heart attack and
   I'm terrified"* → `family_member`, with `emotion = ["fear"]`. These two
   fields deliberately describe different people.
2. **Both self and a relative** → `self` if the author's own health is discussed
   substantively; otherwise `family_member`.
3. **Advice to a reader about the reader's case** → `general`. *(few-shot 4
   and 5.)*
   **Where the advice also mentions the author's own case** — the commonest
   confusion on this field (3 of its 6 split rows in the pilot) — use `self`
   only when their own health is **described substantively**. A passing mention
   offered in support of the advice (*"I take the same drug, so get it
   checked"*) stays `general`.
4. **Pure courtesy with no health content** → `general`. *(few-shot 7.)*

---

## 6. `health_claim` — generalisable propositions present  · Tier 2

The **strongest** generalisable health proposition present in the post, whether
the author asserts it, questions it, or quotes somebody else.

| Value | Means |
|---|---|
| `no_claim` | No generalisable health proposition |
| `personal_experience_claim` | A statement about the author alone — *"magnesium stopped my palpitations"* |
| `general_health_claim` | A statement presented as holding for other people |
| `causal_or_cure_claim` | A generalised cause, cure, prevention or reversal |

**Priority:** `causal_or_cure_claim` > `general_health_claim` >
`personal_experience_claim` > `no_claim`

### This field does not judge truth

A correct claim gets the same label as a false one. *"Fatigue is caused by
decreased cardiac output"* is medically correct and is still
`causal_or_cure_claim` *(few-shot 4)*. Truth is adjudicated downstream, by a
human, on the review queue below. **The model never decides whether a claim is
true, and the report must not say it does.**

### Boundary cases

1. **Questions embedding a claim still count.** *"Is it true that statins cause
   dementia?"* → `causal_or_cure_claim`, with `speech_act = asks_question`.
   Crossing the two fields separates *asking about* a claim from *promoting*
   it, which is more than either field gives alone.
2. **Attributed claims count.** *"My cardiologist said X"* carries X regardless
   of who is credited.
3. **`no_claim` vs `personal_experience_claim`** — the commonest confusion on
   this field (3 of its 6 split rows in the pilot). **Bare history is
   `no_claim`.** It becomes a claim only where the post **links something to an
   effect**. *"I got a stent in 2019"* → `no_claim`. *"I got a stent and the
   pain went"* → `personal_experience_claim`. A long illness narrative with no
   cause-and-effect anywhere in it is `no_claim`, however detailed.
4. **A statement about a condition rather than about oneself is general.** *"it
   can be treated with bisoprolol"*, written inside a personal story, is still
   `general_health_claim`. *(few-shot 3.)*

### Downstream: the misinformation review queue

```
review = health_claim == "causal_or_cure_claim"
      or (health_claim == "general_health_claim" and speech_act == "gives_advice")
```

Posts matching this go to a human for verification. Expected volume: roughly
2–4% of the corpus, which is reviewable. The project reports **"the model
surfaced N candidate claims, of which humans confirmed M"** — a precision
figure on an enriched slice, never an F1 for misinformation detection, because
at ~0.3% prevalence by keyword there are too few positives for one to mean
anything.

---

## 7. `womens_specific` — women-specific content  · Tier 2

`true` when the content concerns heart health in ways specific to women:

- pregnancy or postpartum cardiac problems
- menopause, HRT or oestrogen in relation to cardiac risk
- SCAD, FMD, Takotsubo
- sex differences in symptoms, diagnosis or treatment
- dismissal the author attributes to being a woman

### Boundary cases

1. **The author being a woman does not make the content women-specific.** A
   woman posting about generic AFib → `false`. We do **not** infer anyone's
   gender: the field is about the subject matter, which is visible in the text.
2. *"Women's symptoms are different from men's"* → `true`.
3. *"I was dismissed because I'm a woman"* → `true`, with
   `care_experience = dismissed`.
4. **Incidental mention is not enough.** *"After menopause I started walking
   more"*, with no cardiac link, → `false`.
5. SCAD or Takotsubo discussed at all → `true`. *(few-shot 5.)*

---

## 8. Derived: `sentiment`

Not a model output. Computed from `emotion` by `schema.derive_sentiment`, and
applied to the **human gold labels by the same rule**, so the comparison against
VADER and roberta stays fair.

```python
negative = {fear, frustration, grief}
positive = {hope, wellbeing, gratitude}
both  -> "mixed"      only negative -> "negative"
only positive -> "positive"    neither (none) -> "neutral"
```

Deriving rather than asking means the model cannot return
`sentiment = positive` next to `emotion = ["grief"]`, and the downstream
sentiment indicator costs no extra output tokens.

**Why a domain rule is needed at all:** general-purpose sentiment models read
few-shot example 1 — EF 20%, very low blood pressure, a painful groin, a
hurricane evacuation — as strongly negative. Under this rubric it is `positive`,
because the author twice says he feels good. That gap is the measurable reason
this project fine-tunes rather than prompting an off-the-shelf classifier.

---

## 9. Reliability — what was measured, and what each figure is worth

Four figures were taken over the same 100-row blind set. They are not
interchangeable, and the order below is the order of how much they are worth.

The short version: **the three cheap figures all passed, and the one real
figure failed on three fields.** That is the finding, not a footnote.

### Gate 1 — inter-annotator κ (two people) · **run 2026-10-07**

Two annotators labelled the same 100 rows from this rubric alone, in different
row orders, compared by post id, with no discussion while they worked. The
brief that protects that independence is `docs/annotator-b-brief.md`; its key
rule is that a borderline case is never asked about, because once you ask and
are told, the number stops measuring the instructions and starts measuring the
conversation.

| Field | κ | Tier | Verdict |
|---|---|---|---|
| `subject` | 0.750 | 2 | marginal — report as a ceiling |
| `womens_specific` | 0.709 | 2 | marginal — report as a ceiling |
| `speech_act` | 0.646 | 1 | marginal — report as a ceiling |
| `emotion` | 0.547 | 1 | **below floor** |
| `clinical_stage` | 0.319 | 3 | **below floor — dropped** |
| `care_experience` | 0.293 | 1 | **below floor** |
| `health_claim` | 0.136 | 2 | **below floor — exploratory only** |

| κ | Action |
|---|---|
| ≥ 0.75 | Good; proceed |
| 0.60 – 0.75 | Proceed, and report the figure as a ceiling on that field's metrics |
| < 0.60 | **Stop.** Rewrite the definition and re-pilot, or drop the field |

Cut order if fields must go: `clinical_stage` (Tier 3) first, then Tier 2
(`subject`, `health_claim`, `womens_specific`). The Tier 1 fields
(`speech_act`, `care_experience`, `emotion`) carry the project's findings — if
one of those fails, the task design needs revisiting, not the field.

**What was done about it.** The disagreements were read before being acted on,
and they were *systematic*: each annotator had applied a consistent reading the
other did not share. That is an underspecified definition, not a careless
annotator, so the definitions were fixed rather than the people. 61 of the
disputed rows were worked through together in one 66-minute session
(`docs/calibration.md`), grouped by which pair of labels clashed, and every row
records **the rule that settled it**, worded so that someone holding only this
rubric reaches the same answer. Those 88 decisions are what versions 1.3 and
1.4 of this document were written from. `clinical_stage` was dropped outright;
`health_claim` is kept and labelled but marked exploratory, because 65 of its
100 rows could not be settled even with the rubric open.

**The principle that came out of it:** a row two people cannot settle with the
rubric in front of them is a row the model should not be scored on either.
Hence `UNCLEAR`, and hence `gold_blind_sources.csv`, which records for every
cell whether it was `agreed`, `adjudicated` or left `unresolved`. Metrics are
reported split by that provenance and the halves are never pooled.

### Gate 2 — teacher self-disagreement · **passed 2026-10-06**

Every row is annotated three times at temperature 0.6. Where the model cannot
reproduce its own label across three samples, the *definition* is ambiguous,
not the model. This needs no human labels at all, which is what makes it worth
running first: it is free, and it catches the crudest problems before anyone
spends an hour and a half labelling.

Threshold: a field that splits the vote on **more than 25% of rows** needs its
definition rewritten before hand labelling.

Result on the 100-row pilot (`agreement.py --teacher-self --pool pilot`):

| Field | Unanimous | No majority | Tier |
|---|---|---|---|
| `womens_specific` | 99% | 1% | 2 |
| `care_experience` | 98% | 2% | 1 |
| `speech_act` | 96% | 4% | 1 |
| `subject` | 94% | 6% | 2 |
| `health_claim` | 94% | 6% | 2 |
| `clinical_stage` | 92% | 8% | 3 |
| `emotion` | 79% | **21%** | 1 |

No field crossed 25%, so no field was dropped, and the four commonest
confusions were traced to missing rules which were added in v1.1.

**Read this table against Gate 1 before trusting it anywhere else.** It passed
`care_experience` at 98% unanimous and `health_claim` at 94%; the two humans
then returned κ 0.293 and 0.136 on those same fields. A model agreeing with
itself at temperature 0.6 is evidence that a definition is *stated* clearly
enough to be followed consistently — not that it is the definition two readers
would independently arrive at. It is a useful cheap filter and a poor proxy,
and it should never be presented as a substitute for a second reader.

Three values are too thin to carry a per-class F1 and are reported as counts
only: `access_barrier` (2 of 100), `delayed_diagnosis` (4), `emotion: grief` (1).

### Gate 3 — intra-annotator κ (test–retest)

The same 100 posts labelled by hand **twice by the same person, at least three
days apart, in a different order, without looking at the first pass**. Cohen's
κ between the two passes, against the same thresholds as Gate 1.

**This figure is an upper bound on reliability and must be quoted as one**, and
never without the Gate 1 figure beside it. One annotator shares their own
reading of every edge case, so it shows the definitions are *stable*, not that
they are *shared* — which is precisely the gap Gate 1 found. Budget 2 × 1.5
hours.

### Gate 4 — human–model agreement

κ between the human labels and the teacher's majority vote. Convergent
validity: two independent judgements landing in the same place is evidence the
labels track something beyond one reader. The teacher's accuracy here also caps
what the student model can learn.

---
## 10. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-10-05 | First version, awaiting pilot. Replaces an earlier draft whose `sentiment` field was collinear with `emotion` (now derived), whose `post_type` mixed purpose with timeline (now split into `speech_act` and `clinical_stage`), and whose `misinfo_risk` required clinical knowledge the annotators do not have (now `health_claim`, which records claim-making only). |
| 1.4 | 2026-10-08 | Written from the calibration session's own output: all 88 disputed decisions were settled and all 61 rows carry the rule that settled them. **v1.3's `care_experience` correction held** — seven of the rules restate "a clinical event or outcome is not a care experience" independently, which is the test working. Added: what a clinician *advised* is not how the advising went (nine rules, the commonest trap); a phrase that *performs* a feeling counts without naming it (*"Hang in there"* is `hope`, ten rules); the emotion is the author's own, not a third party's (five rules); a second emotion needs its own clause (three rules); `wellbeing` is a resolved worry where `hope` is forward-looking. Also recorded that `emotion` order is never consumed, so it is compared as a set — the session produced both `hope\|gratitude` and `gratitude\|hope` for the same judgement. No value changed. |
| 1.3 | 2026-10-08 | **`care_experience` corrected — v1.2 contradicted itself** and the contradiction was found mid-calibration, by the annotator, on the largest disagreement block (`positive` / `not_mentioned`, 9 rows). v1.2's "point at what a clinician DID or FAILED TO DO" made every treatment narrative into a care experience, which boundary case 7 flatly denied, and it wrongly sent *"my cardiologist is great"* — an explicit evaluation of one's own care — to `not_mentioned`. The thing to point at is **how the care went**, not whether an action occurred; the real line is **specific case vs general opinion**, which boundary case 2 always had right. Cases 7, 8 and 9 now cover clinical events without valence, evaluations without actions, and clinical outcomes. No value changed. `clinical_stage` dropped (κ 0.319) and `health_claim` marked exploratory (κ 0.136) after the inter-annotator gate; see `schema.py` for the reasons. |
| 1.2 | 2026-10-07 | §0 rule 8 added: posts with no cardiac content are labelled, not skipped, and `clinical_stage = not_applicable` is the marker that recovers them. Prompted by a pilot row about hysterectomy recovery on Mayo Clinic Connect, which the rubric gave no way to handle. No value changed. Two preprocessing bugs were fixed at the same time (quoted parents leaking into Mayo text; a name redactor that was damaging ordinary words) — see README.md. |
| 1.1 | 2026-10-06 | Four decision rules added, each from a confusion the pilot's three-vote split exposed. No field was dropped and **no allowed value changed**, so the value space is unaffected; borderline rows may now resolve differently, which is the point. §9 rewritten for a single annotator. <br>• `emotion`: a test for when to give a second emotion (21% of rows split, almost all one-vs-two rather than which). <br>• `clinical_stage`: `treatment` vs `recovery_longterm` turns on what the post discusses, not how long ago the event was (4 of 8 split rows). <br>• `health_claim`: bare history is `no_claim`; a claim needs something linked to an effect (3 of 6). <br>• `subject`: advice mentioning the author's own case is `self` only where that health is described substantively (3 of 6). |
