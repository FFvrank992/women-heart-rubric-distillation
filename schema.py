"""The annotation schema: fields, allowed values, validation, derived sentiment.

Single source of truth. `RUBRIC.md` is the human-facing version of the same
definitions with the boundary cases; the values here are what the teacher
prompt, the student's training targets and the metrics all read.

`sentiment` is deliberately *not* a field. It is derived from `emotion` by
`derive_sentiment`, so the model cannot emit a sentiment that contradicts the
emotions it just listed, and so the downstream sentiment indicator costs no
extra output tokens.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass

# PII placeholders an upstream redaction engine may leave in the text. They
# stand for a redacted span and must not change any label: read "[NAME] told
# me" as "a person told me".
PLACEHOLDERS = ("[URL]", "[EMAIL]", "[EMAIL:ORG]", "[PHONE]", "[USER]",
                "[NAME]", "[ADDRESS]", "[ID]", "[DOB]")


@dataclass(frozen=True)
class Field:
    name: str
    kind: str                       # "single" | "multi" | "bool"
    short: str                      # the definition that goes into the prompt
    values: tuple[str, ...] = ()
    max_values: int = 1             # "multi" only
    tier: int = 1                   # 1 = must work, 3 = first to cut
    priority: tuple[str, ...] = ()  # tie-break, strongest first
    # "active"      - carries a headline finding, metrics reported normally
    # "exploratory" - kept and labelled, but agreement is too low to support a
    #                 claim; reported with its kappa and described as such
    # "dropped"     - no longer labelled or predicted. Kept in this file with
    #                 the reason, because the report has to explain the choice.
    status: str = "active"
    dropped_because: str = ""


_ALL_FIELDS: tuple[Field, ...] = (
    Field(
        name="speech_act",
        kind="single",
        tier=1,
        short=("What the post does. A post that tells a story and also asks the readers "
               "something is asks_question: getting an answer is the point of writing. "
               "Advice aimed at a reader outranks background narrative. offers_support is "
               "sympathy, encouragement, prayers or thanks with no substantive content of "
               "its own."),
        values=("asks_question", "shares_experience", "gives_advice",
                "offers_support", "shares_information"),
        priority=("asks_question", "gives_advice", "shares_information",
                  "shares_experience", "offers_support"),
    ),
    Field(
        name="clinical_stage",
        kind="single",
        tier=3,
        status="dropped",
        dropped_because=(
            "Inter-annotator kappa 0.319 on the 100-row blind set, 43.2% raw "
            "agreement, the second worst of the seven. recovery_longterm acted "
            "as a catch-all for anything in the past: of its 59 disagreeing "
            "cells, 14 were not_applicable against recovery_longterm, 8 "
            "treatment against recovery_longterm and 6 diagnosis_workup "
            "against recovery_longterm. Collapsing to 4 classes made it worse "
            "(0.260) and to a binary barely better (0.348), so the granularity "
            "was not the problem. Designated tier 3 and first to cut from the "
            "start; dropped 2026-10-07 rather than spending a second "
            "annotation round on it."),
        short=("Which phase of the cardiac journey the post is mainly about. "
               "presymptomatic_worry = no diagnosis and no event, worry about risk or "
               "ambiguous sensations; acute_event = a heart attack, cardiac arrest or "
               "emergency, in progress or recounted as an event; diagnosis_workup = tests, "
               "results, getting or awaiting a diagnosis; treatment = procedures, devices "
               "and ongoing medication management; recovery_longterm = rehab and living "
               "with the condition afterwards. treatment vs recovery_longterm: treatment "
               "when the post is about the drugs or the procedure themselves (doses, side "
               "effects, a device), recovery_longterm when it is about the life being lived "
               "around them. For a post spanning the whole timeline, pick the stage it "
               "dwells on, and on a tie the latest stage mentioned."),
        values=("presymptomatic_worry", "acute_event", "diagnosis_workup",
                "treatment", "recovery_longterm", "not_applicable"),
    ),
    Field(
        name="emotion",
        kind="multi",
        max_values=2,
        tier=1,
        short=("Affect the author expresses in this post. It is none UNLESS you can point "
               "at a word or phrase that states a feeling. Describing a frightening event "
               "is not fear ('I had a heart attack' is none); saying so is ('I was "
               "terrified'). Describing a good outcome is not wellbeing ('the test was "
               "clear' is none); saying so is ('I feel great'). A phrase that PERFORMS "
               "the feeling counts even when it does not name it: 'Hang in there' and "
               "'You will make it' are hope. The emotion is the AUTHOR'S OWN: another "
               "person's improvement is not the author's wellbeing, and saying others "
               "gave up hope is not the author's despair. 'Feel reassured' is a worry "
               "resolved, so wellbeing rather than hope. Judge what is written, never "
               "what the events would normally make someone feel. Give a SECOND emotion "
               "only when it has its own clause; where the second is merely implied by "
               "the first, or you are unsure whether it applies, give one emotion only. "
               "wellbeing = reports feeling well, improved, or a worry resolved. gratitude "
               "needs the author thanking or appreciating someone; offering prayers for "
               "someone else is not gratitude. none means no affect at all and must appear "
               "alone, never beside another value."),
        values=("fear", "frustration", "grief", "hope", "wellbeing", "gratitude", "none"),
    ),
    Field(
        name="care_experience",
        kind="single",
        tier=1,
        short=("HOW THE CARE WENT for a specific person whose case the post describes, "
               "whoever that is. Choose anything other than not_mentioned only when you "
               "can point at words showing whether the care was good or bad. Three things "
               "are not_mentioned: no care described at all; clinical events narrated "
               "with no sign of how they went ('I had a cath on Tuesday'); and opinions "
               "about doctors in general rather than a described case ('doctors never "
               "listen to women'). Recommendations, referrals and explanations a "
               "clinician gave are also not_mentioned on their own: what was advised is "
               "not how the advising went. An evaluation of one's own care is enough on "
               "its own, "
               "with no action attached: 'my cardiologist is great' is positive. The "
               "clinical OUTCOME is not the care experience: 'the surgery was a success' "
               "is about the body, not about how the clinicians treated anyone. "
               "access_barrier further requires that the care was NOT OBTAINED: "
               "difficulty, delay or expense on the way to care that was still received "
               "is positive, dismissed or delayed_diagnosis instead. "
               "positive = clinicians listened, acted promptly, were competent or kind; "
               "dismissed = symptoms minimised, disbelieved, or put down to anxiety, stress "
               "or weight without a workup; delayed_diagnosis = the right answer came late, "
               "repeat visits or a misdiagnosis before it was found; access_barrier = care "
               "could not be obtained at all (cost, insurance, waiting lists, distance, no "
               "specialist). A general opinion with no episode behind it (doctors never "
               "listen to women) is not_mentioned, and so is a dismissal the author only "
               "feared or imagined. If several apply: dismissed > delayed_diagnosis > "
               "access_barrier > positive."),
        values=("not_mentioned", "positive", "dismissed",
                "delayed_diagnosis", "access_barrier"),
        priority=("dismissed", "delayed_diagnosis", "access_barrier",
                  "positive", "not_mentioned"),
    ),
    Field(
        name="subject",
        kind="single",
        tier=2,
        short=("Whose health the post is mainly about. self = the author's own; "
               "family_member = a named or identified person close to them; general = no "
               "specific individual, or someone the author only addresses without "
               "describing their case. This is about whose health, not whose feelings: "
               "my husband had a heart attack and I am terrified is family_member with "
               "emotion fear. Advice to a reader that also mentions the author's own case "
               "is self only where their own health is described substantively; a passing "
               "mention in support of the advice stays general."),
        values=("self", "family_member", "general"),
    ),
    Field(
        name="health_claim",
        kind="single",
        tier=2,
        status="exploratory",
        short=("The strongest generalisable health proposition present in the post, whether "
               "the author asserts it, questions it or quotes someone else. no_claim = "
               "none; personal_experience_claim = a statement about the author alone "
               "(magnesium stopped my palpitations); general_health_claim = a statement "
               "presented as holding for other people; causal_or_cure_claim = a generalised "
               "cause, cure, prevention or reversal. no_claim vs personal_experience_claim: "
               "bare history (I had a stent in 2019) is no_claim, and becomes a claim only "
               "where the post links something to an effect (I had a stent and the pain "
               "went). This field does not judge whether the claim is true: correct claims "
               "are labelled the same way. Strongest scope wins."),
        values=("no_claim", "personal_experience_claim",
                "general_health_claim", "causal_or_cure_claim"),
        priority=("causal_or_cure_claim", "general_health_claim",
                  "personal_experience_claim", "no_claim"),
    ),
    Field(
        name="womens_specific",
        kind="bool",
        tier=2,
        short=("true when the content concerns heart health in ways specific to women: "
               "pregnancy or postpartum cardiac problems, menopause, HRT or oestrogen and "
               "cardiac risk, SCAD, Takotsubo, sex differences in symptoms, diagnosis or "
               "treatment, or dismissal attributed to being a woman. The author being a "
               "woman does not by itself make the content women-specific."),
    ),
)

# What the prompts, the sheets and the metrics see. A dropped field stays in
# _ALL_FIELDS with its reason, so the decision is auditable from the source.
FIELDS: tuple[Field, ...] = tuple(f for f in _ALL_FIELDS if f.status != "dropped")
DROPPED: tuple[Field, ...] = tuple(f for f in _ALL_FIELDS if f.status == "dropped")
EXPLORATORY: frozenset[str] = frozenset(
    f.name for f in FIELDS if f.status == "exploratory")

BY_NAME: dict[str, Field] = {f.name: f for f in FIELDS}
NAMES: tuple[str, ...] = tuple(f.name for f in FIELDS)

# What an unparseable answer falls back to: always the least committal value,
# never a rare class, so a parse failure cannot invent a `dismissed` or an
# `acute_event`. Rows that needed a fallback carry parse_ok = False and are
# dropped before training; these values only keep the dataframe dtypes sane.
DEFAULTS: dict = {
    "speech_act": "shares_experience",
    "clinical_stage": "not_applicable",
    "emotion": ["none"],
    "care_experience": "not_mentioned",
    "subject": "general",
    "health_claim": "no_claim",
    "womens_specific": False,
}
assert set(NAMES) <= set(DEFAULTS), "DEFAULTS must cover every active field"

EMOTION_NEGATIVE = frozenset({"fear", "frustration", "grief"})
EMOTION_POSITIVE = frozenset({"hope", "wellbeing", "gratitude"})

# The queue a human reviews for misinformation. The model never decides truth:
# it marks claim-making, which is visible in the text, and this cross picks out
# the posts where a generalised claim is asserted or promoted rather than asked
# about. See RUBRIC.md, "Downstream: the misinformation review queue".
def in_review_queue(annotation: dict) -> bool:
    claim = annotation.get("health_claim")
    if claim == "causal_or_cure_claim":
        return True
    return (claim == "general_health_claim"
            and annotation.get("speech_act") == "gives_advice")


def derive_sentiment(emotions) -> str:
    """The downstream sentiment indicator, computed from `emotion`.

    Documented in RUBRIC.md and applied to the human gold labels too, so the
    comparison against VADER/roberta stays fair.
    """
    found = {e for e in (emotions or []) if e != "none"}
    negative, positive = bool(found & EMOTION_NEGATIVE), bool(found & EMOTION_POSITIVE)
    if negative and positive:
        return "mixed"
    if negative:
        return "negative"
    if positive:
        return "positive"
    return "neutral"


def blank() -> dict:
    """An all-default annotation, used when a reply cannot be parsed at all."""
    return {name: (list(value) if isinstance(value, list) else value)
            for name, value in DEFAULTS.items()}


def validate(obj: dict) -> tuple[dict, list[str]]:
    """Coerce a model answer onto the schema. Returns (annotation, problems).

    Anything unfixable becomes the field's default and is named in `problems`,
    so one bad reply never kills a run and the JSON validity rate stays
    measurable.
    """
    clean: dict = {}
    problems: list[str] = []
    defaults = blank()
    for f in FIELDS:
        raw = obj.get(f.name)
        if raw is None:
            clean[f.name] = defaults[f.name]
            problems.append(f"{f.name}: missing")
            continue

        if f.kind == "bool":
            if isinstance(raw, bool):
                clean[f.name] = raw
            elif str(raw).strip().lower() in ("true", "false"):
                clean[f.name] = str(raw).strip().lower() == "true"
            else:
                clean[f.name] = False
                problems.append(f"{f.name}: not a boolean ({raw!r})")
            continue

        if f.kind == "multi":
            items = raw if isinstance(raw, list) else [raw]
            lowered = [str(v).strip().lower() for v in items]
            unknown = [v for v in lowered if v not in f.values]
            if unknown:
                problems.append(f"{f.name}: unknown {unknown}")
            seen: set[str] = set()
            ordered: list[str] = []
            for v in lowered:                     # keep the model's own order
                if v in f.values and v not in seen:
                    seen.add(v)
                    ordered.append(v)
            if len(ordered) > 1 and "none" in ordered:
                ordered = [v for v in ordered if v != "none"]
                problems.append(f"{f.name}: none combined with other values")
            if len(ordered) > f.max_values:
                problems.append(f"{f.name}: {len(ordered)} values, keeping {f.max_values}")
                ordered = ordered[: f.max_values]
            clean[f.name] = ordered or defaults[f.name]
            continue

        # single
        if isinstance(raw, list):
            candidates = [str(v).strip().lower() for v in raw
                          if str(v).strip().lower() in f.values]
            if not candidates:
                clean[f.name] = defaults[f.name]
                problems.append(f"{f.name}: unknown {raw!r}")
            else:
                order = f.priority or f.values
                clean[f.name] = min(candidates, key=order.index)
                problems.append(f"{f.name}: list given, applied priority -> {clean[f.name]}")
            continue

        value = str(raw).strip().lower()
        if value in f.values:
            clean[f.name] = value
        else:
            clean[f.name] = defaults[f.name]
            problems.append(f"{f.name}: unknown {raw!r}")
    return clean, problems


def json_skeleton() -> str:
    """The exact shape the teacher is asked to return."""
    lines = []
    for f in FIELDS:
        if f.kind == "bool":
            lines.append(f'  "{f.name}": true|false')
        elif f.kind == "multi":
            lines.append(f'  "{f.name}": [1-{f.max_values} of: ' + " | ".join(f.values) + "]")
        else:
            lines.append(f'  "{f.name}": one of: ' + " | ".join(f.values))
    return "{\n" + ",\n".join(lines) + "\n}"


def prompt_field_block() -> str:
    """Field definitions for the prompt."""
    out = []
    for f in FIELDS:
        allowed = "true | false" if f.kind == "bool" else " | ".join(f.values)
        body = textwrap.fill(f.short, width=92,
                             initial_indent="    ", subsequent_indent="    ")
        out.append(f"{f.name} ({allowed})\n{body}")
    return "\n\n".join(out)
