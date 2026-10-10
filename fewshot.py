"""The hand-labelled examples that anchor the teacher prompt.

Seven real posts from the corpus, annotated by hand against RUBRIC.md.
They are short excerpts of public posts, carried here because the teacher
prompt is not reproducible without them. Each one
was chosen to pin down a rule that the model would otherwise get wrong, and the
`note` says which. Notes are for humans: they are kept out of the prompt, which
sends only the text and the JSON, so the fixed prefix stays small.

`id` is the corpus row id, so every example is traceable.
These texts are also excluded from the sample pools (see sample.py) to keep
them out of training and evaluation.
"""
from __future__ import annotations

EXAMPLES: tuple[dict, ...] = (
    {
        "id": "Inspire_aa8be7e5b2ca1cdd",
        "note": "The rule that matters most: severe clinical facts (EF 20%, very low BP, "
                "painful groin) do not make a post negative. The author twice says he "
                "feels good, so emotion is wellbeing and derived sentiment is positive. "
                "It is also the canonical care_experience case: a cath happened, "
                "clinicians are named, and the cath 'went well' - but nothing says how "
                "the author was treated. A clinical event is not a care experience, and "
                "nor is a clinical outcome, so the field is not_mentioned.",
        "text": "The Cath went well, the Dr. said my EF is 20% but everything else cardiac "
                "related is good. Don't really know what that means but, everyone at the "
                "hospital said they were happy for me. I am to just continue to wear the "
                "life vest and take the meds. Sadly the day after the cath I had to "
                "evacuate my house because of the hurricane. As I write I am at a hotel. I "
                "do however feel good, just my groin is very painful. I don't know if the "
                "5-10% they originally said, has already improved to the 20% the Dr. said "
                "after the cath or it is just a better measurement... My BP has been very "
                "very low, but I still feel real good.",
        "labels": {
            "speech_act": "shares_experience",
            "clinical_stage": "diagnosis_workup",
            "emotion": ["wellbeing"],
            "care_experience": "not_mentioned",
            "subject": "self",
            "health_claim": "no_claim",
            "womens_specific": False,
        },
    },
    {
        "id": "YouTube_1c584ee0572acd8f",
        "note": "A story that ends in a question is asks_question, not shares_experience. "
                "No diagnosis and no event, so presymptomatic_worry. Fear is stated "
                "outright.",
        "text": "I'm 15 and my chest started to hurt like 3 months ago, it lasted for a "
                "week, but yesterday it started to hurt again. If I breathe in deeply it "
                "gets worse, when lying down I can't feel it. I drink a lot of water so I "
                "don't think that's the problem. Does anyone know what could this be, or is "
                "there someone who has the same problem? I'm scared to tell my parents.",
        "labels": {
            "speech_act": "asks_question",
            "clinical_stage": "presymptomatic_worry",
            "emotion": ["fear"],
            "care_experience": "not_mentioned",
            "subject": "self",
            "health_claim": "no_claim",
            "womens_specific": False,
        },
    },
    {
        "id": "YouTube_c672fa175d455b82",
        "note": "Both a dismissal and a good doctor are described, so the priority rule "
                "gives dismissed. Opening thanks do not make it offers_support, because "
                "the post has substantive content of its own. Two emotions, strongest "
                "first. 'it can be treated with bisoprolol' is stated as a fact about the "
                "condition, so general_health_claim rather than personal.",
        "text": "Thanks doctor for this helpful information. I was diagnosed with "
                "Inappropriate sinus tachycardia very recently and I am so happy that I "
                "know with what I'm dealing and that it can be treated with bisoprolol. "
                "Other doctor blamed anxiety and that I was fat when I was 14 and I was "
                "63kg.. now I am 18 I went to a good doctor I he told me right away what "
                "it was.",
        "labels": {
            "speech_act": "shares_experience",
            "clinical_stage": "diagnosis_workup",
            "emotion": ["wellbeing", "gratitude"],
            "care_experience": "dismissed",
            "subject": "self",
            "health_claim": "general_health_claim",
            "womens_specific": False,
        },
    },
    {
        "id": "Inspire_25493604a3850555",
        "note": "health_claim does not judge truth. 'That is caused by decreased cardiac "
                "output' is a correct causal statement and is still "
                "causal_or_cure_claim. No affect is expressed anywhere, so none.",
        "text": "Actually even in the UK, if you have undiagnosed chest pain you should be "
                "checked at A&E. It is DIAGNOSED angina that is treated ongoing at the "
                "doctor's office. That's a big difference. But as you have probably already "
                "seen the doc at this point, that is academic. Yes, fatigue and feeling "
                "\"woozy\" comes along with angina. That is caused by decreased cardiac "
                "output which means you are not getting enough oxygen to your brain and "
                "other tissues.",
        "labels": {
            "speech_act": "gives_advice",
            "clinical_stage": "diagnosis_workup",
            "emotion": ["none"],
            "care_experience": "not_mentioned",
            "subject": "general",
            "health_claim": "causal_or_cure_claim",
            "womens_specific": False,
        },
    },
    {
        "id": "Inspire_0856531827a19ff5",
        "note": "SCAD makes this women-specific. 'some know more (or less) than others' is "
                "a general opinion about doctors with no episode behind it, so "
                "care_experience stays not_mentioned. 'Best wishes to you' is an outward "
                "courtesy, not the author's own affect.",
        "text": "Yes, keep pursuing this. You could have had a SCAD heart attack - a "
                "Spontaneous Coronary Artery Dissection. It happens often to women who are "
                "young, healthy and have no other risk factors. Cardiologists know about "
                "this but some know more (or less) than others. Best wishes to you.",
        "labels": {
            "speech_act": "gives_advice",
            "clinical_stage": "diagnosis_workup",
            "emotion": ["none"],
            "care_experience": "not_mentioned",
            "subject": "general",
            "health_claim": "general_health_claim",
            "womens_specific": True,
        },
    },
    {
        "id": "Heart Sisters_14bb45a5dc2c4735",
        "note": "The dismissals happened to the author's readers, not to the author: "
                "care_experience records the episode, subject records whose it was. "
                "Generalising from those cases is shares_information rather than advice.",
        "text": "So true. And it's not only anxiety that's a great all-purpose "
                "misdiagnosis. I've had readers who, because of lifelong histories of "
                "asthma, for example, have also had their cardiac symptoms dismissed as "
                "\"just asthma\". As I've often said, there are few things in life more "
                "anxiety-provoking than being in the middle of a frickety-frackin' heart "
                "attack!",
        "labels": {
            "speech_act": "shares_information",
            "clinical_stage": "diagnosis_workup",
            "emotion": ["frustration"],
            "care_experience": "dismissed",
            "subject": "general",
            "health_claim": "general_health_claim",
            "womens_specific": False,
        },
    },
    {
        "id": "Inspire_8aa19f2ff2b6216c",
        "note": "Thanks with no content of its own: offers_support, and no cardiac stage "
                "to speak of.",
        "text": "Thanks for your helpful information. Blessings.",
        "labels": {
            "speech_act": "offers_support",
            "clinical_stage": "not_applicable",
            "emotion": ["gratitude"],
            "care_experience": "not_mentioned",
            "subject": "general",
            "health_claim": "no_claim",
            "womens_specific": False,
        },
    },
)

EXAMPLE_IDS: frozenset[str] = frozenset(e["id"] for e in EXAMPLES)
