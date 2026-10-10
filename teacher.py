"""Teacher annotations for the sampled pools.

    python teacher.py --provider none                  # build prompts, send nothing
    python teacher.py --provider none --calls 5        # also dump 5 full calls
    python teacher.py --provider gemini                # the real run
    python teacher.py --provider gemini --limit 20     # smoke test on 20 rows

`--provider none` is the review gate: it writes the exact system instruction and
a sample of the exact user messages to `prompts/`, reports what would leave the
machine, and sends nothing. Nothing should be sent to an API until that output
has been read.

The whole rubric and all seven examples live in the *system instruction*, which
is byte-identical on every call, so the per-row cost is only the document
itself (~90 tokens). Answers are cached by prompt hash and the cache is saved
after each one, so a stopped run resumes instead of paying twice, and a re-run
gives the same labels.

Each row is annotated `--votes` times at a non-zero temperature. Per field the
majority wins and the agreement is recorded, so rows the teacher is unsure
about can be dropped or sent to a human.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd

import config
import schema
from fewshot import EXAMPLES

# Per million tokens, as of 2026-10. Re-check before a paid run: the cheapest
# option, gemini-2.5-flash-lite, retires 2026-10-16.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}

ROLE = """\
You annotate public online posts about women's heart health (patient forums, \
YouTube comments, blogs) against a fixed schema. The posts are de-identified \
public content.

Return one JSON object and nothing else, with exactly these keys:"""

GLOBAL_RULES = """\
Rules that override any instinct to the contrary:
  1. Label only the post given to you. For a reply you also get the opening of
     the parent post: use it to resolve what the reply refers to, but annotate
     the reply.
  2. Label what is written, not what you infer. A post that recounts a severe
     event in a flat, factual register has emotion "none", however frightening
     the event was. Do not supply feelings the author did not express.
  3. Placeholders [URL] [EMAIL] [EMAIL:ORG] [PHONE] [USER] [NAME] [ADDRESS]
     [ID] [DOB] mark spans that were redacted. They never change a label: read
     "[NAME] told me" as "a person told me".
  4. No medical knowledge is needed. In particular health_claim records whether
     a generalisable claim is present, never whether it is true.
  5. Every field is independent except where a priority rule says otherwise.
  6. Choose only from the listed values, and always commit to one."""


def build_system() -> str:
    """The fixed prefix: role, output shape, field definitions, rules, examples."""
    blocks = [ROLE, schema.json_skeleton(), "FIELDS", schema.prompt_field_block(),
              GLOBAL_RULES, "EXAMPLES"]
    for i, example in enumerate(EXAMPLES, start=1):
        labels = json.dumps(example["labels"], ensure_ascii=False)
        blocks.append(f"Example {i}\nPost: {example['text']}\nAnswer: {labels}")
    return "\n\n".join(blocks)


# The delivered corpus turned out to carry no redaction placeholders at all
# (zero [NAME]/[URL]/... in 53,864 rows, although `n_placeholders` is non-zero
# on 14,419 of them), and real forum handles survive in salutations: "HI
# LoriBeth,", "Thank you TinmanT.". The person being addressed is the
# de-identification risk worth removing here, and a salutation is a precise
# place to remove it: a general Title-Case sweep would also eat "Mayo Clinic"
# and "Heart Sisters" and cost label quality.
#
# Clinician and public-figure names in the body ("Dr. Norman Shumway") are left
# alone: they are not the subject's personal data and removing them would take
# clinical meaning with them.
#
# This runs inside build_user, so the teacher, the training data and inference
# all see byte-identical text. See the note in README.md.
# The salutation word is matched case-insensitively ("HI LoriBeth" is common);
# the name itself still has to be capitalised.
SALUTATION_RX = re.compile(
    r"\b((?i:hi|hello|hey|dear|thanks|thank you|welcome|bless you|good luck))"
    r"([,!]?\s+)([A-Z][A-Za-z]{1,19}\.?)")

# Minimum lower-case occurrences in the corpus for a token to count as an
# ordinary English word rather than a handle. Handles are near-unique; "thank"
# appears thousands of times.
WORD_FLOOR = 5


@functools.lru_cache(maxsize=1)
def common_words() -> frozenset[str]:
    """Tokens the corpus uses in lower case, i.e. words rather than names.

    Cached for the process: `prepare` runs over every row of the corpus when
    ids have to be remapped, and re-reading the word list per token would make
    that unusable.

    The first version of this used a hand-written stoplist, which is
    whack-a-mole: it replaced "Thank" 93 times, "Welcome" 41, "Please" 29,
    "Sorry" 16 and - worst - "SCAD" 5 times, which is a clinical term this
    project's `womens_specific` field depends on. 881 replacements corpus-wide
    and only a small minority were actually handles.

    The corpus is a better discriminator than any list we could write, and it
    costs one pass, cached to disk.
    """
    cache = config.DATA / "corpus_lowercase_words.json"
    if cache.exists():
        return frozenset(json.loads(cache.read_text(encoding="utf-8")))

    config.require_corpus()
    import collections

    counts: collections.Counter = collections.Counter()
    for chunk in pd.read_csv(config.CORPUS, usecols=["text"], chunksize=20000):
        for text in chunk["text"].dropna().astype(str):
            counts.update(re.findall(r"\b[a-z][a-z'-]{1,19}\b", text))
    words = sorted(word for word, n in counts.items() if n >= WORD_FLOOR)
    config.DATA.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(words), encoding="utf-8")
    print(f"built {cache.name}: {len(words):,} words seen >= {WORD_FLOOR} times")
    return frozenset(words)


def looks_like_a_name(word: str) -> bool:
    """A capitalised token that the corpus never uses as a word."""
    bare = word.rstrip(".")
    if len(bare) < 3 or bare.isupper():      # "SCAD", "HRT", "PCOS", "ICD"
        return False
    return bare.lower() not in common_words()


def deidentify(text: str) -> str:
    """Replace a name addressed in a salutation with [NAME]. Nothing else."""
    def swap(match: re.Match) -> str:
        if not looks_like_a_name(match.group(3)):
            return match.group(0)
        return f"{match.group(1)}{match.group(2)}[NAME]"
    return SALUTATION_RX.sub(swap, text)


# Mayo Clinic Connect replies arrive with the quoted parent comment prepended
# and the site's own "Jump to this post" link left in the middle of it, so the
# stored text is parent + UI chrome + reply run together. 642 rows, 20.7% of
# every Mayo row in the corpus. The marker never appears twice in a row, and
# everything before it is the quote, so the split is exact.
#
# This is a scraper bug, patched here because it corrupts the labels: the
# quoted parent routinely contains questions, which drags `speech_act` towards
# asks_question when the reply itself is doing something else. See the note in
# README.md - it wants fixing upstream too.
MAYO_QUOTE_MARKER = "Jump to this post"


def split_quoted_parent(text: str) -> tuple[str, str]:
    """(the actual post, the quoted parent) for a row that has both run together."""
    if MAYO_QUOTE_MARKER not in text:
        return text, ""
    quoted, _, reply = text.partition(MAYO_QUOTE_MARKER)
    reply = reply.strip()
    if not reply:                       # marker at the very end: nothing to split
        return text, ""
    return reply, quoted.strip()


def prepare(row) -> tuple[str, str]:
    """The post and its parent context, exactly as every stage must see them.

    One function, because the teacher prompt, the annotation sheets, the
    training data and inference all have to agree byte for byte. Changing
    anything here means re-labelling.
    """
    text = " ".join(str(row.get("text") or "").split())
    post, recovered = split_quoted_parent(text)

    parent = " ".join(str(row.get("parent_text") or "").split())
    if parent in ("", "nan"):
        # 3.5% of replies have no parent_text stored, and for the Mayo rows the
        # parent is sitting in the text we just split off.
        parent = recovered
    is_reply = str(row.get("is_reply", "")).strip().lower() in ("true", "1")
    if not is_reply and not recovered:
        parent = ""

    return (deidentify(post)[: config.DOC_CHARS],
            deidentify(parent)[: config.PARENT_CHARS])


def student_system() -> str:
    """The short prompt the fine-tuned model is trained and served with.

    The teacher gets `build_system()`: the whole rubric and seven worked
    examples, ~2,400 tokens, because in-context learning is the only way it can
    know the task. The student learns the task in its weights, so repeating the
    rubric on every example would only pay ~2,400 tokens of activation memory
    and inference cost for something it already knows - and at seq_len 1024 the
    prompt alone overruns the window, masking every answer token and producing
    a nan loss.

    The value lists stay, at ~180 tokens, because they are cheap and keep the
    output well formed from the first step. Training and inference must use
    this byte for byte.
    """
    return ("Annotate this post about women's heart health. "
            "Reply with one JSON object and nothing else:\n\n"
            + schema.json_skeleton())


def build_user(row: pd.Series) -> str:
    """The variable part: one document, truncated the same way at training time."""
    post, parent = prepare(row)
    lines = [f"Platform: {row['platform']}  Type: {row['type']}"]
    if parent:
        lines.append(f"Parent post (context only, do not annotate): {parent} ...")
    lines.append(f"Post: {post}")
    lines.append("Answer:")
    return "\n".join(lines)


# ---------- Answers ----------

def parse_reply(reply: str) -> tuple[dict, list[str]]:
    """A validated annotation from a model reply, plus whatever had to be fixed."""
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)
    if not match:
        return schema.blank(), ["no JSON object in reply"]
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return schema.blank(), [f"invalid JSON: {exc.msg}"]
    if not isinstance(data, dict):
        return schema.blank(), ["JSON was not an object"]
    return schema.validate(data)


def vote(answers: list[dict]) -> tuple[dict, dict]:
    """Majority per field across the votes, with each field's agreement share."""
    final, agreement = {}, {}
    for field in schema.FIELDS:
        values = [a[field.name] for a in answers]
        if field.kind == "multi":
            counts = Counter(v for value in values for v in value)
            kept = [v for v, n in counts.most_common() if n > len(values) / 2]
            kept = kept[: field.max_values] or list(values[0])
            final[field.name] = kept
            agreement[field.name] = sum(set(v) == set(kept) for v in values) / len(values)
        else:
            winner, n = Counter(values).most_common(1)[0]
            final[field.name] = winner
            agreement[field.name] = n / len(values)
    return final, agreement


# ---------- Providers ----------

def api_key(names: tuple[str, ...] = ()) -> str:
    """The LLM key from the environment, else from the repo's .env. Never logged."""
    import os

    for name in (names or config.KEY_NAMES):
        if os.environ.get(name):
            return os.environ[name]
    if config.ENV_FILE.exists():
        for line in config.ENV_FILE.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() in (names or config.KEY_NAMES) and value.strip():
                return value.strip().strip("\"'")
    raise SystemExit(f"No API key: set one of {', '.join(names or config.KEY_NAMES)} "
                     f"in the environment or in {config.ENV_FILE}")


def gemini_provider(model: str, system: str, temperature: float):
    from google import genai                       # pip install google-genai
    from google.genai import types

    client = genai.Client(api_key=api_key())
    settings = types.GenerateContentConfig(system_instruction=system,
                                           temperature=temperature,
                                           response_mime_type="application/json")

    def ask(user: str) -> str:
        return client.models.generate_content(model=model, contents=user,
                                              config=settings).text
    return ask


def anthropic_provider(model: str, system: str, temperature: float):
    from anthropic import Anthropic           # pip install anthropic

    client = Anthropic(api_key=api_key(("ANTHROPIC_API_KEY",)))

    def ask(user: str) -> str:
        reply = client.messages.create(
            model=model, max_tokens=300, temperature=temperature, system=system,
            messages=[{"role": "user", "content": user}])
        return "".join(block.text for block in reply.content if block.type == "text")
    return ask


def ollama_provider(model: str, system: str, temperature: float,
                    host: str = "http://localhost:11434"):
    """A model served locally by Ollama. No key, no quota, no 503s.

    Measured on this machine against the hosted alternatives, with the real
    2,637-token prompt: qwen3:4b 1.19s per call at four workers;
    gemini-3.1-flash-lite ~7s and failing 2 calls in 4;
    gemini-3.5-flash-lite 24.9s. Local is both faster and deterministic, which
    matters more than either when there is a deadline.

    `think: false` matters for the hybrid Qwen3 checkpoints, which otherwise
    spend their output budget on a <think> block. `format` asks for JSON;
    `keep_alive` stops the weights being unloaded between calls.
    """
    import urllib.request

    def ask(user: str) -> str:
        body = json.dumps({
            "model": model, "stream": False, "think": False, "keep_alive": "30m",
            "format": "json",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"temperature": temperature, "num_predict": 300},
        }).encode()
        request = urllib.request.Request(
            f"{host}/api/chat", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())["message"]["content"]
    return ask


def with_retries(ask, attempts: int = 5, wait: float = 4.0):
    """Retry with a growing pause: 429 and 503 are common and usually pass."""
    def call(user: str) -> str:
        for attempt in range(attempts):
            try:
                return ask(user)
            except Exception as exc:               # noqa: BLE001 - any API error is retryable
                if attempt == attempts - 1:
                    raise
                print(f"  {type(exc).__name__}: {str(exc)[:120]} "
                      f"- retrying in {wait * 2 ** attempt:.0f}s", flush=True)
                time.sleep(wait * 2 ** attempt)
        raise AssertionError("unreachable")
    return call


# ---------- Review gate ----------

NAME_RX = re.compile(r"\b(?:Dr\.?|Doctor)\s+[A-Z][a-z]{2,}")
HANDLE_RX = re.compile(r"(?<![\w/])@[A-Za-z][\w.]{2,}")
TITLECASE_PAIR_RX = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\s[A-Z][a-z]{2,}\b")


def exposure_report(rows: pd.DataFrame, system: str, model: str, votes: int) -> str:
    """What a real run would send, how much it costs, and what PII may ride along."""
    users = [build_user(row) for _, row in rows.iterrows()]
    system_tokens = len(system) / 4                  # chars/4: fine for English
    user_tokens = [len(u) / 4 for u in users]
    calls = len(rows) * votes
    in_tokens = calls * (system_tokens + sum(user_tokens) / len(users))
    out_tokens = calls * 85                          # seven fields of JSON

    lines = [
        "WHAT A REAL RUN WOULD SEND",
        f"  rows                {len(rows):,}",
        f"  votes per row       {votes}",
        f"  calls               {calls:,}",
        f"  fixed prefix        {system_tokens:,.0f} tokens, identical every call",
        f"  document            {sum(user_tokens) / len(users):,.0f} tokens mean, "
        f"{max(user_tokens):,.0f} max",
        f"  input total         {in_tokens / 1e6:,.1f}M tokens",
        f"  output total        {out_tokens / 1e6:,.2f}M tokens",
        "",
        "COST (list prices as of 2026-10, no batch discount, no context caching)",
    ]
    for name, (price_in, price_out) in PRICES.items():
        cost = in_tokens / 1e6 * price_in + out_tokens / 1e6 * price_out
        mark = "  <- selected" if name == model else ""
        lines.append(f"  {name:<26} ${cost:6.2f}{mark}")
    lines += [
        "",
        "  A Gemini Batch API run is normally half of the above.",
        f"  {system_tokens / (system_tokens + sum(user_tokens) / len(users)):.0%} of input "
        "is the fixed prefix, so context caching is where the rest of the savings are.",
        "",
        "RESIDUAL PII (heuristic scan of the text as it would actually be sent)",
    ]
    raw = rows["text"].astype(str).map(lambda t: " ".join(t.split())[: config.DOC_CHARS])
    sent = raw.map(deidentify)
    removed = (raw != sent).sum()
    checks = {
        "salutation handles ('Hi LoriBeth')": SALUTATION_RX,
        "clinician names ('Dr. Smith')": NAME_RX,
        "@handles": HANDLE_RX,
        "Title Case name-like pairs": TITLECASE_PAIR_RX,
    }
    for label, pattern in checks.items():
        before = raw.str.contains(pattern, regex=True, na=False).sum()
        after = sent.str.contains(pattern, regex=True, na=False).sum()
        lines.append(f"  {label:<36} {before:,} rows -> {after:,} after redaction")
    placeholders = sum(raw.str.count(re.escape(p)).sum() for p in schema.PLACEHOLDERS)
    lines += [
        f"  {'redaction placeholders in the corpus':<36} {placeholders:,}",
        f"  {'rows this run redacts':<36} {removed:,} ({removed / len(rows):.1%})",
        "",
        "  Read these numbers with the caveats:",
        "  - If the placeholder count above is zero, the text is unredacted for",
        "    names whatever an upstream pipeline reports, and should be treated",
        "    that way. Check it against the corpus rather than against a claim.",
        "  - The salutation pass above removes the handle of the person being",
        "    addressed, which is the risk worth removing, and nothing else.",
        "  - The Title Case counter is deliberately crude and over-counts: it also",
        "    matches organisation names. It is not a leak count.",
        "  - Clinician and public-figure names in the body are left in on purpose.",
        "  Record the decision in the project's data-handling notes before sending.",
    ]
    return "\n".join(lines)


def write_prompt_files(rows: pd.DataFrame, system: str, calls: int,
                       model: str, votes: int) -> None:
    config.PROMPTS.mkdir(parents=True, exist_ok=True)
    (config.PROMPTS / "system.txt").write_text(system, encoding="utf-8")

    # Spread the dumped calls across the file rather than taking the first few,
    # which would all be gold rows from one pool.
    stride = max(1, len(rows) // max(1, calls))
    sample = rows.iloc[[i * stride for i in range(calls) if i * stride < len(rows)]]
    dump = []
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        dump.append(f"{'=' * 78}\nCALL {i}  (id {row['id']}, pool {row['pool']}, "
                    f"split {row['split']})\n{'=' * 78}\n"
                    f"--- system instruction: identical every call, see system.txt ---\n"
                    f"--- user message ---\n{build_user(row)}")
    (config.PROMPTS / "sample_calls.txt").write_text("\n\n".join(dump), encoding="utf-8")

    report = exposure_report(rows, system, model, votes)
    (config.PROMPTS / "exposure_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {config.PROMPTS / 'system.txt'} "
          f"({len(system):,} chars, ~{len(system) / 4:,.0f} tokens)")
    print(f"wrote {config.PROMPTS / 'sample_calls.txt'} ({calls} calls)")
    print(f"wrote {config.PROMPTS / 'exposure_report.txt'}")
    print("\nnothing was sent.")


# ---------- Run ----------

def load_pools(which: str) -> pd.DataFrame:
    paths = {"random": [config.POOL_RANDOM], "enriched": [config.POOL_ENRICHED],
             "pilot": [config.POOL_PILOT], "debug": [config.POOL_DEBUG],
             "verify": [config.POOL_VERIFY],
             "all": [config.POOL_RANDOM, config.POOL_ENRICHED]}[which]
    missing = [p for p in paths if not p.exists()]
    if missing:
        script = {"pilot": "pilot.py", "debug": "pilot.py --debug-slice",
                  "verify": "verify.py --build-pool"}.get(which, "sample.py")
        raise SystemExit(f"Missing {missing[0]}. Run `python {script}` first.")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def annotate(rows: pd.DataFrame, ask, system: str, model: str, provider: str,
             votes: int, cache: dict, save, workers: int = 1,
             pause: float = 0.0) -> pd.DataFrame:
    """Every (row, vote) pair, run across `workers` threads.

    Four workers is the measured sweet spot against a local Ollama on this
    8 GB card: 2.7x the single-threaded rate, with eight no better because the
    KV caches no longer fit. The cache is written under a lock and flushed
    every so often rather than per answer, so a stopped run still resumes
    without the threads fighting over the file.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    pending = 0

    def fetch(item: tuple[str, int]) -> tuple[str, str | None]:
        """(cache key, error). The answer itself goes straight into the cache."""
        nonlocal pending
        key, user = item[0], item[1]
        with lock:
            if key in cache:
                return key, None
        if pause:
            time.sleep(pause)
        try:
            answer = ask(user)
        except Exception as exc:                   # noqa: BLE001 - keep the other rows
            return key, f"{type(exc).__name__}: {str(exc)[:100]}"
        with lock:
            cache[key] = answer
            pending += 1
            if pending >= 50:
                save()
                pending = 0
        return key, None

    keys: dict[str, list[str]] = {}
    work: list[tuple[str, str]] = []
    for _, row in rows.iterrows():
        user = build_user(row)
        keys[row["id"]] = []
        for v in range(votes):
            key = hashlib.sha1(
                f"{provider}|{model}|{v}|{system}|{user}".encode()).hexdigest()
            keys[row["id"]].append(key)
            work.append((key, user))

    todo = sum(1 for key, _ in work if key not in cache)
    print(f"  {len(work):,} calls, {len(work) - todo:,} already cached, "
          f"{todo:,} to fetch on {workers} worker(s)")
    errors: dict[str, str] = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (key, error) in enumerate(pool.map(fetch, work), start=1):
            if error:
                errors[key] = error
            if done % 200 == 0 or done == len(work):
                rate = done / max(time.time() - started, 1e-6)
                left = (len(work) - done) / max(rate, 1e-6)
                print(f"  ({done:,}/{len(work):,})  {rate:.2f} calls/s  "
                      f"~{left / 60:.0f} min left", flush=True)
    save()

    out = []
    for _, row in rows.iterrows():
        answers, problems = [], []
        for v, key in enumerate(keys[row["id"]]):
            if key in errors:
                problems.append(f"vote {v}: {errors[key]}")
                continue
            if key not in cache:
                problems.append(f"vote {v}: no answer")
                continue
            parsed, issues = parse_reply(cache[key])
            answers.append(parsed)
            problems += [f"vote {v}: {p}" for p in issues]

        if answers:
            final, agreement = vote(answers)
            parse_ok = not problems
        else:
            final, agreement = schema.blank(), dict.fromkeys(schema.NAMES, 0.0)
            parse_ok = False

        out.append({
            "id": row["id"], "pool": row["pool"], "split": row["split"],
            **final,
            "sentiment": schema.derive_sentiment(final["emotion"]),
            **{f"agree_{k}": round(v, 3) for k, v in agreement.items()},
            "parse_ok": parse_ok, "problems": "; ".join(problems),
            "votes": len(answers), "provider": provider, "model": model,
        })
    frame = pd.DataFrame(out)
    frame["emotion"] = frame["emotion"].apply(lambda v: "|".join(v))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider",
                        choices=("none", "gemini", "anthropic", "ollama"),
                        default="none")
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel calls (default: 4 for ollama, 1 otherwise)")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="ollama server")
    parser.add_argument("--model", default=config.DEFAULT_MODEL)
    parser.add_argument("--pool",
                        choices=("random", "enriched", "pilot", "debug",
                                 "verify", "all"),
                        default="all")
    parser.add_argument("--votes", type=int, default=config.DEFAULT_VOTES)
    parser.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE)
    parser.add_argument("--limit", type=int, default=0, help="first N rows only")
    parser.add_argument("--train-rows", type=int, default=0,
                        help="cap the train split, sampled across the pools")
    parser.add_argument("--val-rows", type=int, default=0, help="cap the val split")
    parser.add_argument("--calls", type=int, default=3,
                        help="full calls to dump under --provider none")
    parser.add_argument("--out", help="where to write the labels "
                                      "(default: data/teacher_labels_<pool>.csv)")
    args = parser.parse_args()

    config.ensure_dirs()
    rows = load_pools(args.pool)
    if args.train_rows or args.val_rows:
        # Keep the random/enriched mix of each split rather than taking a head,
        # so the rare fields stay represented in a smaller budget.
        caps = {"train": args.train_rows, "val": args.val_rows}
        kept = []
        for split, cap in caps.items():
            block = rows[rows["split"] == split]
            if cap and len(block) > cap:
                share = block["pool"].value_counts(normalize=True)
                parts = [block[block["pool"] == name].sample(
                    min(int(round(cap * fraction)), (block["pool"] == name).sum()),
                    random_state=config.SEED)
                    for name, fraction in share.items()]
                block = pd.concat(parts)
            kept.append(block)
        rows = pd.concat(kept).sample(frac=1, random_state=config.SEED).reset_index(drop=True)
        print(f"  capped to {len(rows):,} rows: "
              + ", ".join(f"{k} {int((rows['split'] == k).sum())}" for k in caps))
    if args.limit:
        rows = rows.head(args.limit)
    system = build_system()

    if args.provider == "none":
        write_prompt_files(rows, system, args.calls, args.model, args.votes)
        return

    cache_path: Path = config.TEACHER_CACHE
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    def save() -> None:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    if args.provider == "ollama":
        ask = with_retries(ollama_provider(args.model, system, args.temperature,
                                           args.host), attempts=3, wait=2.0)
        workers = args.workers or 4
        pause = 0.0
    else:
        build = {"gemini": gemini_provider, "anthropic": anthropic_provider}
        ask = with_retries(build[args.provider](args.model, system, args.temperature))
        workers = args.workers or 1
        pause = config.RPM_PAUSE
    print(f"{len(rows):,} rows x {args.votes} votes with {args.model} "
          f"({args.provider})")
    labels = annotate(rows, ask, system, args.model, args.provider, args.votes,
                      cache, save, workers=workers, pause=pause)
    save()

    # One file per pool, so a debug run cannot overwrite the pilot's labels.
    # --out is for the second model of the verify step, which needs its own file.
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (config.TEACHER_LABELS if args.pool == "all" else
                    config.TEACHER_LABELS.with_name(f"teacher_labels_{args.pool}.csv"))
    labels.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nwrote {out_path}")
    print(f"parse_ok: {labels['parse_ok'].mean():.1%}")
    for field in schema.NAMES:
        low = (labels[f"agree_{field}"] < 0.67).mean()
        print(f"  {field:<18} mean agreement {labels[f'agree_{field}'].mean():.3f}, "
              f"{low:.1%} of rows below 2/3")


if __name__ == "__main__":
    main()
