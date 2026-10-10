"""Build the two annotation pools from the cleaned corpus.

    python sample.py                  # writes data/pool_random.csv + pool_enriched.csv
    python sample.py --dry-run        # print the plan, write nothing

Two pools, kept separate on purpose:

**pool_random** — a stratified sample over platform x type x length, with a cap
on any one platform (YouTube is 65% of the corpus and would otherwise swamp
everything) and a floor per cell so the small platforms are usable at all.
Within a cell, rows are spread across the 33 BERTopic topics so no single topic
fills a cell. Because the strata are not sampled proportionally, every row
carries a `weight` = (rows of that stratum in the corpus) / (rows drawn from
it). Corpus-level statements ("X% of posts describe being dismissed") must use
those weights; model quality metrics (macro-F1) do not.

**pool_enriched** — rows retrieved by keyword for the three fields that are too
rare to learn or measure from a random sample. These heuristics decide only
*what gets looked at*; they never become labels, and enriched rows never enter
the random pool or its gold set, so the unbiased estimates stay unbiased.

Splits: pool_random -> gold_random / val / train, pool_enriched -> gold_enriched
/ train. `gold_*` rows are the ones humans annotate.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import config
from fewshot import EXAMPLE_IDS

# Retrieval heuristics for the enriched pool. Oversampling only: never labels.
ENRICH_PATTERNS: dict[str, str] = {
    "care": (r"(?:dismiss|not taken serious|wasn.?t taken serious|just anxiety|only anxiety|"
             r"it was anxiety|all in my head|brush(?:ed)? (?:me )?off|didn.?t believe me|"
             r"did ?n.?t listen|would ?n.?t listen|told me i was fine|sent me home|"
             r"misdiagnos|blamed (?:it on )?(?:my )?(?:weight|anxiety|stress)|"
             r"said i was too young|could ?n.?t afford|no insurance|waiting list|"
             r"wait(?:ed)? (?:months|a year))"),
    "claim": (r"(?:cure[sd]?|reverse[sd]?|heal(?:ed|s)?|prevent[sd]?|caused by|causes |"
              r"big pharma|statins are|cholesterol (?:myth|hoax)|"
              r"instead of (?:meds|medication|statins)|better than (?:meds|medication)|"
              r"apple cider vinegar|cayenne|turmeric|\bcoq10\b|magnesium)"),
    "womens": (r"(?:pregnan|postpartum|post.?partum|menopaus|peri.?menopaus|\bhrt\b|estrogen|"
               r"oestrogen|\bscad\b|spontaneous coronary|\bfmd\b|takotsubo|broken heart "
               r"syndrome|hysterectom|women.?s symptoms|because i.?m a woman)"),
}


# ---------- Allocation ----------

def apportion(sizes: dict, total: int, floor: int = 0, caps: dict | None = None) -> dict:
    """Spread `total` draws across keys in proportion to `sizes`.

    Honours a per-key `floor`, optional per-key `caps`, and never asks a key for
    more rows than it has. Leftovers from a capped key flow to the others.
    """
    keys = [k for k, v in sizes.items() if v > 0]
    limit = {k: min((caps or {}).get(k, sizes[k]), sizes[k]) for k in keys}
    alloc = dict.fromkeys(keys, 0)
    left = min(total, sum(limit.values()))

    for k in keys:                                  # floors first
        take = min(floor, limit[k], left)
        alloc[k] += take
        left -= take

    while left > 0:
        open_keys = [k for k in keys if alloc[k] < limit[k]]
        if not open_keys:
            break
        weight = sum(sizes[k] for k in open_keys)
        share = {k: left * sizes[k] / weight for k in open_keys}
        added = 0
        for k in open_keys:                         # integer part
            take = min(int(share[k]), limit[k] - alloc[k], left - added)
            alloc[k] += take
            added += take
        if added == 0:                              # remainders, largest first
            for k in sorted(open_keys, key=lambda k: -share[k]):
                if added >= left:
                    break
                alloc[k] += 1
                added += 1
        left -= added
    return alloc


def spread_over_topics(group: pd.DataFrame, n: int) -> pd.DataFrame:
    """Take `n` rows from one stratum, spread evenly across its topics.

    Rows are shuffled, then ordered into topic blocks, then sampled at a fixed
    stride: a systematic sample over topics, so a cell cannot come out as one
    topic repeated.
    """
    if n >= len(group):
        return group
    ordered = (group.sample(frac=1, random_state=config.SEED)
               .sort_values("topic", kind="stable"))
    step = len(ordered) / n
    return ordered.iloc[[int(i * step) for i in range(n)]]


# ---------- Load ----------

def load_corpus() -> pd.DataFrame:
    config.require_corpus()
    df = pd.read_csv(config.CORPUS)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    df = df[~df["id"].isin(EXAMPLE_IDS)].copy()     # few-shot posts stay out

    full = pd.read_csv(config.CORPUS_FULL, usecols=["id", "is_reply", "parent_text"])
    full.columns = [c.lstrip("﻿") for c in full.columns]
    df = df.merge(full, on="id", how="left")

    # Topic assignments are optional: they only spread the draw within a
    # stratum, so without them every row sits in one topic and the spread
    # becomes a plain shuffle.
    if config.DOC_TOPICS.exists():
        topics = pd.read_csv(config.DOC_TOPICS, usecols=["id", "topic"])
        df = df.merge(topics, on="id", how="left")
    else:
        print(f"no {config.DOC_TOPICS.name}: drawing without topic spread")
        df["topic"] = -1
    df["topic"] = df["topic"].fillna(-1).astype(int)

    df["text"] = df["text"].astype(str)
    df["n_words"] = df["text"].str.split().str.len()
    short, long = config.LENGTH_BINS
    df["length_bin"] = pd.cut(df["n_words"], bins=[-1, short, long, 10**9],
                              labels=["short", "medium", "long"]).astype(str)
    df["stratum"] = df["platform"] + " | " + df["type"] + " | " + df["length_bin"]
    df["is_reply"] = df["is_reply"].fillna(False).astype(bool)
    df["parent_text"] = df["parent_text"].fillna("")
    return df


# ---------- Pools ----------

def draw_random(df: pd.DataFrame) -> pd.DataFrame:
    """The stratified pool, with sampling weights for unbiased estimates."""
    platform_sizes = df.groupby("platform").size().to_dict()
    cap = int(config.N_RANDOM * config.MAX_PLATFORM_SHARE)
    per_platform = apportion(platform_sizes, config.N_RANDOM,
                             caps=dict.fromkeys(platform_sizes, cap))

    picked = []
    for platform, quota in per_platform.items():
        block = df[df["platform"] == platform]
        cell_sizes = block.groupby("stratum").size().to_dict()
        per_cell = apportion(cell_sizes, quota, floor=config.MIN_PER_STRATUM)
        for stratum, n in per_cell.items():
            if n:
                picked.append(spread_over_topics(block[block["stratum"] == stratum], n))

    pool = pd.concat(picked, ignore_index=True)
    corpus_counts = df.groupby("stratum").size()
    drawn_counts = pool.groupby("stratum").size()
    pool["weight"] = (pool["stratum"].map(corpus_counts)
                      / pool["stratum"].map(drawn_counts)).round(4)
    pool["pool"] = "random"
    return pool


def draw_enriched(df: pd.DataFrame, taken: set[str]) -> pd.DataFrame:
    """Keyword-retrieved rows for the rare fields, excluding the random pool."""
    rest = df[~df["id"].isin(taken)]
    lowered = rest["text"].str.lower()
    per_pattern = config.N_ENRICHED // len(ENRICH_PATTERNS)

    picked, used = [], set()
    for name, pattern in ENRICH_PATTERNS.items():
        hits = rest[lowered.str.contains(pattern, regex=True, na=False)
                    & ~rest["id"].isin(used)]
        chosen = spread_over_topics(hits, min(per_pattern, len(hits))).copy()
        chosen["retrieved_by"] = name
        used.update(chosen["id"])
        picked.append(chosen)

    pool = pd.concat(picked, ignore_index=True)
    pool["weight"] = pd.NA            # retrieved, not a probability sample
    pool["pool"] = "enriched"
    return pool


def assign_splits(pool: pd.DataFrame, gold_n: int, val_n: int = 0) -> pd.DataFrame:
    """gold first, then val, then train — drawn at random inside the pool."""
    shuffled = pool.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    gold_name = "gold_random" if pool["pool"].iloc[0] == "random" else "gold_enriched"
    splits = ([gold_name] * gold_n + ["val"] * val_n
              + ["train"] * max(0, len(shuffled) - gold_n - val_n))
    shuffled["split"] = splits[: len(shuffled)]
    return shuffled


# ---------- Main ----------

COLUMNS = ["id", "date", "platform", "type", "topic", "n_words", "length_bin",
           "stratum", "weight", "is_reply", "parent_text", "text",
           "pool", "split", "retrieved_by"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = parser.parse_args()

    config.ensure_dirs()
    df = load_corpus()
    print(f"corpus: {len(df):,} rows, {df['topic'].nunique()} topics, "
          f"{df['stratum'].nunique()} strata")

    random_pool = assign_splits(draw_random(df), config.N_GOLD_RANDOM, config.N_VAL)
    enriched_pool = assign_splits(draw_enriched(df, set(random_pool["id"])),
                                  config.N_GOLD_ENRICHED)

    for name, pool in (("pool_random", random_pool), ("pool_enriched", enriched_pool)):
        print(f"\n{name}: {len(pool):,} rows")
        print(pool["split"].value_counts().to_string())
        print((pool["platform"].value_counts(normalize=True) * 100).round(1).to_string())
        if "retrieved_by" in pool and pool["retrieved_by"].notna().any():
            print(pool["retrieved_by"].value_counts().to_string())

    teacher_rows = len(random_pool) + len(enriched_pool)
    print(f"\nteacher calls: {teacher_rows:,} rows x {config.DEFAULT_VOTES} votes "
          f"= {teacher_rows * config.DEFAULT_VOTES:,}")
    print(f"human gold: {config.N_GOLD_RANDOM + config.N_GOLD_ENRICHED:,} rows")

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    for path, pool in ((config.POOL_RANDOM, random_pool),
                       (config.POOL_ENRICHED, enriched_pool)):
        out = pool.reindex(columns=COLUMNS)
        out.to_csv(path, index=False, encoding="utf-8")
        print(f"wrote {path}")

    manifest = {
        "seed": config.SEED,
        "corpus": str(config.CORPUS),
        "corpus_rows": int(len(df)),
        "doc_chars": config.DOC_CHARS,
        "parent_chars": config.PARENT_CHARS,
        "max_platform_share": config.MAX_PLATFORM_SHARE,
        "min_per_stratum": config.MIN_PER_STRATUM,
        "length_bins": list(config.LENGTH_BINS),
        "excluded_fewshot_ids": sorted(EXAMPLE_IDS),
        "pools": {
            name: {"rows": int(len(pool)),
                   "splits": pool["split"].value_counts().to_dict(),
                   "platforms": pool["platform"].value_counts().to_dict()}
            for name, pool in (("random", random_pool), ("enriched", enriched_pool))
        },
        "enrich_patterns": ENRICH_PATTERNS,
    }
    config.SAMPLE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {config.SAMPLE_MANIFEST}")


if __name__ == "__main__":
    main()
