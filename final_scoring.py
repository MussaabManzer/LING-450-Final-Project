from pathlib import Path
import math
import re

import pandas as pd


YEARS = [2000, 2010, 2015, 2017, 2021, 2023]

INPUT_DIR = Path("path/to/tagged")
OUTPUT_CSV = Path("path/to/output/features_master.csv")

INPUT_TEMPLATE = "turns_{year}_House_tagged.csv"

REQUIRED_COLS = [
    "doc_id",
    "date",
    "year",
    "chamber",
    "speaker",
    "speaker_bioguide",
    "word_count",
    "text",
    "party",
    "is_tea_party",
    "is_freedom_caucus",
    "is_populist_republican",
    "caucus",
]


ALL_PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
}

FIRST_PLURAL = {"we", "us", "our", "ours", "ourselves"}

PEOPLE_LEXICON = {
    "demonyms": {
        "american", "americans", "citizen", "citizens", "voter", "voters",
        "taxpayer", "taxpayers", "patriot", "patriots", "neighbor", "neighbors",
        "constituent", "constituents",
    },
    "folk_terms": {
        "people", "folks", "public", "population", "masses", "majority",
        "heartland", "grassroots", "mainstream", "ordinary", "everyday",
        "hardworking", "working", "common", "average", "real",
    },
    "worker_terms": {
        "worker", "workers", "employee", "employees", "labor", "labour",
        "workforce", "job", "jobs", "family", "families", "household",
        "households", "community", "communities",
    },
}

PEOPLE_UNIGRAMS = (
    PEOPLE_LEXICON["demonyms"]
    | PEOPLE_LEXICON["folk_terms"]
    | PEOPLE_LEXICON["worker_terms"]
)

PEOPLE_BIGRAMS = {
    "our people",
    "your people",
    "our families",
    "our workers",
    "our citizens",
    "our neighbors",
    "our communities",
    "our country",
    "our nation",
    "our future",
    "our children",
    "our jobs",
}


TIER1_ROOTS = [
    "corrupt", "betray", "deceit", "deceiv", "propagand", "undemocrat",
    "swamp", "crony", "oligarch", "plutocrat", "cabal", "weaponiz",
    "rigg", "uniparty", "globalist", "sellout", "puppet",
]

TIER1_PHRASES = [
    "deep state", "drain the swamp", "corrupt system", "broken system",
    "failed system", "rigged system", "witch hunt", "fake news", "out of touch",
    "bought and paid for", "stacked against", "corrupt elite", "corrupt politicians",
    "career politician", "political class", "ruling class", "shadow government",
    "crony capitalism", "special interests", "washington insiders", "dc insiders",
    "political insiders", "unelected bureaucrat", "faceless bureaucrat",
    "administrative state", "wall street", "big pharma", "big tech",
    "mainstream media", "lying media", "dishonest media", "take advantage of",
]

TIER2_ELITE_ROOTS = [
    "establishment", "bureaucrat", "bureaucracy", "lobbyist", "lobbying",
    "insider", "politician", "elite",
]

NEGATIVE_MODIFIERS = {
    "corrupt", "corrupted", "corruption", "dishonest", "dishonesty", "lying",
    "lied", "lies", "crooked", "criminal", "fraudulent", "fraud",
    "betray", "betrayed", "betrayal", "sold", "sellout", "broken", "failed",
    "failing", "rigged", "stacked", "dysfunctional", "incompetent", "incompetence",
    "disgrace", "disgraced", "shameful", "outrageous", "unconscionable",
    "unacceptable", "deplorable", "control", "controlled", "manipulate",
    "manipulated", "exploit", "exploited", "abuse", "abused", "ignore", "ignored",
    "abandon", "abandoned", "disconnected", "arrogant", "entrenched",
    "unaccountable", "unelected",
}

WINDOW_SIZE = 10


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def make_bigrams(tokens: list[str]) -> list[str]:
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


def people_centrism(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {
            "pronoun_ratio": math.nan,
            "first_plural_count": 0,
            "all_pronoun_count": 0,
            "people_term_density": math.nan,
            "people_unigram_count": 0,
            "people_bigram_count": 0,
            "people_term_hits": 0,
            "demonym_count": 0,
            "folk_term_count": 0,
            "worker_term_count": 0,
            "people_centrism_composite": math.nan,
        }

    tokens = tokenize(text)
    bigrams = make_bigrams(tokens)

    all_pronouns = sum(1 for token in tokens if token in ALL_PRONOUNS)
    first_plural = sum(1 for token in tokens if token in FIRST_PLURAL)
    pronoun_ratio = first_plural / all_pronouns if all_pronouns > 0 else 0.0

    unigram_hits = [token for token in tokens if token in PEOPLE_UNIGRAMS]
    bigram_hits = [bigram for bigram in bigrams if bigram in PEOPLE_BIGRAMS]
    total_hits = len(unigram_hits) + len(bigram_hits)
    density = total_hits / len(tokens) if tokens else 0.0
    composite = (pronoun_ratio + density) / 2

    return {
        "pronoun_ratio": round(pronoun_ratio, 4),
        "first_plural_count": first_plural,
        "all_pronoun_count": all_pronouns,
        "people_term_density": round(density, 4),
        "people_unigram_count": len(unigram_hits),
        "people_bigram_count": len(bigram_hits),
        "people_term_hits": total_hits,
        "demonym_count": sum(1 for token in tokens if token in PEOPLE_LEXICON["demonyms"]),
        "folk_term_count": sum(1 for token in tokens if token in PEOPLE_LEXICON["folk_terms"]),
        "worker_term_count": sum(1 for token in tokens if token in PEOPLE_LEXICON["worker_terms"]),
        "people_centrism_composite": round(composite, 4),
    }


def consume_phrases(text_lower: str, phrases: list[str]) -> tuple[int, str]:
    count = 0
    for phrase in sorted(phrases, key=len, reverse=True):
        occurrences = text_lower.count(phrase)
        if occurrences:
            count += occurrences
            text_lower = text_lower.replace(phrase, " " * len(phrase))
    return count, text_lower


def anti_elitism(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {
            "anti_elitism_score": math.nan,
            "raw_count": 0,
            "tier1_phrase_count": 0,
            "tier1_token_count": 0,
            "tier2_windowed_count": 0,
            "ae_word_count": 0,
        }

    text_lower = text.lower()
    word_count = len(tokenize(text_lower))

    tier1_phrase_count, remaining_text = consume_phrases(text_lower, TIER1_PHRASES)
    tokens = tokenize(remaining_text)

    tier1_hits = []
    for token in tokens:
        for root in TIER1_ROOTS:
            if root in token:
                tier1_hits.append(token)
                break

    tier1_token_count = len(tier1_hits)

    tier2_hits = []
    for i, token in enumerate(tokens):
        if not any(root in token for root in TIER2_ELITE_ROOTS):
            continue

        window = tokens[max(0, i - WINDOW_SIZE): min(len(tokens), i + WINDOW_SIZE + 1)]
        if any(modifier in window for modifier in NEGATIVE_MODIFIERS):
            tier2_hits.append(token)

    tier2_count = len(tier2_hits)

    raw_count = tier1_phrase_count + tier1_token_count + tier2_count
    score = (raw_count / word_count * 1000) if word_count > 0 else 0.0

    return {
        "anti_elitism_score": round(score, 4),
        "raw_count": raw_count,
        "tier1_phrase_count": tier1_phrase_count,
        "tier1_token_count": tier1_token_count,
        "tier2_windowed_count": tier2_count,
        "ae_word_count": word_count,
    }


def process_year(year: int, input_dir: Path) -> pd.DataFrame:
    file_path = input_dir / INPUT_TEMPLATE.format(year=year)
    print(f"Loading {file_path}...")

    df = pd.read_csv(file_path, usecols=REQUIRED_COLS)
    print(f"  Loaded {len(df)} rows")

    people_rows = df["text"].apply(people_centrism).apply(pd.Series)
    anti_elite_rows = df["text"].apply(anti_elitism).apply(pd.Series)

    return pd.concat([df, people_rows, anti_elite_rows], axis=1)


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    yearly_frames = []

    for year in YEARS:
        df_year = process_year(year, INPUT_DIR)
        yearly_frames.append(df_year)

        print(
            f"  Done: {year} — {len(df_year)} rows, "
            f"mean AE={df_year['anti_elitism_score'].mean():.4f}, "
            f"mean PC={df_year['people_centrism_composite'].mean():.4f}"
        )

    master_df = pd.concat(yearly_frames, ignore_index=True)
    master_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(master_df)} total rows to:\n{OUTPUT_CSV}")

    print("\nSummary by party and year:")
    summary = (
        master_df.groupby(["year", "party"])[
            ["anti_elitism_score", "people_centrism_composite"]
        ]
        .mean()
        .round(4)
    )
    print(summary.to_string())

    print("\nPopulist Republican vs Establishment Republican:")
    republican_df = master_df[master_df["party"] == "Republican"]
    group_summary = (
        republican_df.groupby("is_populist_republican")[
            ["anti_elitism_score", "people_centrism_composite"]
        ]
        .mean()
        .round(4)
    )
    print(group_summary.to_string())


if __name__ == "__main__":
    main()
