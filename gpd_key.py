"""
gpd_keyness.py
--------------
Computes word-level keyness (log-odds ratio) between high- and low-populism
speeches by matching Global Populism Database (GPD) entries to local text files.

Outputs (written to OUTPUT_DIR):
    matched_gpd_speeches.csv     — matched rows with metadata and raw text
    unmatched_gpd_ids.csv        — GPD rows that could not be matched to a file
    document_token_stats.csv     — per-document token counts
    word_keyness_full.csv        — full ranked word comparison table
    top_high_populism_words.csv  — top N words overrepresented in high-populism speeches
    top_low_populism_words.csv   — top N words overrepresented in low-populism speeches
    candidate_lexicon_words.csv  — subset matching seed patterns for lexicon development
"""

import math
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SPEECH_DIR = "data/speeches"          # directory of .txt speech files
GPD_PATH   = "data/GPD_v2.1.csv"      # GPD CSV with scores and merging variable
OUTPUT_DIR = "outputs/keyness"

SCORE_COL  = "totalaverage"           # populism score column in GPD
ID_COL     = "merging_variable"       # column used to match GPD rows to filenames

MIN_WORD_FREQ = 5                     # drop words appearing fewer than this many times total
TOP_N         = 200                   # number of words to save in top-high / top-low files

ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "to", "of", "in", "on", "for", "from", "with", "by", "as",
    "at", "is", "are", "was", "were", "be", "been", "being", "it", "its", "into",
    "about", "over", "under", "after", "before", "during", "through", "because",
    "while", "where", "when", "who", "whom", "which", "what", "why", "how",
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "they",
    "them", "their", "theirs", "themselves", "we", "us", "our", "ours", "ourselves",
    "am", "do", "does", "did", "doing", "have", "has", "had", "having", "not",
    "no", "yes", "can", "could", "shall", "should", "will", "would", "may", "might",
    "must", "also", "very", "more", "most", "less", "many", "much", "some", "any",
    "all", "each", "every", "other", "another", "such", "than", "so", "just", "only",
    "up", "down", "out", "off", "again", "further", "here", "there",
}

# Seed patterns for surfacing candidate anti-elite / institutional terms
SEED_PATTERNS = [
    r"elite", r"elit", r"establish", r"corrupt", r"politic", r"bureaucr",
    r"lobb", r"washington", r"insider", r"special", r"interest",
    r"congress", r"government", r"media", r"wall", r"state",
    r"global", r"rig", r"dishonest", r"betray", r"scandal", r"truth",
    r"unfair", r"deep",
]


# ============================================================
# HELPERS
# ============================================================

def safe_read(path: str) -> str:
    """Read a text file, trying multiple encodings."""
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode {path}")


def normalize_id(value) -> str:
    """Normalize a GPD ID or filename stem for fuzzy matching."""
    if pd.isna(value):
        return ""
    s = str(value).strip().lower().replace(".txt", "")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def clean_tokens(tokens: list[str]) -> list[str]:
    return [
        t for t in tokens
        if len(t) >= 3 and t not in STOPWORDS
    ]


def log_odds_ratio(
    count_high: int, total_high: int,
    count_low: int,  total_low: int,
    alpha: float = 0.5,
) -> float:
    """
    Smoothed log-odds ratio.
    Positive  => overrepresented in high-populism speeches.
    Negative  => overrepresented in low-populism speeches.
    """
    p_high = (count_high + alpha) / (total_high + 2 * alpha)
    p_low  = (count_low  + alpha) / (total_low  + 2 * alpha)
    return math.log(p_high / (1 - p_high)) - math.log(p_low / (1 - p_low))


def match_file(gpd_id_norm: str, file_lookup: dict[str, str]) -> str | None:
    """Return the file path for a normalized GPD ID, or None if not found."""
    if gpd_id_norm in file_lookup:
        return file_lookup[gpd_id_norm]
    candidates = [
        p for norm, p in file_lookup.items()
        if gpd_id_norm and (gpd_id_norm in norm or norm in gpd_id_norm)
    ]
    return candidates[0] if len(candidates) == 1 else None


# ============================================================
# LOAD DATA
# ============================================================

def load_csv(path: str) -> pd.DataFrame:
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read CSV: {path}")


print("Loading GPD...")
gpd = load_csv(GPD_PATH)

missing_cols = [c for c in [ID_COL, SCORE_COL] if c not in gpd.columns]
if missing_cols:
    raise ValueError(f"Missing required GPD columns: {missing_cols}")

gpd["_id_norm"] = gpd[ID_COL].apply(normalize_id)
gpd = gpd[gpd["_id_norm"].ne("") & gpd[SCORE_COL].notna()].copy()
print(f"  GPD rows with usable ID and score: {len(gpd)}")

print("Scanning speech files...")
speech_files = list(Path(SPEECH_DIR).glob("*.txt"))
if not speech_files:
    raise ValueError(f"No .txt files found in: {SPEECH_DIR}")

file_lookup: dict[str, str] = {}
for p in speech_files:
    norm = normalize_id(p.stem)
    file_lookup[norm] = str(p)   # last writer wins on duplicates

print(f"  Found {len(speech_files)} text files.")


# ============================================================
# MATCH GPD ROWS TO TEXT FILES
# ============================================================

matched_rows, unmatched_ids = [], []

for _, row in gpd.iterrows():
    path = match_file(row["_id_norm"], file_lookup)
    if path is None:
        unmatched_ids.append(row[ID_COL])
        continue
    try:
        text = safe_read(path)
    except Exception as e:
        print(f"  Could not read {path}: {e}")
        unmatched_ids.append(row[ID_COL])
        continue

    matched_rows.append({
        ID_COL:       row[ID_COL],
        "country":    row.get("country"),
        "leader":     row.get("leader"),
        "party":      row.get("party"),
        "speechtype": row.get("speechtype"),
        "yearbegin":  row.get("yearbegin"),
        "yearend":    row.get("yearend"),
        SCORE_COL:    row[SCORE_COL],
        "file":       path,
        "text":       text,
    })

matched = pd.DataFrame(matched_rows)
print(f"  Matched: {len(matched)}  |  Unmatched: {len(unmatched_ids)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
matched.drop(columns="text").to_csv(f"{OUTPUT_DIR}/matched_gpd_speeches.csv", index=False)
if unmatched_ids:
    pd.DataFrame({"unmatched_id": unmatched_ids}).to_csv(
        f"{OUTPUT_DIR}/unmatched_gpd_ids.csv", index=False
    )

if matched.empty:
    raise ValueError("No speeches matched. Check that filenames align with merging_variable.")


# ============================================================
# SPLIT INTO HIGH / LOW POPULISM
# ============================================================

median_score = matched[SCORE_COL].median()
matched["pop_group"] = (matched[SCORE_COL] > median_score).map({True: "high", False: "low"})
print(f"\nMedian {SCORE_COL}: {median_score:.4f}")
print(matched["pop_group"].value_counts().to_string())


# ============================================================
# TOKENIZE AND COUNT
# ============================================================

high_counter, low_counter = Counter(), Counter()
doc_stats = []

for _, row in matched.iterrows():
    raw    = tokenize(row["text"])
    clean  = clean_tokens(raw)
    group  = row["pop_group"]

    doc_stats.append({
        ID_COL:              row[ID_COL],
        "pop_group":         group,
        "raw_token_count":   len(raw),
        "clean_token_count": len(clean),
        SCORE_COL:           row[SCORE_COL],
    })

    (high_counter if group == "high" else low_counter).update(clean)

pd.DataFrame(doc_stats).to_csv(f"{OUTPUT_DIR}/document_token_stats.csv", index=False)

total_high = sum(high_counter.values())
total_low  = sum(low_counter.values())
print(f"\nClean tokens — high: {total_high:,}  |  low: {total_low:,}")


# ============================================================
# BUILD KEYNESS TABLE
# ============================================================

vocab = set(high_counter) | set(low_counter)

rows = []
for word in vocab:
    ch, cl = high_counter[word], low_counter[word]
    if ch + cl < MIN_WORD_FREQ:
        continue
    rows.append({
        "word":               word,
        "count_high":         ch,
        "count_low":          cl,
        "total_count":        ch + cl,
        "log_odds":           log_odds_ratio(ch, total_high, cl, total_low),
        "ratio_high_low_p1":  ch / (cl + 1),
    })

if not rows:
    raise ValueError("No words survived filtering. Try lowering MIN_WORD_FREQ.")

word_df = (
    pd.DataFrame(rows)
    .sort_values("log_odds", ascending=False)
    .reset_index(drop=True)
)

word_df.to_csv(f"{OUTPUT_DIR}/word_keyness_full.csv", index=False)
word_df.head(TOP_N).to_csv(f"{OUTPUT_DIR}/top_high_populism_words.csv", index=False)
word_df.tail(TOP_N).sort_values("log_odds").to_csv(
    f"{OUTPUT_DIR}/top_low_populism_words.csv", index=False
)

seed_re = re.compile("|".join(SEED_PATTERNS), flags=re.IGNORECASE)
word_df[word_df["word"].str.contains(seed_re, na=False)].to_csv(
    f"{OUTPUT_DIR}/candidate_lexicon_words.csv", index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\nTop 50 HIGH-populism words:")
print(word_df[["word", "count_high", "count_low", "log_odds"]].head(50).to_string(index=False))

print("\nTop 50 LOW-populism words:")
print(word_df[["word", "count_high", "count_low", "log_odds"]].tail(50).to_string(index=False))

print(f"\nDone. Outputs saved to: {OUTPUT_DIR}")
