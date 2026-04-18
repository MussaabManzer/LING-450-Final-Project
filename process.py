import glob
import json
import os
import re
from pathlib import Path

import pandas as pd


YEARS = [2000, 2010, 2015, 2017, 2021, 2023]

INPUT_ROOT = Path("path/to/congressional_record_output")
OUTPUT_ROOT = Path("path/to/preprocessed_output")

MIN_WORD_COUNT = 20

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_words(text: str) -> int:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    return len(tokens)


def is_procedural_filler(text: str) -> bool:
    if not text:
        return True

    text = text.lower().strip()

    if count_words(text) < MIN_WORD_COUNT:
        return True

    procedural_patterns = [
        r"\bi yield\b",
        r"\bi yield back\b",
        r"\bi reserve the balance of my time\b",
        r"\bi reserve\b",
        r"\bwithout objection\b",
        r"\bso ordered\b",
        r"\bthe clerk will\b",
        r"\bthe clerk read as follows\b",
        r"\bthe reading was dispensed with\b",
        r"\bmadam speaker\b",
        r"\bmr\. speaker\b",
        r"\bthe speaker pro tempore\b",
        r"\bpursuant to the rule\b",
        r"\bthe gentleman from\b",
        r"\bthe gentlewoman from\b",
        r"\bthe gentleman yields\b",
        r"\bthe gentlewoman yields\b",
        r"\bis recognized\b",
        r"\bfor what purpose does the gentleman\b",
        r"\bfor what purpose does the gentlewoman\b",
        r"\bto revise and extend\b",
        r"\bpermission to revise and extend\b",
        r"\bquorum is not present\b",
        r"\bthe question is on\b",
        r"\bthose in favor say aye\b",
        r"\bthose opposed\b",
        r"\bthe motion is adopted\b",
        r"\bthe motion was agreed to\b",
        r"\bthe amendment was agreed to\b",
        r"\bthe amendment is agreed to\b",
        r"\ba recorded vote was ordered\b",
        r"\bthe house stands in recess\b",
        r"\bthe house is adjourned\b",
    ]

    for pattern in procedural_patterns:
        if re.search(pattern, text) and count_words(text) < 60:
            return True

    if count_words(text) < 40:
        address_markers = [
            "madam speaker",
            "mr. speaker",
            "i yield",
            "without objection",
            "the gentleman from",
            "the gentlewoman from",
            "is recognized",
        ]
        hits = sum(1 for phrase in address_markers if phrase in text)
        if hits >= 1:
            return True

    return False


def process_year(year: int, input_root: Path) -> pd.DataFrame:
    year_dir = input_root / str(year)
    json_files = glob.glob(str(year_dir / "**" / "*.json"), recursive=True)

    print(f"\n=== {year} ===")
    print(f"Found {len(json_files)} JSON files in {year_dir}")

    records = []
    skipped_files = 0
    skipped_non_house = 0
    skipped_non_speech = 0
    skipped_missing_bioguide = 0
    skipped_procedural = 0

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                document = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            skipped_files += 1
            continue

        header = document.get("header", {})
        chamber = header.get("chamber", "")

        if chamber != "House":
            skipped_non_house += 1
            continue

        date_string = f"{header.get('year', '')}-{header.get('month', '')}-{header.get('day', '')}"
        document_id = document.get("id", os.path.basename(file_path))

        for item in document.get("content", []):
            if item.get("kind") != "speech":
                skipped_non_speech += 1
                continue

            speaker_bioguide = item.get("speaker_bioguide")
            if not speaker_bioguide:
                skipped_missing_bioguide += 1
                continue

            speaker = str(item.get("speaker", "Unknown")).strip()
            text = clean_text(item.get("text", ""))

            if is_procedural_filler(text):
                skipped_procedural += 1
                continue

            records.append(
                {
                    "doc_id": document_id,
                    "date": date_string,
                    "year": year,
                    "chamber": chamber,
                    "speaker": speaker,
                    "speaker_bioguide": speaker_bioguide,
                    "word_count": count_words(text),
                    "text": text,
                }
            )

    df = pd.DataFrame(records)

    print(f"Kept {len(df)} speech turns")
    print(f"Skipped unreadable files: {skipped_files}")
    print(f"Skipped non-House files: {skipped_non_house}")
    print(f"Skipped non-speech items: {skipped_non_speech}")
    print(f"Skipped missing bioguide IDs: {skipped_missing_bioguide}")
    print(f"Skipped procedural or short turns: {skipped_procedural}")

    if not df.empty:
        print(f"Unique speakers: {df['speaker_bioguide'].nunique()}")
        print(f"Mean word count: {df['word_count'].mean():.2f}")

    return df


def main() -> None:
    summary_rows = []

    for year in YEARS:
        df = process_year(year, INPUT_ROOT)

        output_file = OUTPUT_ROOT / f"turns_{year}_house_clean.csv"
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"Saved: {output_file}")

        summary_rows.append(
            {
                "year": year,
                "rows": len(df),
                "unique_speakers": df["speaker_bioguide"].nunique() if not df.empty else 0,
                "mean_word_count": round(df["word_count"].mean(), 2) if not df.empty else 0,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_file = OUTPUT_ROOT / "preprocess_summary.csv"
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

    print("\n=== Done ===")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
