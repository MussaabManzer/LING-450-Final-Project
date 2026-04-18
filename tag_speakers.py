from pathlib import Path

import pandas as pd


YEARS = [2000, 2010, 2015, 2017, 2021, 2023]

INPUT_DIR = Path("path/to/preprocessed")
OUTPUT_DIR = Path("path/to/tagged")
LEGISLATOR_DIR = Path("path/to/legislator_info")

CURRENT_PATH = LEGISLATOR_DIR / "legislators-current.csv"
HISTORICAL_PATH = LEGISLATOR_DIR / "legislators-historical.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FREEDOM_CAUCUS_BIOGUIDES = {
    "M001210", "B001321", "C001135", "B001302", "G000565", "B001296",
    "F000472", "D000628", "C001116", "F000469", "M001224", "S001213",
    "H001077", "H001052", "B001301", "H001122", "J000289", "B001299",
    "P000605", "N000190", "B001325", "H001091", "D000616", "O000175",
    "S001209", "R000614", "G000599", "C001115", "C001118", "G000568",
    "T000165", "A000367", "B000755", "L000564", "P000592", "N000184",
    "L000583", "M001177", "R000587", "R000598", "S001183", "L000589",
    "P000609", "S001218", "C001140", "H001096", "B001297", "G000596",
    "D000626", "W000816", "B001294", "B001290", "C001111", "G000590",
    "G000580", "H001057", "H001087", "R000611", "R000409", "S001177",
    "B000213", "B001311", "B001283", "B001274", "B001305", "C001102",
    "D000621", "D000615", "F000451", "H001036", "K000395", "L000573",
    "L000571", "M000975", "M001195", "M000355", "N000181", "P000588",
    "P000602", "R000603", "S000018", "Y000065",
}

TEA_PARTY_BIOGUIDES = {
    "S001183", "M001177", "B001257", "S000250", "W000798", "S001172",
    "W000795", "C001051", "B001250", "R000582", "M001158", "Y000065",
    "K000362", "B000213", "B001273", "C000880", "F000458", "S001213",
    "B000755", "B001262", "C001075", "L000564", "F00040", "F000451",
    "G000551", "H000067", "L000569", "H001057", "J000290", "M001180",
    "N000182", "P000588", "P000592", "P000601", "R000592", "S000583",
    "B001248", "R000487", "P000591", "M000355", "D000615", "W000796",
    "G000589", "S001189",
}

POPULIST_CAUCUS_BIOGUIDES = TEA_PARTY_BIOGUIDES | FREEDOM_CAUCUS_BIOGUIDES


def load_legislators(current_path: Path, historical_path: Path) -> pd.DataFrame:
    current = pd.read_csv(current_path)
    historical = pd.read_csv(historical_path)

    combined = pd.concat([current, historical], ignore_index=True)

    required_columns = ["bioguide_id", "full_name", "party"]
    missing_columns = [col for col in required_columns if col not in combined.columns]
    if missing_columns:
        raise ValueError(f"Missing expected legislator columns: {missing_columns}")

    combined = combined[required_columns].dropna(subset=["bioguide_id"])
    combined = combined.drop_duplicates(subset="bioguide_id", keep="first")
    combined = combined.set_index("bioguide_id")

    return combined


def normalize_party(party: object) -> str:
    if pd.isna(party):
        return "Unknown"

    party = str(party).strip()

    if party in {"Democrat", "Democratic"}:
        return "Democrat"
    if party == "Republican":
        return "Republican"

    return party


def assign_caucus(bioguide: object, party: object, year: object) -> str:
    if pd.isna(bioguide):
        return "Other"

    party = normalize_party(party)

    try:
        year = int(year)
    except (TypeError, ValueError):
        return "Other"

    if party == "Democrat":
        return "Democrat"

    if party != "Republican":
        return "Other"

    if year < 2010:
        return "Establishment Republican"

    if 2010 <= year < 2015:
        return (
            "Tea Party / Freedom Caucus"
            if str(bioguide).strip() in TEA_PARTY_BIOGUIDES
            else "Establishment Republican"
        )

    return (
        "Tea Party / Freedom Caucus"
        if str(bioguide).strip() in POPULIST_CAUCUS_BIOGUIDES
        else "Establishment Republican"
    )


def lookup_party(bioguide: object, legislators_df: pd.DataFrame) -> str:
    if pd.isna(bioguide):
        return "Unknown"

    bioguide = str(bioguide).strip()
    if bioguide in legislators_df.index:
        return legislators_df.loc[bioguide, "party"]

    return "Unknown"


def tag_turn_file(input_path: Path, legislators_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    required_columns = ["speaker_bioguide", "year"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{input_path.name} is missing required columns: {missing_columns}")

    df["party"] = df["speaker_bioguide"].apply(lambda x: lookup_party(x, legislators_df))
    df["party"] = df["party"].apply(normalize_party)

    df["is_tea_party"] = df["speaker_bioguide"].apply(
        lambda x: int(pd.notna(x) and str(x).strip() in TEA_PARTY_BIOGUIDES)
    )
    df["is_freedom_caucus"] = df["speaker_bioguide"].apply(
        lambda x: int(pd.notna(x) and str(x).strip() in FREEDOM_CAUCUS_BIOGUIDES)
    )
    df["is_populist_republican"] = df["speaker_bioguide"].apply(
        lambda x: int(pd.notna(x) and str(x).strip() in POPULIST_CAUCUS_BIOGUIDES)
    )

    df["caucus"] = df.apply(
        lambda row: assign_caucus(
            bioguide=row["speaker_bioguide"],
            party=row["party"],
            year=row["year"],
        ),
        axis=1,
    )

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    print("Loading legislator data...")
    legislators = load_legislators(CURRENT_PATH, HISTORICAL_PATH)
    print(f"Loaded {len(legislators)} unique legislators")

    summary_rows = []

    for year in YEARS:
        input_file = INPUT_DIR / f"turns_{year}_House_clean.csv"
        output_file = OUTPUT_DIR / f"turns_{year}_House_tagged.csv"

        if not input_file.exists():
            print(f"\nSkipping missing file: {input_file}")
            continue

        print(f"\nProcessing {input_file.name}...")
        df = tag_turn_file(input_file, legislators, output_file)

        print(f"Saved: {output_file}")
        print(f"Total turns: {len(df)}")

        print("\nParty distribution:")
        print(df["party"].value_counts(dropna=False).to_string())

        print("\nCaucus distribution:")
        print(df["caucus"].value_counts(dropna=False).to_string())

        summary_rows.append(
            {
                "year": year,
                "rows": len(df),
                "unique_speakers": df["speaker_bioguide"].nunique(dropna=True),
                "democrats": (df["party"] == "Democrat").sum(),
                "republicans": (df["party"] == "Republican").sum(),
                "tea_party_flags": df["is_tea_party"].sum(),
                "freedom_caucus_flags": df["is_freedom_caucus"].sum(),
                "populist_republican_flags": df["is_populist_republican"].sum(),
            }
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = OUTPUT_DIR / "tagging_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print("\nDone.")
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
