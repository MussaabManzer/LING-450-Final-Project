# Scripts Overview

This repository follows a sequential pipeline. Each script performs a specific stage of the analysis, from raw Congressional Record data to final feature extraction.

---

## 1. preprocess_turns_all_years.py

**Purpose:**  
Convert raw Congressional Record JSON files into clean CSVs.

**What it does:**
- Reads JSON files from the `congressional-record` parser
- Keeps:
  - House speeches only
  - valid speakers (Bioguide IDs)
  - substantive speech turns
- Removes:
  - procedural filler (e.g., “I yield”, “the clerk will…”)
  - very short turns

**Input:**
- Parsed JSON files

**Output:**
- `turns_<year>_House_clean.csv`

---

## 2. tag_turns_party_caucus_all_years.py

**Purpose:**  
Add political metadata to each speech turn.

**What it does:**
- Maps speakers to party using Bioguide IDs
- Flags:
  - Tea Party membership
  - Freedom Caucus membership
  - populist Republican indicator
- Assigns time-aware caucus labels

**Input:**
- Clean CSVs
- Legislator files (`legislators-current.csv`, `legislators-historical.csv`)

**Output:**
- `turns_<year>_House_tagged.csv`
- `tagging_summary.csv`

---

## 3. run_pipeline.py

**Purpose:**  
Extract populism-related linguistic features.

### People-Centrism
- First-person plural pronoun ratio (`we`, `our`)
- Density of people-related terms (lexicon-based)

### Anti-Elitism
- Tier 1: explicit phrases (e.g., “corrupt elite”, “deep state”)
- Tier 2: elite terms + negative context (±10 token window)

**Input:**
- Tagged CSVs

**Output:**
- `features_master.csv`

---

## 4. roberta_scoring.py

**Purpose:**  
Test a pretrained transformer model for populism detection.

**What it does:**
- Loads `roberta-large-ft-trump-populism`
- Runs predictions on sample text
- Outputs logits and probabilities

**Notes:**
- Used to demonstrate domain mismatch (campaign vs congressional speech)

---

## 5. gpd_keyness.py

**Purpose:**  
Identify words associated with populist vs non-populist speech using the Global Populism Database.

**What it does:**
- Matches GPD speeches to text files
- Splits into high vs low populism groups
- Computes log-odds ratios for words

**Output:**
- `word_keyness_full.csv`
- `top_high_populism_words.csv`
- `top_low_populism_words.csv`
- `candidate_lexicon_words.csv`

---

## Pipeline Summary
