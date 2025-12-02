"""
==========================================
Text Cleaning and Tokenization Script
==========================================

This script processes folders of text files (e.g., British and US debates),
cleans and tokenizes them using spaCy, and writes the results into
corresponding "clean" and "lemma" folders.

Pipeline:
1. Read raw .txt.
2. Remove trivial classification signals:
   - Remove US/UK tokens.
   - For US Congressional files: strip headers per <pre> block and keep bodies.
3. Tokenize with spaCy.
4. Remove whitespace tokens.
5. Save:
   - tokens  → CLEAN_DIR/<corpus_name>/*.clean.txt
   - lemmas  → LEMMA_DIR/<corpus_name>/*.lemma.txt
"""

import os
import glob
import spacy

from s_00_clean_classification_words import remove_classification_words

# =========================
# Configuration
# =========================

DATA_DIR = "data"  # Root folder that contains subfolders with raw text files

BRITISH = "british_parliament_debates"
US = "US_congress_debates"

CLEAN_DIR = "clean_docs"
LEMMA_DIR = "lemmas"

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(LEMMA_DIR, exist_ok=True)

# Load English language model (tagger + lemmatizer kept, parser/NER disabled)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])


# =========================
# Helper functions
# =========================

def extract_text(file_path: str) -> str:
    """Extract text content from a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def tokenize_and_clean(text: str):
    """
    Tokenize and clean the given text using spaCy.

    Returns:
        tokens: list of token texts (words and punctuation), excluding pure whitespace tokens.
        lemmas: list of lemmatized tokens (lowercased), with pronouns handled specially.
    """
    doc = nlp(text)
    tokens = []
    lemmas = []

    for t in doc:
        if t.is_space:
            continue

        tokens.append(t.text)

        lemma = t.lemma_.lower()
        # Handle special case for pronouns in older spaCy models ("-PRON-")
        if lemma == "-pron-":
            lemma = t.text.lower()

        lemmas.append(lemma)

    return tokens, lemmas


def build_clean_and_lemmatized_files(
    raw_dir: str,
    clean_root: str = CLEAN_DIR,
    lemma_root: str = LEMMA_DIR,
):
    """
    Process all text files in raw_dir and create cleaned / lemmatized files.

    Output structure:
      clean_root/<corpus_name>/<filename>.clean.txt
      lemma_root/<corpus_name>/<filename>.lemma.txt
    """
    corpus_name = os.path.basename(os.path.normpath(raw_dir))
    clean_dir = os.path.join(clean_root, corpus_name)
    lemma_dir = os.path.join(lemma_root, corpus_name)
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(lemma_dir, exist_ok=True)

    # Decide which country logic to apply for classification cleaning
    if corpus_name == US:
        country = "us"
    elif corpus_name == BRITISH:
        country = "uk"
    else:
        country = ""

    processed = 0
    skipped = 0

    pattern = os.path.join(raw_dir, "*.txt")
    for text_path in sorted(glob.glob(pattern)):
        base = os.path.basename(text_path)
        clean_path = os.path.join(clean_dir, base + ".clean.txt")
        lemma_path = os.path.join(lemma_dir, base + ".lemma.txt")

        # Skip if this file was already processed
        if os.path.exists(clean_path) and os.path.exists(lemma_path):
            skipped += 1
            continue

        print(f"Processing {text_path}...")

        text = extract_text(text_path)
        text = remove_classification_words(text, country=country)

        tokens, lemmas = tokenize_and_clean(text)

        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(" ".join(tokens))

        with open(lemma_path, "w", encoding="utf-8") as f:
            f.write(" ".join(lemmas))

        processed += 1

    print(
        f"[{corpus_name}] Done. "
        f"Processed: {processed} files | Skipped (already existed): {skipped}"
    )


# =========================
# Main
# =========================

if __name__ == "__main__":
    # British debates
    build_clean_and_lemmatized_files(
        raw_dir=os.path.join(DATA_DIR, BRITISH)
    )

    # US debates
    build_clean_and_lemmatized_files(
        raw_dir=os.path.join(DATA_DIR, US)
    )

    print("All corpora cleaned and saved.")
