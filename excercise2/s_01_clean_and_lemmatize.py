"""
==========================================
Text Cleaning and Tokenization Script
==========================================

This script processes folders of text files (e.g., British and US debates),
cleans and tokenizes them using spaCy, and writes the results into
corresponding "clean" and "lemma" folders.

Process summary:
1️⃣ Reads each text file in the given raw directory.
2️⃣ Extracts all text content from the file.
3️⃣ Tokenizes the text using spaCy.
4️⃣ Removes pure whitespace tokens.
5️⃣ Saves:
   - cleaned tokens to CLEAN_DIR/<corpus_name>/*.clean.txt
   - lemmatized tokens to LEMMA_DIR/<corpus_name>/*.lemma.txt
"""

import os
import glob
import spacy

from s_00_clean_classification_words import remove_classification_words


# =========================
# Configuration
# =========================

DATA_DIR = "data"  # Root folder that contains subfolders with raw text files

import os

# Base directory for this script file. This ensures running the script from
# the repo root (or any other CWD) still locates the bundled `data/` folder.
BASE_DIR = os.path.dirname(__file__)

BRITISH = "british_parliament_debates"
US = "US_congress_debates"

DATA_DIR = os.path.join(BASE_DIR, "data")

# Output dirs inside the exercise folder
CLEAN_DIR = os.path.join(BASE_DIR, "clean_docs")
LEMMA_DIR = os.path.join(BASE_DIR, "lemmas")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(LEMMA_DIR, exist_ok=True)

# Load English language model (tagger + lemmatizer kept, parser/NER disabled)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])


# =========================
# Helper functions
# =========================

def extract_text(file_path: str) -> str:
    """
    Extract text content from a text file.

    Args:
        file_path: Path to the text file.

    Returns:
        The file contents as a string.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def tokenize_and_clean(text: str):
    """
    Tokenize and clean the given text using spaCy.

    Args:
        text: The input text to tokenize.

    Returns:
        (tokens, lemmas):
            tokens: list of token texts (words and punctuation),
                    excluding pure whitespace tokens.
            lemmas: list of lemmatized tokens (lowercased),
                    with pronouns handled specially.
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

    The structure will be:
        clean_root/<corpus_name>/<filename>.clean.txt
        lemma_root/<corpus_name>/<filename>.lemma.txt

    Args:
        raw_dir: Directory with raw .txt files.
        clean_root: Root directory for cleaned files.
        lemma_root: Root directory for lemma files.
    """
    corpus_name = os.path.basename(os.path.normpath(raw_dir))
    clean_dir = os.path.join(clean_root, corpus_name)
    lemma_dir = os.path.join(lemma_root, corpus_name)
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(lemma_dir, exist_ok=True)

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

        corpus_lower = corpus_name.lower()
        if "congress" in corpus_lower or corpus_lower.startswith("us"):
            country = "us"
        else:
            country = "uk"
            
        print(f"Processing {text_path}...")

        text = extract_text(text_path)
        
        text = remove_classification_words(text,country)

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
