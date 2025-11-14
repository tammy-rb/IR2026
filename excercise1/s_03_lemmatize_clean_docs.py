"""
===========================================
 Lemmatization Script – British Parliament Corpus
===========================================

This script reads all pre-cleaned text files (in the folder "clean")
and creates, for each file, a corresponding lemma file in "lemmas".

Each lemma file contains the base form (lemma) of every word, filtered to include
only alphabetic tokens of length ≥ 2. Punctuation, numbers, and whitespace are excluded.

Tool used:
    spaCy "en_core_web_sm" model for lemmatization.
"""

import os
import glob
import spacy

CLEAN_DIR = "clean_docs"
LEMMA_DIR = "lemmas"

os.makedirs(LEMMA_DIR, exist_ok=True)

# Load spaCy English model
# Disabled components for faster performance – we only need tokenizer + lemmatizer
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])


def build_lemma_files():
    """
    Build lemma files for all cleaned text files.

    For each file in CLEAN_DIR:
        1. Read the text.
        2. Process it with spaCy.
        3. Extract lemmatized words (base forms).
        4. Save them to a new file in LEMMA_DIR with the same base name.
    """
    for clean_path in sorted(glob.glob(os.path.join(CLEAN_DIR, "*.clean.txt"))):
        print(f"Processing {clean_path}...")
        base = os.path.basename(clean_path).replace(".clean.txt", "")
        lemma_path = os.path.join(LEMMA_DIR, base + ".lemma.txt")
        if os.path.exists(lemma_path):
            continue
        with open(clean_path, encoding="utf-8") as f:
            text = f.read()
        # Run spaCy NLP pipeline (tokenization + lemmatization)
        doc = nlp(text)
        lemmas = []
        for t in doc:
            if t.is_space or t.is_punct:
                continue
            token = t.lemma_.lower()
            # Handle special case for pronouns, replace "-PRON-" with the original token
            if token == "-pron-":
                token = t.lower_
            if token.isalpha() and len(token) >= 2:
                lemmas.append(token)

        with open(lemma_path, "w", encoding="utf-8") as f:
            f.write(" ".join(lemmas))


if __name__ == "__main__":
    build_lemma_files()
    print("All lemma files created and saved.")