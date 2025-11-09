"""
==========================================
XML Cleaning and Tokenization Script
==========================================

This script processes a folder of XML files, extracts all text content,
cleans and tokenizes it using spaCy, and writes the results into a new folder.

Process summary:
1️⃣ Reads each XML file in RAW_DIR.
2️⃣ Extracts all inner text (ignores XML tags).
3️⃣ Tokenizes the text into words and punctuation.
4️⃣ Removes extra spaces.
5️⃣ Saves each cleaned text as a separate .clean.txt file in CLEAN_DIR.

"""

import os
import glob
import spacy
from lxml import etree

RAW_DIR = "docs"     # Folder with the raw XML files
CLEAN_DIR = "clean_docs"    # Folder to save cleaned text files
SAMPLE_RAW_DIR = "sample_docs"    # Folder with sample XML files
SAMPLE_CLEAN_DIR = "sample_clean"  # Folder to save sample cleaned text files

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(SAMPLE_CLEAN_DIR, exist_ok=True)

# Load English language model (only tokenizer enabled)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])


def extract_text_from_xml(path):
    """
    Extract all textual content from an XML file.

    Args:
        path (str): Path to the XML file.

    Returns:
        str: A single string containing all the text inside the XML,
             concatenated and stripped of leading/trailing spaces.
    """
    tree = etree.parse(path)
    texts = tree.xpath("//text()")  # Find all text nodes in the XML
    return " ".join(t.strip() for t in texts if t.strip())


def tokenize_and_clean(text):
    """
    Tokenize and clean the given text using spaCy.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list[str]: A list of tokens (words and punctuation marks),
                   excluding pure whitespace tokens.
    """
    doc = nlp(text)
    tokens = []
    for t in doc:
        if t.is_space:  # Skip whitespace tokens
            continue
        tokens.append(t.text)
    return tokens


def build_clean_files():
    """
    Process all XML files in RAW_DIR and create cleaned .txt files in CLEAN_DIR.

    For each XML file:
      - Extracts its text.
      - Tokenizes and cleans it.
      - Saves the cleaned text into CLEAN_DIR with the same base name
        plus ".clean.txt" extension.
    """
    for xml_path in sorted(glob.glob(os.path.join(RAW_DIR, "*.xml"))):
        print(f"Processing {xml_path}...")
        base = os.path.basename(xml_path)
        clean_path = os.path.join(CLEAN_DIR, base + ".clean.txt")

        # Skip if this file was already processed
        if os.path.exists(clean_path):
            continue

        # Extract and clean the text
        text = extract_text_from_xml(xml_path)
        tokens = tokenize_and_clean(text)

        # Save tokens joined by spaces
        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(" ".join(tokens))


if __name__ == "__main__":
    build_clean_files()
    print("All files cleaned and saved.")
