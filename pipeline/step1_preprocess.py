"""
pipeline/step1_preprocess.py

Reads raw 20 Newsgroups dataset, cleans every document,
saves structured corpus to data/processed/clean_corpus.json.

Run:
    python pipeline/step1_preprocess.py
"""

import os
import json
import logging
from pathlib import Path
from tqdm import tqdm

from app.utils.preprocessing import clean_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw/20_newsgroups")
OUTPUT_PATH = Path("data/processed/clean_corpus.json")


def load_and_clean_dataset() -> list:
    """
    Walks the 20_newsgroups directory structure, cleans each file,
    returns a list of document dicts.

    Directory structure:
        20_newsgroups/
            alt.atheism/
                49960
                51060
            comp.graphics/
                37261
            ...
    """
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_DIR}. "
            "Place the extracted 20_newsgroups folder under data/raw/"
        )

    documents = []
    doc_id = 0
    skipped = 0

    categories = sorted(os.listdir(RAW_DATA_DIR))
    logger.info(f"Found {len(categories)} categories")

    for category in tqdm(categories, desc="Processing categories"):
        cat_path = RAW_DATA_DIR / category
        if not cat_path.is_dir():
            continue

        files = sorted(os.listdir(cat_path))
        for filename in files:
            file_path = cat_path / filename

            try:
                # Some files have encoding issues — latin-1 covers most
                with open(file_path, "r", encoding="latin-1") as f:
                    raw_text = f.read()
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
                skipped += 1
                continue

            cleaned = clean_document(raw_text)

            if cleaned is None:
                # Document was too short after cleaning — discard
                skipped += 1
                continue

            documents.append({
                "id": doc_id,
                "text": cleaned,
                "label": category,
                "original_length": len(raw_text),
                "cleaned_length": len(cleaned)
            })
            doc_id += 1

    return documents, skipped


def main():
    logger.info("=== Step 1: Preprocessing ===")

    documents, skipped = load_and_clean_dataset()

    logger.info(f"Documents retained: {len(documents)}")
    logger.info(f"Documents discarded: {skipped}")

    avg_reduction = sum(
        1 - d["cleaned_length"] / d["original_length"]
        for d in documents
    ) / len(documents)
    logger.info(f"Average noise reduction: {avg_reduction:.1%}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(documents, f, indent=2)

    logger.info(f"Saved to {OUTPUT_PATH}")
    logger.info("=== Step 1 complete ===")


if __name__ == "__main__":
    main()
