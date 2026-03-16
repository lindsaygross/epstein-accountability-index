# Project: The Impunity Index
# Authors: Lindsay Gross, Shreya Mendi, Andrew Jin
# Advisor: Brinnae Bent, PhD
# Claude chat: https://claude.ai/chat/f8744002-3279-48ab-9d9a-8efa1fdb1af1
# Built with Claude AI assistance

"""
Unified data loader for text-based model training.

Combines local raw documents (data/raw/ds*_agg.json) with jmail.world
cached data to build:
1. Per-person document corpora (for TF-IDF + Random Forest)
2. Per-document labeled samples (for Legal-BERT fine-tuning)
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Non-person topics
NON_PERSON_TOPICS = {
    "Dentist", "Pilot", "Pizza", "McDonald's", "Bitcoin",
    "Lolita Express", "Zorro Ranch", "Epstein Island",
    "Little St. James", "Bear Stearns", "Deutsche Bank",
    "JP Morgan", "Victoria's Secret", "MIT Media Lab",
    "Harvard University", "Ohio State", "Dalton School"
}

# Common surnames that cause false positives in substring matching
COMMON_SURNAMES = {"black", "bush", "gates", "young", "wolf", "long", "grant"}


def load_person_names(scores_path: str = "data/scraped/epsteinoverview_scores.json") -> List[str]:
    """Load the 66 tracked person names."""
    base = Path(__file__).parent.parent
    with open(base / scores_path) as f:
        scores = json.load(f)
    return [s["name"] for s in scores if s["name"] not in NON_PERSON_TOPICS]


def build_search_variants(name: str) -> List[str]:
    """Build regex-safe search variants for a person name."""
    variants = [name.lower()]

    if " and " in name:
        parts = re.split(r',\s*|\s+and\s+', name)
        last_word = name.split()[-1]
        for part in parts:
            part = part.strip()
            if part:
                if ' ' not in part and part != last_word:
                    full = f"{part} {last_word}"
                    if full.lower().split()[-1] not in COMMON_SURNAMES or len(full.split()) > 1:
                        variants.append(full.lower())
                else:
                    variants.append(part.lower())

    if name == "RFK":
        variants.extend(["robert f. kennedy", "rfk"])
    elif name == "Oprah":
        variants.append("oprah winfrey")
    elif "Justice " in name:
        variants.append(name.replace("Justice ", "").lower())
    elif "George H.W." in name:
        variants.extend(["george h.w. bush", "george w. bush", "george bush"])

    return list(set(variants))


def build_regex_patterns(variants: List[str]) -> List[re.Pattern]:
    """Build word-boundary regex patterns from name variants."""
    patterns = []
    for v in variants:
        escaped = re.escape(v)
        try:
            patterns.append(re.compile(r'\b' + escaped + r'\b', re.IGNORECASE))
        except re.error:
            continue
    return patterns


def load_local_documents(raw_data_dir: str = "data/raw") -> List[Dict]:
    """
    Load all local raw documents from data/raw/ds*_agg.json files.
    Returns list of dicts with 'doc_id', 'text', 'source' keys.
    """
    base = Path(__file__).parent.parent
    raw_dir = base / raw_data_dir
    documents = []

    if not raw_dir.exists():
        logger.warning(f"Raw data dir not found: {raw_dir}")
        return documents

    for json_file in sorted(raw_dir.glob("ds*_agg.json")):
        logger.info(f"Loading {json_file.name}...")
        with open(json_file) as f:
            data = json.load(f)

        if isinstance(data, list):
            for i, doc in enumerate(data):
                text = doc.get("text", "")
                if text and doc.get("success", True):
                    documents.append({
                        "doc_id": f"{json_file.stem}_{i}",
                        "text": text,
                        "source": json_file.stem,
                    })
        elif isinstance(data, dict):
            for key, doc in data.items():
                text = doc.get("text", "") if isinstance(doc, dict) else ""
                if text:
                    documents.append({
                        "doc_id": f"{json_file.stem}_{key}",
                        "text": text,
                        "source": json_file.stem,
                    })

    logger.info(f"Loaded {len(documents)} local documents")
    return documents


def load_jmail_documents(cache_dir: str = "data/jmail_cache") -> List[Dict]:
    """
    Load cached jmail.world documents (Parquet files).
    Returns list of dicts with 'doc_id', 'text', 'source' keys.
    """
    base = Path(__file__).parent.parent
    cache = base / cache_dir
    documents = []

    if not cache.exists():
        logger.info("No jmail cache found, skipping")
        return documents

    for pq_file in cache.glob("documents-full-*.parquet"):
        try:
            df = pd.read_parquet(pq_file)
            text_col = None
            for col in ['text', 'body', 'content', 'ocr_text']:
                if col in df.columns:
                    text_col = col
                    break
            if text_col is None:
                continue

            id_col = 'id' if 'id' in df.columns else None

            for idx, row in df.iterrows():
                text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                if len(text) > 50:  # Skip very short/empty docs
                    doc_id = str(row[id_col]) if id_col else f"{pq_file.stem}_{idx}"
                    documents.append({
                        "doc_id": doc_id,
                        "text": text[:100000],  # Cap at 100K chars per doc
                        "source": pq_file.stem,
                    })
        except Exception as e:
            logger.warning(f"Error loading {pq_file.name}: {e}")

    logger.info(f"Loaded {len(documents)} jmail documents")
    return documents


def load_all_documents(
    raw_data_dir: str = "data/raw",
    jmail_cache_dir: str = "data/jmail_cache",
    use_jmail: bool = True,
) -> List[Dict]:
    """Load all available documents from both local and jmail.world sources."""
    docs = load_local_documents(raw_data_dir)
    if use_jmail:
        docs.extend(load_jmail_documents(jmail_cache_dir))
    logger.info(f"Total documents available: {len(docs)}")
    return docs


def build_person_corpora(
    documents: List[Dict],
    names: List[str],
    max_chars_per_person: int = 50000,
) -> Dict[str, str]:
    """
    For each person, concatenate all documents mentioning them.

    Returns dict mapping person_name -> concatenated text (capped).
    """
    logger.info(f"Building per-person corpora for {len(names)} people...")

    # Pre-build patterns
    name_patterns = {}
    for name in names:
        variants = build_search_variants(name)
        patterns = build_regex_patterns(variants)
        name_patterns[name] = patterns

    corpora: Dict[str, List[str]] = {name: [] for name in names}

    for i, doc in enumerate(documents):
        if i % 500 == 0:
            logger.info(f"  Processing doc {i}/{len(documents)}...")

        text = doc["text"]
        text_lower = text.lower()

        for name, patterns in name_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    corpora[name].append(text)
                    break

    # Concatenate and cap
    result = {}
    for name in names:
        texts = corpora[name]
        if texts:
            combined = "\n\n---\n\n".join(texts)
            result[name] = combined[:max_chars_per_person]
        else:
            result[name] = ""

    non_empty = sum(1 for v in result.values() if v)
    logger.info(f"Built corpora: {non_empty}/{len(names)} people have documents")
    return result


def build_document_level_dataset(
    documents: List[Dict],
    names: List[str],
    consequences_path: str = "data/processed/consequences.csv",
    max_window: int = 512,
) -> pd.DataFrame:
    """
    Build per-document classification dataset for Legal-BERT.

    For each document, find the most prominently mentioned known person,
    label with their consequence class, and extract a text window.

    Returns DataFrame with columns: [doc_id, person_name, text_window, label]
    """
    base = Path(__file__).parent.parent
    cons_df = pd.read_csv(base / consequences_path)
    cons_map = dict(zip(cons_df['name'], cons_df['consequence_tier']))

    # Convert to binary
    binary_map = {name: (1 if tier > 0 else 0) for name, tier in cons_map.items()}

    logger.info(f"Building document-level dataset from {len(documents)} docs...")

    # Pre-build patterns
    name_patterns = {}
    for name in names:
        variants = build_search_variants(name)
        patterns = build_regex_patterns(variants)
        name_patterns[name] = patterns

    samples = []

    for i, doc in enumerate(documents):
        if i % 500 == 0 and i > 0:
            logger.info(f"  Processing doc {i}/{len(documents)}...")

        text = doc["text"]
        if len(text) < 100:
            continue

        # Find all mentioned people and their mention counts
        mentions = {}
        for name, patterns in name_patterns.items():
            count = 0
            for pattern in patterns:
                count += len(pattern.findall(text))
            if count > 0:
                mentions[name] = count

        if not mentions:
            continue

        # For each mentioned person, create a training sample
        for person_name, count in mentions.items():
            if person_name not in binary_map:
                continue

            label = binary_map[person_name]

            # Extract text window around first mention
            window = extract_text_window(text, person_name, max_window * 4)  # chars not tokens

            if len(window) < 50:
                continue

            samples.append({
                "doc_id": doc["doc_id"],
                "person_name": person_name,
                "text_window": window,
                "label": label,
                "mention_count": count,
            })

    df = pd.DataFrame(samples)
    logger.info(f"Built {len(df)} document-level samples")
    if len(df) > 0:
        logger.info(f"  Label distribution: {df['label'].value_counts().to_dict()}")
        logger.info(f"  Unique people: {df['person_name'].nunique()}")

    return df


def extract_text_window(text: str, name: str, max_chars: int = 2048) -> str:
    """Extract a window of text around the first mention of a name."""
    variants = build_search_variants(name)

    for variant in variants:
        idx = text.lower().find(variant.lower())
        if idx >= 0:
            half = max_chars // 2
            start = max(0, idx - half)
            end = min(len(text), idx + len(variant) + half)
            return text[start:end].strip()

    # Fallback: return start of document
    return text[:max_chars].strip()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build training datasets")
    parser.add_argument("--use-jmail", action="store_true",
                        help="Include jmail.world cached data")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Output directory")
    args = parser.parse_args()

    names = load_person_names()
    documents = load_all_documents(use_jmail=args.use_jmail)

    # Build and save per-person corpora
    corpora = build_person_corpora(documents, names)
    base = Path(__file__).parent.parent
    out_dir = base / args.output_dir
    with open(out_dir / "person_corpora.json", 'w') as f:
        json.dump(corpora, f)
    logger.info(f"Saved person corpora to {out_dir / 'person_corpora.json'}")

    # Build and save document-level dataset
    doc_dataset = build_document_level_dataset(documents, names)
    doc_dataset.to_csv(out_dir / "doc_level_dataset.csv", index=False)
    logger.info(f"Saved doc-level dataset to {out_dir / 'doc_level_dataset.csv'}")
