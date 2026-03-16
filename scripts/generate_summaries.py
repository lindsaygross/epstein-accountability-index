# Project: The Impunity Index
# Authors: Lindsay Gross, Shreya Mendi, Andrew Jin
# Advisor: Brinnae Bent, PhD
# Claude chat: https://claude.ai/chat/f8744002-3279-48ab-9d9a-8efa1fdb1af1
# Built with Claude AI assistance

"""
Generate structured summaries for each individual with DOJ file citations.

Summaries are template-based and extractive, ensuring every claim is
directly traceable to a source document.

Output: data/processed/summaries.json
"""

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize VADER once
try:
    sid = SentimentIntensityAnalyzer()
except LookupError:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    sid = SentimentIntensityAnalyzer()

NON_PERSON_TOPICS = {
    "Dentist", "Pilot", "Pizza", "McDonald's", "Bitcoin",
    "Lolita Express", "Zorro Ranch", "Epstein Island",
    "Little St. James", "Bear Stearns", "Deutsche Bank",
    "JP Morgan", "Victoria's Secret", "MIT Media Lab",
    "Harvard University", "Ohio State", "Dalton School"
}


def load_person_names(scores_path: str) -> List[str]:
    with open(scores_path) as f:
        scores = json.load(f)
    return [s["name"] for s in scores if s["name"] not in NON_PERSON_TOPICS]


def build_search_variants(name: str) -> List[str]:
    variants = [name.lower()]
    if " and " in name:
        parts = re.split(r',\s*|\s+and\s+', name)
        last_word = name.split()[-1]
        for part in parts:
            part = part.strip()
            if part:
                if ' ' not in part and part != last_word:
                    variants.append(f"{part} {last_word}".lower())
                else:
                    variants.append(part.lower())
    if name == "RFK":
        variants.extend(["robert f. kennedy", "rfk"])
    elif name == "Oprah":
        variants.append("oprah winfrey")
    elif "Justice " in name:
        variants.append(name.replace("Justice ", "").lower())
    return list(set(variants))


def extract_sentences_around_name(text: str, name: str, max_snippets: int = 8) -> List[str]:
    """Extract sentences containing or near the person's name."""
    variants = build_search_variants(name)

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return []

    # Find sentences containing any variant
    matches = []
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        for v in variants:
            if v in sent_lower:
                # Get this sentence + neighbors for context
                start = max(0, i - 1)
                end = min(len(sentences), i + 2)
                context = ' '.join(sentences[start:end])
                if len(context) > 50 and len(context) < 1000:
                    matches.append(context)
                break

    # Deduplicate and score by sentiment (more negative = more interesting)
    unique = list(set(matches))
    scored = []
    for snippet in unique:
        score = sid.polarity_scores(snippet)['compound']
        scored.append((snippet, score))

    # Sort by most negative first (more incriminating)
    scored.sort(key=lambda x: x[1])
    return [s[0] for s in scored[:max_snippets]]


def classify_doc_type(text: str) -> str:
    """Classify document type from content."""
    text_lower = text[:2000].lower()
    if 'from:' in text_lower and ('to:' in text_lower or 'subject:' in text_lower):
        return 'email'
    elif 'deposition' in text_lower or 'q.' in text_lower and 'a.' in text_lower:
        return 'deposition'
    elif 'flight' in text_lower and ('manifest' in text_lower or 'passenger' in text_lower):
        return 'flight_log'
    elif 'affidavit' in text_lower:
        return 'affidavit'
    elif 'indictment' in text_lower or 'complaint' in text_lower:
        return 'legal_filing'
    else:
        return 'document'


def generate_all_summaries(
    raw_data_dir: str = "data/raw",
    scores_path: str = "data/scraped/epsteinoverview_scores.json",
    consequences_path: str = "data/processed/consequences.csv",
    edges_path: str = "data/processed/edges.csv",
    output_path: str = "data/processed/summaries.json",
) -> Dict:
    """Generate summaries for all 66 people."""
    base = Path(__file__).parent.parent

    # Load person names
    names = load_person_names(base / scores_path)
    logger.info(f"Generating summaries for {len(names)} people")

    # Load consequences
    cons_df = pd.read_csv(base / consequences_path)
    cons_map = dict(zip(cons_df['name'], cons_df['consequence_description']))

    # Load edges for connections
    edges_df = pd.read_csv(base / edges_path) if (base / edges_path).exists() else pd.DataFrame()

    # Load all raw documents
    documents = []
    raw_dir = base / raw_data_dir
    for json_file in sorted(raw_dir.glob("ds*_agg.json")):
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

    logger.info(f"Loaded {len(documents)} raw documents")

    # Also load jmail cached documents if available
    jmail_dir = base / "data" / "jmail_cache"
    if jmail_dir.exists():
        for pq_file in jmail_dir.glob("documents-full-*.parquet"):
            try:
                df = pd.read_parquet(pq_file)
                text_col = None
                for col in ['text', 'body', 'content', 'ocr_text']:
                    if col in df.columns:
                        text_col = col
                        break
                if text_col:
                    id_col = 'id' if 'id' in df.columns else None
                    for idx, row in df.iterrows():
                        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                        if len(text) > 100:
                            doc_id = str(row[id_col]) if id_col else f"{pq_file.stem}_{idx}"
                            documents.append({
                                "doc_id": doc_id,
                                "text": text[:50000],
                                "source": pq_file.stem,
                            })
            except Exception as e:
                logger.warning(f"Error loading {pq_file}: {e}")

        logger.info(f"Total documents with jmail: {len(documents)}")

    # Generate summaries
    summaries = {}

    for i, name in enumerate(names):
        logger.info(f"[{i+1}/{len(names)}] {name}")
        variants = build_search_variants(name)

        # Find all documents mentioning this person
        person_docs = []
        for doc in documents:
            text_lower = doc["text"].lower()
            for v in variants:
                if v in text_lower:
                    person_docs.append(doc)
                    break

        total_docs = len(person_docs)

        if total_docs == 0:
            summaries[name] = {
                "total_documents": 0,
                "summary_text": f"{name} was not found in the available case documents.",
                "citations": [],
                "generated_at": datetime.utcnow().isoformat() + "Z",
            }
            continue

        # Classify document types
        doc_types = Counter()
        for doc in person_docs:
            dtype = classify_doc_type(doc["text"])
            doc_types[dtype] += 1

        # Get top connections
        connections = []
        if not edges_df.empty:
            edges_person = edges_df[
                (edges_df['source'] == name) | (edges_df['target'] == name)
            ].sort_values('weight', ascending=False).head(5)
            for _, row in edges_person.iterrows():
                other = row['target'] if row['source'] == name else row['source']
                connections.append(other)

        # Extract key snippets
        all_snippets = []
        for doc in person_docs[:50]:  # Cap to avoid slowness
            snippets = extract_sentences_around_name(doc["text"], name, max_snippets=3)
            for s in snippets:
                all_snippets.append({
                    "snippet": s,
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "doc_type": classify_doc_type(doc["text"]),
                })

        # Build citations
        citations = []
        seen_snippets = set()
        for item in all_snippets[:15]:
            snippet_key = item["snippet"][:100]
            if snippet_key in seen_snippets:
                continue
            seen_snippets.add(snippet_key)

            # Build jmail URL if it looks like a jmail doc
            jmail_url = ""
            doc_id = item["doc_id"]
            if doc_id.startswith("documents-full"):
                jmail_url = f"https://jmail.world/drive"

            citations.append({
                "doc_id": doc_id,
                "doc_type": item["doc_type"],
                "source_volume": item["source"],
                "snippet": item["snippet"][:500],
                "jmail_url": jmail_url,
            })

        # Build summary text
        doc_type_parts = []
        for dtype, count in doc_types.most_common():
            doc_type_parts.append(f"{count} {dtype}{'s' if count > 1 else ''}")
        doc_types_str = ", ".join(doc_type_parts)

        connection_str = ""
        if connections:
            connection_str = f" They are most frequently mentioned alongside {', '.join(connections[:3])}."

        consequence_str = ""
        if name in cons_map and cons_map[name]:
            consequence_str = f" {cons_map[name]}"

        summary_text = (
            f"{name} appears in {total_docs} documents across the Epstein case files, "
            f"including {doc_types_str}.{connection_str}{consequence_str}"
        )

        summaries[name] = {
            "total_documents": total_docs,
            "document_types": dict(doc_types),
            "key_connections": connections,
            "summary_text": summary_text,
            "citations": citations[:10],
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    # Save
    out = base / output_path
    with open(out, 'w') as f:
        json.dump(summaries, f, indent=2)

    non_empty = sum(1 for v in summaries.values() if v.get("total_documents", 0) > 0)
    logger.info(f"\nDone! {non_empty}/{len(names)} people have document summaries")
    logger.info(f"Saved to {out}")

    return summaries


if __name__ == "__main__":
    generate_all_summaries()
