# Project: The Impunity Index
# Authors: Lindsay Gross, Shreya Mendi, Andrew Jin
# Advisor: Brinnae Bent, PhD
# Claude chat: https://claude.ai/chat/f8744002-3279-48ab-9d9a-8efa1fdb1af1
# Built with Claude AI assistance

"""
Build pairwise co-occurrence edges from the raw document corpus.

For each document, identifies which of the 66 known individuals are mentioned,
then creates weighted edges between every pair that co-occurs. This data powers
the D3.js network graph in the dashboard.

Uses simple string matching (no spaCy) for speed and reliability.
"""

import json
import logging
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent.parent / p


# Topics from the severity scraper that are not actual people
NON_PERSON_TOPICS = {
    "dentist", "gynecologist", "pregnant", "whoops",
    "beef jerky", "pizza", "cream cheese",
    "drugs", "bitcoin", "9/11", "zorro ranch",
    "baal and occult references", "israel and mossad",
    "dangene and jennie enterprise", "epstein suicide",
    "qatar", "lifetouch",
}


def load_person_names(scores_path: str) -> List[str]:
    """
    Load the list of known person names from the severity scores JSON.

    Args:
        scores_path: Path to epsteinoverview_scores.json

    Returns:
        List of person names (filtered to exclude non-person topics)
    """
    path = _resolve_path(scores_path)
    with open(path, 'r') as f:
        scores = json.load(f)

    names = []
    for entry in scores:
        name = entry['name']
        if name.lower().strip() not in NON_PERSON_TOPICS:
            names.append(name)

    logger.info(f"Loaded {len(names)} person names")
    return names


def build_search_variants(name: str) -> List[str]:
    """
    Build searchable name variants for a person.

    Handles compound names like "Oren, Alon, and Tal Alexander"
    and "George H.W. Bush and George W. Bush".

    Args:
        name: Full name string

    Returns:
        List of lowercase search variants
    """
    variants = set()
    variants.add(name.lower())

    # Handle "X and Y Z" or "X, Y, and Z LastName" patterns
    and_pattern = re.match(r'^(.+?)\s+and\s+(.+)$', name, re.IGNORECASE)
    if and_pattern:
        left_part = and_pattern.group(1)
        right_part = and_pattern.group(2)

        # Right part likely has the last name
        right_parts = right_part.strip().split()
        if len(right_parts) >= 2:
            last_name = right_parts[-1]
            # Add the full right name
            variants.add(right_part.strip().lower())

            # Split left by commas and "and"
            left_names = re.split(r',\s*|\s+and\s+', left_part)
            for ln in left_names:
                ln = ln.strip()
                if ln:
                    if last_name.lower() not in ln.lower():
                        variants.add(f"{ln} {last_name}".lower())
                    else:
                        variants.add(ln.lower())
        else:
            variants.add(right_part.strip().lower())
            variants.add(left_part.strip().lower())

    # For simple "First Last" names, add just the last name
    # (only if it's reasonably unique - skip very common ones)
    parts = name.split()
    if len(parts) == 2:
        last = parts[-1].lower()
        # Skip very short or common last names that cause false positives
        common_lasts = {'bush', 'black', 'gates', 'musk', 'wolf', 'lee'}
        if len(last) > 3 and last not in common_lasts:
            variants.add(last)

    # Always include the full first + last for 2-part names
    if len(parts) >= 2:
        variants.add(f"{parts[0]} {parts[-1]}".lower())

    return list(variants)


def load_documents(raw_data_dir: str) -> Dict[str, str]:
    """
    Load all raw documents and return mapping of doc_id -> text.

    Args:
        raw_data_dir: Path to raw data directory

    Returns:
        Dictionary mapping document ID to text content
    """
    raw_path = _resolve_path(raw_data_dir)
    documents = {}

    for json_file in sorted(raw_path.glob("ds*_agg.json")):
        logger.info(f"Loading {json_file.name}...")
        with open(json_file, 'r') as f:
            data = json.load(f)

        for doc_id, doc_data in data.items():
            if doc_data.get('success') and doc_data.get('text'):
                documents[doc_id] = doc_data['text']

    logger.info(f"Loaded {len(documents)} documents")
    return documents


def find_people_in_document(
    text: str,
    name_to_patterns: Dict[str, List[re.Pattern]]
) -> Set[str]:
    """
    Find which known people are mentioned in a document.

    Uses word-boundary regex matching to avoid substring false positives
    (e.g., "ting" matching "meeting").

    Args:
        text: Document text
        name_to_patterns: Mapping of canonical name to compiled regex patterns

    Returns:
        Set of canonical names found in the document
    """
    found = set()

    for canonical_name, patterns in name_to_patterns.items():
        for pattern in patterns:
            if pattern.search(text):
                found.add(canonical_name)
                break  # No need to check other patterns for this person

    return found


def build_edges(
    scores_path: str = "data/scraped/epsteinoverview_scores.json",
    raw_data_dir: str = "data/raw",
    output_path: str = "data/processed/edges.csv",
    min_weight: int = 1
) -> pd.DataFrame:
    """
    Build pairwise co-occurrence edges from the document corpus.

    For each document, finds which known people are mentioned, then
    creates edges between every pair that co-occurs.

    Args:
        scores_path: Path to severity scores JSON
        raw_data_dir: Path to raw document directory
        output_path: Path for output CSV
        min_weight: Minimum co-occurrence count to include an edge

    Returns:
        DataFrame with edge data
    """
    # Load names and build search variants with word-boundary regex
    names = load_person_names(scores_path)
    name_to_patterns = {}
    for name in names:
        variants = build_search_variants(name)
        # Compile word-boundary regex for each variant to avoid substring matches
        # (e.g., "ting" should NOT match "meeting")
        patterns = []
        for v in variants:
            escaped = re.escape(v)
            patterns.append(re.compile(r'\b' + escaped + r'\b', re.IGNORECASE))
        name_to_patterns[name] = patterns

    logger.info(f"Built search patterns for {len(name_to_patterns)} people")

    # Load documents
    documents = load_documents(raw_data_dir)

    # Count co-occurrences
    edge_counts = defaultdict(int)
    people_doc_count = defaultdict(int)

    for i, (doc_id, text) in enumerate(documents.items()):
        if i % 500 == 0:
            logger.info(f"Processing document {i}/{len(documents)}...")

        found_people = find_people_in_document(text, name_to_patterns)

        # Count individual appearances
        for person in found_people:
            people_doc_count[person] += 1

        # Count pairwise co-occurrences
        if len(found_people) >= 2:
            for p1, p2 in combinations(sorted(found_people), 2):
                edge_counts[(p1, p2)] += 1

    # Build edge DataFrame
    edges = []
    for (source, target), weight in edge_counts.items():
        if weight >= min_weight:
            edges.append({
                'source': source,
                'target': target,
                'weight': weight
            })

    edges_df = pd.DataFrame(edges)

    if not edges_df.empty:
        edges_df = edges_df.sort_values('weight', ascending=False)

    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"EDGE GENERATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Documents processed: {len(documents)}")
    logger.info(f"People found in docs: {len(people_doc_count)}")
    logger.info(f"Total edges (min_weight={min_weight}): {len(edges_df)}")

    if not edges_df.empty:
        logger.info(f"Max co-occurrence: {edges_df['weight'].max()}")
        logger.info(f"Mean co-occurrence: {edges_df['weight'].mean():.1f}")
        logger.info(f"Median co-occurrence: {edges_df['weight'].median():.1f}")
        logger.info(f"\nTop 10 strongest connections:")
        for _, row in edges_df.head(10).iterrows():
            logger.info(f"  {row['source']} <-> {row['target']}: {row['weight']} docs")

    # Save
    out_path = _resolve_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(out_path, index=False)
    logger.info(f"\nSaved {len(edges_df)} edges to {out_path}")

    return edges_df


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build pairwise co-occurrence edges from document corpus"
    )
    parser.add_argument(
        "--scores-path",
        default="data/scraped/epsteinoverview_scores.json",
        help="Path to severity scores JSON"
    )
    parser.add_argument(
        "--raw-data-dir",
        default="data/raw",
        help="Path to raw data directory"
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/edges.csv",
        help="Path for output CSV"
    )
    parser.add_argument(
        "--min-weight",
        type=int,
        default=1,
        help="Minimum co-occurrence count to include edge"
    )
    args = parser.parse_args()

    build_edges(
        scores_path=args.scores_path,
        raw_data_dir=args.raw_data_dir,
        output_path=args.output_path,
        min_weight=args.min_weight
    )


if __name__ == "__main__":
    main()
