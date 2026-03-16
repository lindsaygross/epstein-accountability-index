# Project: The Impunity Index
# Authors: Lindsay Gross, Shreya Mendi, Andrew Jin
# Advisor: Brinnae Bent, PhD
# Claude chat: https://claude.ai/chat/f8744002-3279-48ab-9d9a-8efa1fdb1af1
# Built with Claude AI assistance

"""
Build feature matrix from document corpus using NER and NLP techniques.

This script extracts named entities from Epstein case files and computes
various features for each individual mentioned.
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import nltk
import pandas as pd
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from rapidfuzz import fuzz
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Keywords for co-occurrence scoring
INCRIMINATING_KEYWORDS = [
    "minor", "minors", "young", "girl", "massage", "payment",
    "wire", "flight", "island", "recruit", "underage", "sex"
]


class FeatureExtractor:
    """Extract NLP features from document corpus."""

    def __init__(self):
        """Initialize the feature extractor with spaCy and VADER."""
        logger.info("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        logger.info("Initializing VADER sentiment analyzer...")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def extract_persons(self, text: str) -> Set[str]:
        """
        Extract person names from text using spaCy NER.

        Args:
            text: Document text

        Returns:
            Set of person names
        """
        doc = self.nlp(text[:1000000])  # Limit text length for memory
        persons = {ent.text for ent in doc.ents if ent.label_ == "PERSON"}
        return persons

    def get_sentence_context(self, text: str, name: str, window: int = 2) -> List[str]:
        """
        Extract sentences around mentions of a name.

        Args:
            text: Document text
            name: Person name to search for
            window: Number of sentences before and after

        Returns:
            List of context strings
        """
        sentences = nltk.sent_tokenize(text)
        contexts = []

        for i, sent in enumerate(sentences):
            if name.lower() in sent.lower():
                start = max(0, i - window)
                end = min(len(sentences), i + window + 1)
                context = " ".join(sentences[start:end])
                contexts.append(context)

        return contexts

    def compute_sentiment(self, text: str) -> float:
        """
        Compute sentiment score using VADER.

        Args:
            text: Text to analyze

        Returns:
            Compound sentiment score (-1 to 1)
        """
        scores = self.sentiment_analyzer.polarity_scores(text)
        return scores['compound']

    def check_subject_line(self, text: str, name: str) -> bool:
        """
        Check if name appears in email subject lines.

        Args:
            text: Document text
            name: Person name

        Returns:
            True if name appears in subject line
        """
        # Simple pattern matching for email subjects
        subject_pattern = r'Subject:.*?(?:\n|$)'
        subjects = re.findall(subject_pattern, text, re.IGNORECASE)

        for subject in subjects:
            if name.lower() in subject.lower():
                return True

        return False

    def compute_cooccurrence_score(self, text: str, name: str) -> int:
        """
        Count co-occurrences of name with incriminating keywords.

        Args:
            text: Document text
            name: Person name

        Returns:
            Co-occurrence count
        """
        text_lower = text.lower()
        name_lower = name.lower()

        # Find all positions of the name
        name_positions = [m.start() for m in re.finditer(re.escape(name_lower), text_lower)]

        cooccurrence_count = 0

        for pos in name_positions:
            # Check 100 characters around each mention
            context_start = max(0, pos - 100)
            context_end = min(len(text), pos + 100)
            context = text_lower[context_start:context_end]

            # Check for keywords in context
            for keyword in INCRIMINATING_KEYWORDS:
                if keyword in context:
                    cooccurrence_count += 1

        return cooccurrence_count

    def infer_doc_type(self, text: str) -> str:
        """
        Infer document type from text patterns.

        Args:
            text: Document text

        Returns:
            Document type string
        """
        text_lower = text.lower()

        if "subject:" in text_lower and ("from:" in text_lower or "to:" in text_lower):
            return "email"
        elif "flight" in text_lower and "passenger" in text_lower:
            return "flight_log"
        elif "deposition" in text_lower or "testimony" in text_lower:
            return "deposition"
        elif "affidavit" in text_lower:
            return "affidavit"
        else:
            return "other"


def load_document_corpus(raw_data_dir: str) -> Dict[str, Dict]:
    """
    Load all aggregated JSON files from raw data directory.

    Args:
        raw_data_dir: Directory containing ds*_agg.json files

    Returns:
        Dictionary mapping doc_id to document data
    """
    raw_path = Path(raw_data_dir)
    if not raw_path.is_absolute():
        raw_path = Path(__file__).resolve().parent.parent / raw_path

    logger.info(f"Loading document corpus from {raw_path}")
    corpus = {}

    json_files = list(raw_path.glob("ds*_agg.json"))

    logger.info(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        logger.info(f"Loading {json_file.name}")
        with open(json_file, 'r') as f:
            data = json.load(f)
            corpus.update(data)

    logger.info(f"Loaded {len(corpus)} documents")
    return corpus


def normalize_name(name: str) -> str:
    """
    Normalize person name for matching.

    Args:
        name: Person name

    Returns:
        Normalized name
    """
    # Remove titles, normalize whitespace
    name = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof)\.?\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def fuzzy_match_names(name1: str, name2: str, threshold: int = 85) -> bool:
    """
    Fuzzy match two names.

    Args:
        name1: First name
        name2: Second name
        threshold: Matching threshold (0-100)

    Returns:
        True if names match above threshold
    """
    score = fuzz.ratio(normalize_name(name1), normalize_name(name2))
    return score >= threshold


def _resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    project_root = Path(__file__).resolve().parent.parent
    return project_root / p


def _build_name_variants(name: str) -> List[str]:
    """
    Build searchable variants from a compound name.

    For entries like "Oren, Alon, and Tal Alexander" we want to match
    "Oren Alexander", "Alon Alexander", and "Tal Alexander" individually.
    For "George H.W. Bush and George W. Bush" we want both Bush names.

    Args:
        name: Full name string from severity data

    Returns:
        List of name variants to search for
    """
    variants = [name]

    # Handle "X and Y Z" patterns (e.g. "John and Tony Podesta")
    and_match = re.match(r'^(.+?)\s+and\s+(.+)$', name)
    if and_match:
        left, right = and_match.group(1), and_match.group(2)
        # If right side has a last name, apply it to left side names
        right_parts = right.strip().split()
        if len(right_parts) >= 2:
            last_name = right_parts[-1]
            variants.append(right.strip())
            # Split left on commas and "and"
            left_names = re.split(r',\s*|\s+and\s+', left)
            for ln in left_names:
                ln = ln.strip()
                if ln and last_name.lower() not in ln.lower():
                    variants.append(f"{ln} {last_name}")
                elif ln:
                    variants.append(ln)

    # Handle "First Last" → also match just "Last" for common names
    # (only for names with 2+ parts, skip single-word names like "Oprah")
    parts = name.split()
    if len(parts) == 2:
        variants.append(parts[-1])  # Last name only

    return list(set(variants))


def build_feature_matrix(
    raw_data_dir: str = "data/raw",
    severity_path: str = "data/processed/severity_scores.csv",
    output_path: str = "data/processed/features.csv"
) -> pd.DataFrame:
    """
    Build complete feature matrix from document corpus.

    Args:
        raw_data_dir: Directory with raw JSON files
        severity_path: Path to severity scores CSV
        output_path: Path to save feature matrix

    Returns:
        Feature matrix DataFrame
    """
    logger.info("Starting feature extraction")

    # Resolve paths relative to project root
    raw_data_dir = str(_resolve_path(raw_data_dir))
    severity_path = str(_resolve_path(severity_path))
    output_path_resolved = _resolve_path(output_path)

    # Load severity scores
    severity_df = pd.read_csv(severity_path)
    logger.info(f"Loaded {len(severity_df)} people from severity scores")

    # Load document corpus
    corpus = load_document_corpus(raw_data_dir)

    # Initialize feature extractor
    extractor = FeatureExtractor()

    # Build a lookup: map name variants to canonical name
    # This lets us match "Oren Alexander" → "Oren, Alon, and Tal Alexander"
    variant_to_canonical = {}
    for canonical_name in severity_df['name']:
        for variant in _build_name_variants(canonical_name):
            variant_to_canonical[normalize_name(variant).lower()] = canonical_name

    logger.info(
        f"Built {len(variant_to_canonical)} name variants "
        f"for {len(severity_df)} people"
    )

    # Storage for person features
    person_features = defaultdict(lambda: {
        'mention_count': 0,
        'total_mentions': 0,
        'sentiment_scores': [],
        'cooccurrence_score': 0,
        'doc_types': set(),
        'in_subject_line': False
    })

    def _match_person(person: str) -> str:
        """Match an NER-extracted person to a known name."""
        normed = normalize_name(person).lower()

        # Exact match on variants
        if normed in variant_to_canonical:
            return variant_to_canonical[normed]

        # Fuzzy match against variants
        for variant_lower, canonical in variant_to_canonical.items():
            if fuzz.ratio(normed, variant_lower) >= 85:
                return canonical

        return None

    # Process each document
    logger.info(f"Processing {len(corpus)} documents...")
    for doc_id, doc_data in tqdm(corpus.items(), desc="Processing documents"):
        if not doc_data.get('success', False):
            continue

        text = doc_data.get('text', '')
        if not text:
            continue

        # Extract persons mentioned in this document
        persons_in_doc = extractor.extract_persons(text)

        # Infer document type
        doc_type = extractor.infer_doc_type(text)

        # Process each person in the document
        for person in persons_in_doc:
            # Match against known names (with variant lookup)
            matched_name = _match_person(person)

            if not matched_name:
                continue

            # Count mentions
            person_lower = person.lower()
            text_lower = text.lower()
            mention_count = text_lower.count(person_lower)

            person_features[matched_name]['mention_count'] += 1
            person_features[matched_name]['total_mentions'] += mention_count

            # Get context and sentiment
            contexts = extractor.get_sentence_context(text, person)
            for context in contexts:
                sentiment = extractor.compute_sentiment(context)
                person_features[matched_name]['sentiment_scores'].append(sentiment)

            # Co-occurrence score
            cooccur = extractor.compute_cooccurrence_score(text, person)
            person_features[matched_name]['cooccurrence_score'] += cooccur

            # Document type
            person_features[matched_name]['doc_types'].add(doc_type)

            # Subject line check
            if extractor.check_subject_line(text, person):
                person_features[matched_name]['in_subject_line'] = True

    # Build feature DataFrame
    logger.info("Building feature matrix...")
    features_list = []

    for name in severity_df['name']:
        if name not in person_features:
            # Person not mentioned in corpus
            features = {
                'name': name,
                'mention_count': 0,
                'total_mentions': 0,
                'mean_context_sentiment': 0.0,
                'cooccurrence_score': 0,
                'doc_type_diversity': 0,
                'name_in_subject_line': 0,
            }
        else:
            pf = person_features[name]
            features = {
                'name': name,
                'mention_count': pf['mention_count'],
                'total_mentions': pf['total_mentions'],
                'mean_context_sentiment': (
                    sum(pf['sentiment_scores']) / len(pf['sentiment_scores'])
                    if pf['sentiment_scores'] else 0.0
                ),
                'cooccurrence_score': pf['cooccurrence_score'],
                'doc_type_diversity': len(pf['doc_types']),
                'name_in_subject_line': int(pf['in_subject_line']),
            }

        features_list.append(features)

    df = pd.DataFrame(features_list)

    # Merge with severity scores
    df = df.merge(severity_df[['name', 'severity_score']], on='name', how='left')

    # Save to CSV
    output_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path_resolved, index=False)

    logger.info(f"Saved feature matrix with {len(df)} rows to {output_path_resolved}")
    logger.info(f"Feature statistics:\n{df.describe()}")

    # Log how many people were actually found in the corpus
    found = sum(1 for n in severity_df['name'] if n in person_features)
    logger.info(
        f"Found {found}/{len(severity_df)} people in document corpus "
        f"({len(severity_df) - found} had zero mentions)"
    )

    return df


def main() -> None:
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build NLP feature matrix from Epstein document corpus"
    )
    parser.add_argument(
        "--raw-data-dir", default="data/raw",
        help="Directory containing ds*_agg.json files"
    )
    parser.add_argument(
        "--severity-path", default="data/processed/severity_scores.csv",
        help="Path to severity scores CSV"
    )
    parser.add_argument(
        "--output", default="data/processed/features.csv",
        help="Output CSV file path"
    )
    args = parser.parse_args()

    build_feature_matrix(
        raw_data_dir=args.raw_data_dir,
        severity_path=args.severity_path,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
