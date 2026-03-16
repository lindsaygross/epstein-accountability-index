# Project: The Impunity Index
# Authors: Lindsay Gross, Shreya Mendi, Andrew Jin
# Advisor: Brinnae Bent, PhD
# Claude chat: https://claude.ai/chat/f8744002-3279-48ab-9d9a-8efa1fdb1af1
# Built with Claude AI assistance

"""
expand_persons.py — Expand the people registry from 66 → 1,264+ people.

Pulls structured person data from the kaggle Epstein persons dataset
(stored in the sibling epstein-paper-trail repo) and merges it with
the existing epsteinoverview severity scores and consequence labels.

Output files:
    data/processed/people_registry.csv  — full 1264+ person registry
    data/processed/consequences_enriched.csv — consequences with geo/sector data
    data/processed/features_extended.csv — features with new columns merged in

Usage:
    python scripts/expand_persons.py
    python scripts/expand_persons.py --kaggle-csv /path/to/epstein-persons.csv
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIBLING_ROOT = PROJECT_ROOT.parent / "epstein-paper-trail"

DEFAULT_KAGGLE_CSV = SIBLING_ROOT / "data" / "raw" / "epstein-persons-2026-02-13_cleaned.csv"
CONSEQUENCE_LABELS_JSON = SIBLING_ROOT / "data" / "processed" / "consequence_labels.json"
EPSTEIN_SCORES_JSON = PROJECT_ROOT / "data" / "scraped" / "epsteinoverview_scores.json"
CONSEQUENCES_CSV = PROJECT_ROOT / "data" / "processed" / "consequences.csv"
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"

OUTPUT_REGISTRY = PROJECT_ROOT / "data" / "processed" / "people_registry.csv"
OUTPUT_CONSEQUENCES_ENRICHED = PROJECT_ROOT / "data" / "processed" / "consequences_enriched.csv"

# Sector mapping: kaggle category → our sector label
CATEGORY_TO_SECTOR = {
    "associate": "associate",
    "business": "finance",
    "celebrity": "entertainment",
    "politician": "politics",
    "academic": "academia",
    "legal": "legal",
    "socialite": "socialite",
    "royalty": "royalty",
    "military-intelligence": "intelligence",
    "other": "other",
}

# Jurisdiction inference from nationality (best-effort)
NATIONALITY_TO_JURISDICTION = {
    "American": "us_federal",
    "British": "uk",
    "French": "eu",
    "Israeli": "israel",
    "Australian": "australia",
    "Canadian": "canada",
    "German": "eu",
    "Italian": "eu",
    "Russian": "russia",
    "Brazilian": "brazil",
    "Belgian": "eu",
    "Swedish": "eu",
    "Norwegian": "eu",
    "Dutch": "eu",
    "Spanish": "eu",
    "Austrian": "eu",
    "Swiss": "eu",
    "Saudi": "middle_east",
    "Emirati": "middle_east",
    "Qatari": "middle_east",
}


def _normalize_name(name: str) -> str:
    """Normalize whitespace and case for name matching."""
    return re.sub(r"\s+", " ", str(name)).strip()


def _infer_jurisdiction(nationality: str) -> str:
    """Map nationality string to jurisdiction label."""
    if not nationality or pd.isna(nationality):
        return "unknown"
    nat = str(nationality).strip()
    # Handle hyphenated (e.g. "Russian-American")
    for part in nat.replace("-", " ").split():
        for key, val in NATIONALITY_TO_JURISDICTION.items():
            if key.lower() in part.lower():
                return val
    return "other"


def _infer_country(nationality: str) -> str:
    """Map nationality string to country name."""
    if not nationality or pd.isna(nationality):
        return "Unknown"
    nat = str(nationality).strip()
    country_map = {
        "American": "USA", "British": "UK", "French": "France",
        "Israeli": "Israel", "Australian": "Australia", "Canadian": "Canada",
        "German": "Germany", "Italian": "Italy", "Russian": "Russia",
        "Brazilian": "Brazil", "Belgian": "Belgium", "Swedish": "Sweden",
        "Norwegian": "Norway", "Dutch": "Netherlands", "Spanish": "Spain",
        "Austrian": "Austria", "Swiss": "Switzerland", "Polish": "Poland",
        "Czech": "Czech Republic", "Slovak": "Slovakia", "Hungarian": "Hungary",
        "Saudi": "Saudi Arabia", "Emirati": "UAE", "Qatari": "Qatar",
        "South African": "South Africa", "Mexican": "Mexico",
        "Argentine": "Argentina", "Colombian": "Colombia",
    }
    for key, val in country_map.items():
        if key.lower() in nat.lower():
            return val
    return nat  # Fall back to raw nationality string


def load_kaggle_persons(csv_path: Path) -> pd.DataFrame:
    """
    Load the kaggle Epstein persons CSV.

    Args:
        csv_path: Path to epstein-persons-*_cleaned.csv

    Returns:
        DataFrame with columns: name, category, bio, flights, documents,
        connections, in_black_book, nationality, sector, country, jurisdiction
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Kaggle CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records from {csv_path.name}")

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename to standard names
    rename = {
        "name": "name",
        "category": "category",
        "bio": "bio",
        "flights": "flights",
        "documents": "documents_count",
        "connections": "connections",
        "in_black_book": "in_black_book",
        "nationality": "nationality",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Clean
    df["name"] = df["name"].apply(_normalize_name)
    df["category"] = df["category"].fillna("other").str.strip().str.lower()
    df["nationality"] = df["nationality"].fillna("").str.strip()
    df["flights"] = pd.to_numeric(df.get("flights", 0), errors="coerce").fillna(0).astype(int)
    df["documents_count"] = pd.to_numeric(df.get("documents_count", 0), errors="coerce").fillna(0).astype(int)
    df["connections"] = pd.to_numeric(df.get("connections", 0), errors="coerce").fillna(0).astype(int)
    df["in_black_book"] = df.get("in_black_book", False).astype(bool)

    # Derived fields
    df["sector"] = df["category"].map(CATEGORY_TO_SECTOR).fillna("other")
    df["country"] = df["nationality"].apply(_infer_country)
    df["jurisdiction"] = df["nationality"].apply(_infer_jurisdiction)

    # Drop index column if present
    if "unnamed:_0" in df.columns or "" in df.columns:
        df = df.loc[:, ~df.columns.str.startswith("unnamed")]

    logger.info(f"After cleaning: {len(df)} people, {df['nationality'].ne('').sum()} with nationality")
    logger.info(f"Sectors: {df['sector'].value_counts().to_dict()}")

    return df[["name", "category", "sector", "bio", "flights",
               "documents_count", "connections", "in_black_book",
               "nationality", "country", "jurisdiction"]]


def load_epsteinoverview_scores(json_path: Path) -> pd.DataFrame:
    """Load epsteinoverview severity scores."""
    import json
    if not json_path.exists():
        logger.warning(f"Scores file not found: {json_path}")
        return pd.DataFrame(columns=["name", "severity_score"])

    with open(json_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["name"] = df["name"].apply(_normalize_name)
    return df[["name", "severity_score"]].drop_duplicates("name")


def load_existing_consequences(csv_path: Path) -> pd.DataFrame:
    """Load existing hand-labeled consequences for 66 people."""
    if not csv_path.exists():
        return pd.DataFrame(columns=["name", "consequence_tier",
                                     "consequence_description", "source_url"])
    df = pd.read_csv(csv_path)
    df["name"] = df["name"].apply(_normalize_name)
    return df


def load_paper_trail_labels(json_path: Path) -> pd.DataFrame:
    """
    Load consequence labels from paper-trail project.

    Paper-trail uses a 4-tier system (reversed from ours):
        0 = Charged/Convicted
        1 = Settled Civilly
        2 = Named/Investigated Only
        3 = No Consequences

    We convert to our 3-tier system:
        0 = No Consequence
        1 = Soft (Settled/Named)
        2 = Hard (Charged/Convicted)
    """
    import json
    if not json_path.exists():
        logger.warning(f"Paper-trail labels not found: {json_path}")
        return pd.DataFrame(columns=["name", "consequence_tier_pt"])

    with open(json_path) as f:
        data = json.load(f)

    labels = data.get("labels", {})
    records = []
    for name, info in labels.items():
        pt_tier = info.get("tier", 3)
        # Convert: pt_tier 0=convicted→our 2, pt_tier 1=civil→our 1,
        #          pt_tier 2=named→our 1, pt_tier 3=none→our 0
        our_tier = {0: 2, 1: 1, 2: 1, 3: 0}.get(pt_tier, 0)
        records.append({
            "name": _normalize_name(name),
            "consequence_tier_pt": pt_tier,
            "consequence_tier_inferred": our_tier,
        })

    return pd.DataFrame(records)


def build_registry(
    kaggle_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    consequences_df: pd.DataFrame,
    pt_labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all data sources into a unified people registry.

    Priority for consequence_tier:
        1. Existing hand-labeled consequences (most accurate, 66 people)
        2. Paper-trail labels (32 labeled people)
        3. Default: 0 (no consequence)

    Args:
        kaggle_df: 1264 people from kaggle CSV
        scores_df: Severity scores from epsteinoverview
        consequences_df: Existing hand-labeled 66-person consequences
        pt_labels_df: Paper-trail consequence labels

    Returns:
        Unified registry DataFrame
    """
    # Start with kaggle as the base (widest coverage)
    registry = kaggle_df.copy()

    # Merge severity scores
    registry = registry.merge(scores_df, on="name", how="left")
    registry["severity_score"] = registry["severity_score"].fillna(0.0)

    # Merge existing consequences (hand-labeled, highest priority)
    registry = registry.merge(
        consequences_df[["name", "consequence_tier", "consequence_description", "source_url"]],
        on="name",
        how="left"
    )

    # Merge paper-trail labels for people not already covered
    registry = registry.merge(pt_labels_df, on="name", how="left")

    # Fill consequence_tier: prefer hand-labeled, then pt_inferred, then 0
    mask_missing = registry["consequence_tier"].isna()
    registry.loc[mask_missing, "consequence_tier"] = (
        registry.loc[mask_missing, "consequence_tier_inferred"].fillna(0)
    )
    registry["consequence_tier"] = registry["consequence_tier"].astype(int)

    # Drop interim columns
    registry = registry.drop(
        columns=["consequence_tier_pt", "consequence_tier_inferred"],
        errors="ignore"
    )

    # Fill remaining NaNs
    registry["consequence_description"] = registry["consequence_description"].fillna("")
    registry["source_url"] = registry["source_url"].fillna("")
    registry["bio"] = registry["bio"].fillna("")

    # Mark data source
    hand_labeled = set(consequences_df["name"].tolist())
    pt_labeled = set(pt_labels_df["name"].tolist()) if not pt_labels_df.empty else set()
    def _label_source(row):
        if row["name"] in hand_labeled:
            return "hand_labeled"
        elif row["name"] in pt_labeled:
            return "paper_trail"
        else:
            return "kaggle_only"
    registry["consequence_source"] = registry.apply(_label_source, axis=1)

    logger.info(f"Registry built: {len(registry)} people")
    logger.info(f"Consequence tier distribution: {registry['consequence_tier'].value_counts().sort_index().to_dict()}")
    logger.info(f"Consequence sources: {registry['consequence_source'].value_counts().to_dict()}")
    logger.info(f"Countries: {registry['country'].value_counts().head(10).to_dict()}")
    logger.info(f"Sectors: {registry['sector'].value_counts().to_dict()}")

    return registry.sort_values("name").reset_index(drop=True)


def enrich_consequences_csv(registry: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Write an enriched consequences CSV with geographic and sector data.

    This is the expanded replacement for data/processed/consequences.csv
    that includes all 1264 people with their geographic metadata.

    Args:
        registry: Full people registry DataFrame
        output_path: Where to write the enriched CSV

    Returns:
        Enriched DataFrame
    """
    cols = [
        "name", "consequence_tier", "consequence_description", "source_url",
        "category", "sector", "nationality", "country", "jurisdiction",
        "flights", "connections", "in_black_book", "consequence_source"
    ]
    enriched = registry[[c for c in cols if c in registry.columns]].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    logger.info(f"Saved enriched consequences ({len(enriched)} rows) to {output_path}")
    return enriched


def expand_persons(kaggle_csv: Optional[Path] = None) -> pd.DataFrame:
    """
    Main entry point: expand the people registry.

    Args:
        kaggle_csv: Path to kaggle persons CSV (defaults to sibling repo location)

    Returns:
        Full people registry DataFrame
    """
    csv_path = kaggle_csv or DEFAULT_KAGGLE_CSV

    logger.info("=" * 60)
    logger.info("EXPANDING PEOPLE REGISTRY")
    logger.info("=" * 60)

    # Load all data sources
    kaggle_df = load_kaggle_persons(csv_path)
    scores_df = load_epsteinoverview_scores(EPSTEIN_SCORES_JSON)
    consequences_df = load_existing_consequences(CONSEQUENCES_CSV)
    pt_labels_df = load_paper_trail_labels(CONSEQUENCE_LABELS_JSON)

    logger.info(f"\nData source sizes:")
    logger.info(f"  Kaggle persons: {len(kaggle_df)}")
    logger.info(f"  Severity scores: {len(scores_df)}")
    logger.info(f"  Existing hand-labeled consequences: {len(consequences_df)}")
    logger.info(f"  Paper-trail labels: {len(pt_labels_df)}")

    # Build registry
    registry = build_registry(kaggle_df, scores_df, consequences_df, pt_labels_df)

    # Save outputs
    OUTPUT_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(OUTPUT_REGISTRY, index=False)
    logger.info(f"\nSaved full registry ({len(registry)} people) to {OUTPUT_REGISTRY}")

    enrich_consequences_csv(registry, OUTPUT_CONSEQUENCES_ENRICHED)

    logger.info("\nDone! Summary:")
    logger.info(f"  Total people: {len(registry)}")
    logger.info(f"  With nationality: {(registry['nationality'] != '').sum()}")
    logger.info(f"  With consequences (tier > 0): {(registry['consequence_tier'] > 0).sum()}")
    logger.info(f"  With severity scores: {(registry['severity_score'] > 0).sum()}")
    logger.info(f"  In black book: {registry['in_black_book'].sum()}")

    return registry


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Expand people registry from 66 → 1264+ people"
    )
    parser.add_argument(
        "--kaggle-csv",
        default=None,
        help="Path to kaggle Epstein persons CSV (default: sibling repo location)"
    )
    args = parser.parse_args()

    csv_path = Path(args.kaggle_csv) if args.kaggle_csv else None
    expand_persons(csv_path)


if __name__ == "__main__":
    main()
