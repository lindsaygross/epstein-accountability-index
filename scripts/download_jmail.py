# Attribution: Scaffolded with AI assistance (Claude, Anthropic)

"""
Download and cache data from jmail.world for summary generation and model training.

The JMail Data API (https://data.jmail.world/v1/) provides free, open access
to Epstein case files with no authentication required.

This script downloads:
- Document metadata and text (25 MB Parquet)
- Email metadata and body text (334 MB Parquet) — queried remotely via DuckDB

Results are cached locally in data/jmail_cache/ to avoid re-downloading.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

JMAIL_BASE = "https://data.jmail.world/v1"

# Parquet file URLs
DATASETS = {
    "documents": f"{JMAIL_BASE}/documents.parquet",
    "emails_slim": f"{JMAIL_BASE}/emails-slim.parquet",
    "emails_full": f"{JMAIL_BASE}/emails.parquet",
}

# Document-full volumes
DOC_VOLUMES = {
    "VOL00008": f"{JMAIL_BASE}/documents-full/VOL00008.parquet",
    "VOL00009": f"{JMAIL_BASE}/documents-full/VOL00009.parquet",
    "VOL00010": f"{JMAIL_BASE}/documents-full/VOL00010.parquet",
    "DataSet11": f"{JMAIL_BASE}/documents-full/DataSet11.parquet",
    "other": f"{JMAIL_BASE}/documents-full/other.parquet",
}


def ensure_duckdb():
    """Import duckdb, installing if needed."""
    try:
        import duckdb
        return duckdb
    except ImportError:
        raise ImportError("duckdb is required: pip install duckdb pyarrow")


def download_parquet_local(url: str, cache_path: Path) -> Path:
    """Download a Parquet file to local cache."""
    if cache_path.exists():
        logger.info(f"  Using cached: {cache_path.name}")
        return cache_path

    import requests
    logger.info(f"  Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(cache_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            total += len(chunk)

    logger.info(f"  Downloaded {total / 1024 / 1024:.1f} MB -> {cache_path.name}")
    return cache_path


def query_remote_parquet(sql: str) -> pd.DataFrame:
    """Execute SQL against remote Parquet via DuckDB. Returns DataFrame."""
    duckdb = ensure_duckdb()
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    result = conn.execute(sql).fetchdf()
    conn.close()
    return result


def load_person_names(scores_path: str = "data/scraped/epsteinoverview_scores.json") -> List[str]:
    """Load the 66 tracked person names."""
    base = Path(__file__).parent.parent
    NON_PERSON = {
        "Dentist", "Pilot", "Pizza", "McDonald's", "Bitcoin",
        "Lolita Express", "Zorro Ranch", "Epstein Island",
        "Little St. James", "Bear Stearns", "Deutsche Bank",
        "JP Morgan", "Victoria's Secret", "MIT Media Lab",
        "Harvard University", "Ohio State", "Dalton School"
    }
    with open(base / scores_path) as f:
        scores = json.load(f)
    return [s["name"] for s in scores if s["name"] not in NON_PERSON]


def build_search_variants(name: str) -> List[str]:
    """Build search variants for a person name."""
    variants = [name.lower()]

    # Handle compound names like "Oren, Alon, and Tal Alexander"
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

    # Handle "George H.W. Bush and George W. Bush"
    if " Bush" in name and "George" in name:
        variants.extend(["george h.w. bush", "george w. bush", "george bush"])

    # Short names
    if name == "RFK":
        variants.extend(["robert f. kennedy", "rfk jr", "robert kennedy"])
    elif name == "Oprah":
        variants.append("oprah winfrey")
    elif "Justice " in name:
        variants.append(name.replace("Justice ", "").lower())

    return list(set(variants))


def download_documents_metadata(cache_dir: Path) -> pd.DataFrame:
    """Download the documents metadata Parquet file (25 MB)."""
    cache_path = cache_dir / "documents.parquet"
    download_parquet_local(DATASETS["documents"], cache_path)
    return pd.read_parquet(cache_path)


def download_document_volumes(cache_dir: Path, volumes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Download document-full volume Parquet files."""
    results = {}
    vols = volumes or list(DOC_VOLUMES.keys())
    for vol_name in vols:
        url = DOC_VOLUMES.get(vol_name)
        if not url:
            continue
        cache_path = cache_dir / f"documents-full-{vol_name}.parquet"
        download_parquet_local(url, cache_path)
        results[vol_name] = pd.read_parquet(cache_path)
    return results


def find_person_in_emails_remote(
    name: str,
    variants: List[str],
    limit: int = 500
) -> pd.DataFrame:
    """
    Query remote emails Parquet for mentions of a person.
    Uses DuckDB remote query to avoid downloading 334MB file.
    """
    duckdb = ensure_duckdb()

    # Build WHERE clause from variants
    conditions = []
    for v in variants:
        escaped = v.replace("'", "''")
        conditions.append(f"LOWER(body) LIKE '%{escaped}%'")
        conditions.append(f"LOWER(subject) LIKE '%{escaped}%'")

    where = " OR ".join(conditions)

    sql = f"""
    SELECT id, subject, date, "from", "to", body
    FROM read_parquet('{DATASETS["emails_full"]}')
    WHERE {where}
    LIMIT {limit}
    """

    try:
        conn = duckdb.connect()
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        result = conn.execute(sql).fetchdf()
        conn.close()
        return result
    except Exception as e:
        logger.warning(f"Remote email query failed for {name}: {e}")
        return pd.DataFrame()


def find_person_in_documents(
    name: str,
    variants: List[str],
    doc_volumes: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Search downloaded document volumes for mentions of a person."""
    matches = []

    for vol_name, df in doc_volumes.items():
        # Find text column
        text_col = None
        for col in ['text', 'body', 'content', 'ocr_text']:
            if col in df.columns:
                text_col = col
                break

        if text_col is None:
            continue

        for variant in variants:
            mask = df[text_col].str.contains(variant, case=False, na=False)
            matched = df[mask].copy()
            if len(matched) > 0:
                matched['source_volume'] = vol_name
                matched['matched_variant'] = variant
                matches.append(matched)

    if matches:
        combined = pd.concat(matches, ignore_index=True).drop_duplicates(
            subset=['id'] if 'id' in matches[0].columns else None
        )
        return combined

    return pd.DataFrame()


def download_all_for_project(
    cache_dir: str = "data/jmail_cache",
    skip_emails: bool = False,
    max_emails_per_person: int = 200,
) -> Dict[str, Dict]:
    """
    Download all relevant jmail.world data for the project.

    Returns dict mapping person_name -> {
        'emails': DataFrame,
        'documents': DataFrame,
        'total_count': int
    }
    """
    base = Path(__file__).parent.parent
    cache = base / cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    output_path = cache / "person_documents.json"
    if output_path.exists():
        logger.info("Loading cached person documents...")
        with open(output_path) as f:
            return json.load(f)

    names = load_person_names()
    logger.info(f"Processing {len(names)} people against jmail.world data")

    # Download document volumes locally (they're smaller)
    logger.info("\n=== Downloading document volumes ===")
    doc_volumes = download_document_volumes(cache)

    results = {}

    for i, name in enumerate(names):
        logger.info(f"\n[{i+1}/{len(names)}] Searching for: {name}")
        variants = build_search_variants(name)

        # Search documents (local)
        docs_df = find_person_in_documents(name, variants, doc_volumes)
        doc_count = len(docs_df)
        logger.info(f"  Documents: {doc_count} matches")

        # Search emails (remote, optional)
        email_count = 0
        if not skip_emails:
            try:
                emails_df = find_person_in_emails_remote(
                    name, variants, limit=max_emails_per_person
                )
                email_count = len(emails_df)
                logger.info(f"  Emails: {email_count} matches")
            except Exception as e:
                logger.warning(f"  Email search failed: {e}")
                emails_df = pd.DataFrame()
        else:
            emails_df = pd.DataFrame()

        # Store summary (not full DataFrames, to keep JSON small)
        results[name] = {
            'doc_count': doc_count,
            'email_count': email_count,
            'total_count': doc_count + email_count,
        }

        time.sleep(0.5)  # Be polite to API

    # Save summary
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved summary to {output_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download jmail.world data")
    parser.add_argument("--skip-emails", action="store_true",
                        help="Skip email queries (faster, documents only)")
    parser.add_argument("--cache-dir", default="data/jmail_cache",
                        help="Cache directory")
    args = parser.parse_args()

    download_all_for_project(
        cache_dir=args.cache_dir,
        skip_emails=args.skip_emails,
    )
