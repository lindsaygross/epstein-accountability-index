# Project: The Impunity Index
# Authors: Lindsay Gross, Shreya Mendi, Andrew Jin
# Advisor: Brinnae Bent, PhD
# Claude chat: https://claude.ai/chat/f8744002-3279-48ab-9d9a-8efa1fdb1af1
# Built with Claude AI assistance

"""
Extract severity scores from epsteinoverview.com data.

This script extracts person names and concern scores (used as severity scores)
from the Epstein Overview website. It supports two modes:

1. Local mode (default): Reads from data/scraped/epsteinoverview_scores.json,
   a pre-scraped snapshot of all 83 topics from the live site, which contains
   names, concern scores (0-10), severity levels, and profile URLs.
2. Live mode: Attempts to scrape epsteinoverview.com using requests+BeautifulSoup.
   NOTE: The site is a React SPA, so live scraping falls back to the local snapshot.

Non-person topics (e.g. "Dentist", "Pizza", "Bitcoin") are filtered out to keep
only named individuals relevant to the ML pipeline.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://epsteinoverview.com"
REQUEST_DELAY = 1.5  # Seconds between requests

# Default path to pre-scraped site data (83 topics, scraped 2026-03-01)
DEFAULT_SCRAPED_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "scraped" / "epsteinoverview_scores.json"
)

# Non-person topics to exclude from severity scores
NON_PERSON_TOPICS = {
    "dentist", "gynecologist", "pregnant", "whoops",
    "beef jerky", "pizza", "cream cheese",
    "drugs", "bitcoin", "9/11", "zorro ranch",
    "baal and occult references", "israel and mossad",
    "dangene and jennie enterprise", "epstein suicide",
    "qatar", "lifetouch",
}


def _is_person_topic(name: str) -> bool:
    """
    Determine if a topic name refers to an actual person.

    Args:
        name: Topic name from the site

    Returns:
        True if the topic represents a person (not a place, concept, etc.)
    """
    return name.lower().strip() not in NON_PERSON_TOPICS


def extract_from_scraped(
    scraped_path: str = DEFAULT_SCRAPED_PATH,
    include_non_persons: bool = False
) -> List[Dict]:
    """
    Extract severity scores from the pre-scraped epsteinoverview.com snapshot.

    This JSON file was scraped from the live React SPA and contains all 83
    topics with concern scores on a 0-10 scale.

    Args:
        scraped_path: Path to epsteinoverview_scores.json
        include_non_persons: If True, include non-person topics

    Returns:
        List of dicts with name, severity_score, profile_url
    """
    data_path = Path(scraped_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Scraped data not found at {data_path}. "
            "Ensure data/scraped/epsteinoverview_scores.json exists."
        )

    with open(data_path, "r") as f:
        topics = json.load(f)

    people = []
    skipped = []

    for topic in topics:
        name = topic.get("name", "").strip()

        if not include_non_persons and not _is_person_topic(name):
            skipped.append(name)
            continue

        person = {
            "name": name,
            "severity_score": float(topic.get("severity_score", 0)),
            "profile_url": topic.get("profile_url", ""),
        }
        people.append(person)

    if skipped:
        logger.info(f"Filtered out {len(skipped)} non-person topics: {skipped}")

    logger.info(f"Extracted {len(people)} people from scraped site data")
    return people


def scrape_person_list(base_url: str = BASE_URL) -> List[Dict[str, str]]:
    """
    Attempt to scrape the person list from epsteinoverview.com.

    NOTE: epsteinoverview.com is a React SPA that renders all content via
    JavaScript. requests+BeautifulSoup cannot extract JS-rendered content.
    This function will detect the empty SPA shell and fall back to the
    pre-scraped local snapshot automatically.

    Args:
        base_url: Base URL of the website

    Returns:
        List of dictionaries with person data
    """
    logger.info(f"Attempting live scrape from {base_url}")

    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {base_url}: {e}")
        logger.info("Falling back to pre-scraped data")
        return extract_from_scraped()

    soup = BeautifulSoup(response.content, "html.parser")

    # Detect React SPA shell (empty <div id="root"></div>)
    root_div = soup.find("div", id="root")
    if root_div and not root_div.get_text(strip=True):
        logger.warning(
            "Detected empty React SPA shell — content is JS-rendered. "
            "Falling back to pre-scraped data."
        )
        return extract_from_scraped()

    # If the site ever switches to server-side rendering, parse the cards.
    # Topic cards use: h3.font-semibold for names, span with X/10 pattern for scores,
    # a[href*="/topic/"] for links, border-l-concern-{color} for severity colors.
    people = []
    topic_links = soup.find_all("a", href=lambda h: h and "/topic/" in h)

    seen_hrefs = set()
    for link in topic_links:
        href = link.get("href", "")
        if href in seen_hrefs:
            continue
        seen_hrefs.add(href)

        name_elem = link.find("h3", class_="font-semibold")
        if not name_elem:
            continue

        name = name_elem.get_text(strip=True)

        # Find concern score badge: text matching "X.X/10 — Level"
        score = None
        for span in link.find_all("span"):
            text = span.get_text(strip=True)
            import re
            match = re.match(r'^([\d.]+)/10\s*[—-]\s*(.+)$', text)
            if match:
                score = float(match.group(1))
                break

        if name and score is not None and _is_person_topic(name):
            people.append({
                "name": name,
                "severity_score": score,
                "profile_url": f"{base_url}{href}",
            })

        time.sleep(REQUEST_DELAY)

    if not people:
        logger.warning("Live scrape returned no results. Using pre-scraped data.")
        return extract_from_scraped()

    logger.info(f"Scraped {len(people)} people from live site")
    return people


def scrape_severity_scores(
    output_path: str = "data/processed/severity_scores.csv",
    local: bool = True,
    scraped_path: str = DEFAULT_SCRAPED_PATH
) -> pd.DataFrame:
    """
    Extract severity scores and save to CSV.

    Args:
        output_path: Path to save the output CSV file
        local: If True, extract from pre-scraped snapshot
        scraped_path: Path to scraped JSON (local mode)

    Returns:
        DataFrame with severity scores
    """
    logger.info("Starting severity score extraction")

    if local:
        people = extract_from_scraped(scraped_path)
    else:
        people = scrape_person_list()

    if not people:
        logger.warning("No severity scores extracted")
        return pd.DataFrame(columns=["name", "severity_score", "profile_url"])

    df = pd.DataFrame(people)

    # Sort by severity score descending
    df = df.sort_values("severity_score", ascending=False).reset_index(drop=True)

    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} severity scores to {output_path}")

    if len(df) > 0:
        logger.info(f"Severity score statistics:\n{df['severity_score'].describe()}")

    return df


def main() -> None:
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract severity scores from Epstein Overview data"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Attempt live scraping from epsteinoverview.com (falls back to local)"
    )
    parser.add_argument(
        "--output", default="data/processed/severity_scores.csv",
        help="Output CSV file path"
    )
    args = parser.parse_args()

    scrape_severity_scores(output_path=args.output, local=not args.live)


if __name__ == "__main__":
    main()
