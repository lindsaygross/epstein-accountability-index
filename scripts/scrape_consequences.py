# Attribution: Scaffolded with AI assistance (Claude, Anthropic)

"""
Scrape consequence data from Wikipedia and Google News.

This script searches for consequence-related information about individuals
mentioned in the Epstein files, assigning consequence tiers based on severity.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup
import wikipediaapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
REQUEST_DELAY = 1.5  # Seconds between requests
CONSEQUENCE_KEYWORDS_SOFT = [
    "resigned", "stepped down", "lost position", "removed from",
    "retired", "departure", "left position"
]
CONSEQUENCE_KEYWORDS_HARD = [
    "arrested", "charged", "convicted", "indicted", "sentenced",
    "prosecution", "criminal charges", "pleaded guilty"
]

# Manual overrides for known cases
MANUAL_OVERRIDES: Dict[str, Dict[str, any]] = {
    "Ghislaine Maxwell": {
        "consequence_tier": 2,
        "consequence_description": "Arrested and convicted on charges related to Epstein case",
        "source_url": "https://en.wikipedia.org/wiki/Ghislaine_Maxwell"
    },
    # Add more manual overrides as needed
}


class ConsequenceScraper:
    """Scraper for consequence data from multiple sources."""

    def __init__(self):
        """Initialize the consequence scraper."""
        self.wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='EpsteinAccountabilityIndex/1.0 (Educational Research)'
        )

    def search_wikipedia(self, name: str) -> Optional[Tuple[int, str, str]]:
        """
        Search Wikipedia for consequence-related information.

        Args:
            name: Person's name to search

        Returns:
            Tuple of (consequence_tier, description, url) or None
        """
        logger.debug(f"Searching Wikipedia for {name}")

        try:
            page = self.wiki_wiki.page(name)

            if not page.exists():
                logger.debug(f"No Wikipedia page found for {name}")
                return None

            text = page.text.lower()
            url = page.fullurl

            # Check for hard consequences first
            for keyword in CONSEQUENCE_KEYWORDS_HARD:
                if keyword in text:
                    # Extract context around keyword
                    idx = text.index(keyword)
                    context_start = max(0, idx - 100)
                    context_end = min(len(text), idx + 100)
                    context = text[context_start:context_end]

                    return (
                        2,
                        f"Hard consequence detected: {keyword}. Context: {context}",
                        url
                    )

            # Check for soft consequences
            for keyword in CONSEQUENCE_KEYWORDS_SOFT:
                if keyword in text:
                    idx = text.index(keyword)
                    context_start = max(0, idx - 100)
                    context_end = min(len(text), idx + 100)
                    context = text[context_start:context_end]

                    return (
                        1,
                        f"Soft consequence detected: {keyword}. Context: {context}",
                        url
                    )

            return (0, "No consequences detected in Wikipedia", url)

        except Exception as e:
            logger.warning(f"Error searching Wikipedia for {name}: {e}")
            return None

    def search_google_news(self, name: str) -> Optional[Tuple[int, str, str]]:
        """
        Search Google News RSS feed for consequence-related articles.

        Args:
            name: Person's name to search

        Returns:
            Tuple of (consequence_tier, description, url) or None
        """
        logger.debug(f"Searching Google News for {name}")

        query = quote_plus(f"{name} epstein")
        rss_url = f"https://news.google.com/rss/search?q={query}"

        try:
            response = requests.get(rss_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "xml")
            items = soup.find_all("item", limit=5)  # Check first 5 articles

            if not items:
                logger.debug(f"No news articles found for {name}")
                return None

            # Analyze article titles and descriptions
            for item in items:
                title = item.title.text.lower() if item.title else ""
                description = item.description.text.lower() if item.description else ""
                link = item.link.text if item.link else ""

                combined_text = f"{title} {description}"

                # Check for hard consequences
                for keyword in CONSEQUENCE_KEYWORDS_HARD:
                    if keyword in combined_text:
                        return (
                            2,
                            f"News: Hard consequence detected - {title[:100]}",
                            link
                        )

                # Check for soft consequences
                for keyword in CONSEQUENCE_KEYWORDS_SOFT:
                    if keyword in combined_text:
                        return (
                            1,
                            f"News: Soft consequence detected - {title[:100]}",
                            link
                        )

            return (0, "No consequences detected in news", rss_url)

        except Exception as e:
            logger.warning(f"Error searching Google News for {name}: {e}")
            return None

    def get_consequence_info(self, name: str) -> Dict[str, any]:
        """
        Get consequence information for a person from all sources.

        Args:
            name: Person's name

        Returns:
            Dictionary with consequence information
        """
        # Check manual overrides first
        if name in MANUAL_OVERRIDES:
            logger.info(f"Using manual override for {name}")
            return MANUAL_OVERRIDES[name]

        # Try Wikipedia first
        wiki_result = self.search_wikipedia(name)
        time.sleep(REQUEST_DELAY)

        # Try Google News
        news_result = self.search_google_news(name)
        time.sleep(REQUEST_DELAY)

        # Combine results (take the highest consequence tier)
        results = [r for r in [wiki_result, news_result] if r is not None]

        if not results:
            return {
                "consequence_tier": 0,
                "consequence_description": "No information found",
                "source_url": None
            }

        # Sort by tier (highest first)
        results.sort(key=lambda x: x[0], reverse=True)
        tier, description, url = results[0]

        return {
            "consequence_tier": tier,
            "consequence_description": description,
            "source_url": url
        }


def scrape_consequences(
    names: List[str],
    output_path: str = "data/processed/consequences.csv"
) -> pd.DataFrame:
    """
    Scrape consequence data for a list of names.

    Args:
        names: List of person names to search
        output_path: Path to save output CSV

    Returns:
        DataFrame with consequence data
    """
    logger.info(f"Scraping consequences for {len(names)} people")

    scraper = ConsequenceScraper()
    results = []

    for i, name in enumerate(names, 1):
        logger.info(f"Processing {i}/{len(names)}: {name}")

        consequence_info = scraper.get_consequence_info(name)
        results.append({
            "name": name,
            **consequence_info
        })

        # Periodic save
        if i % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(f"{output_path}.tmp", index=False)
            logger.info(f"Saved temporary results ({i}/{len(names)})")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Create output directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} consequence records to {output_path}")

    # Log statistics
    tier_counts = df["consequence_tier"].value_counts().sort_index()
    logger.info(f"Consequence tier distribution:\n{tier_counts}")

    return df


def load_names_from_severity_file(
    severity_path: str = "data/processed/severity_scores.csv"
) -> List[str]:
    """
    Load person names from severity scores CSV.

    Args:
        severity_path: Path to severity scores CSV

    Returns:
        List of person names
    """
    df = pd.read_csv(severity_path)
    return df["name"].tolist()


def main() -> None:
    """Main entry point for the script."""
    # Load names from severity scores
    names = load_names_from_severity_file()
    logger.info(f"Loaded {len(names)} names from severity scores")

    # Scrape consequences
    scrape_consequences(names)


if __name__ == "__main__":
    main()
