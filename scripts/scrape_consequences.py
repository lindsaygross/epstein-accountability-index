# Attribution: Scaffolded with AI assistance (Claude, Anthropic)

"""
Scrape consequence data from Wikipedia and Google News.

This script searches for consequence-related information about individuals
mentioned in the Epstein files, assigning consequence tiers based on severity:

    0 = No known consequence
    1 = Soft (resigned, stepped down, lost position, civil settlement)
    2 = Hard (arrested, charged, convicted, sentenced)

Manual overrides are provided for well-documented cases to ensure accuracy,
since automated keyword matching can produce false positives.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
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
    "retired", "departure", "left position", "settled", "settlement",
    "stripped of", "relinquished", "suspended"
]
CONSEQUENCE_KEYWORDS_HARD = [
    "arrested", "charged", "convicted", "indicted", "sentenced",
    "prosecution", "criminal charges", "pleaded guilty", "found guilty",
    "imprisoned", "prison", "jail"
]

# Epstein-related context keywords to validate matches
EPSTEIN_CONTEXT = [
    "epstein", "trafficking", "sexual abuse", "minor", "underage",
    "maxwell", "sex crime", "exploitation"
]

# Manual overrides for well-documented cases.
# These are based on publicly reported, widely covered outcomes.
MANUAL_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # === TIER 2: Hard consequences (arrested, charged, convicted) ===
    "Ghislaine Maxwell": {
        "consequence_tier": 2,
        "consequence_description": "Convicted of sex trafficking and conspiracy; sentenced to 20 years in federal prison (2022)",
        "source_url": "https://en.wikipedia.org/wiki/Ghislaine_Maxwell"
    },
    "Oren, Alon, and Tal Alexander": {
        "consequence_tier": 2,
        "consequence_description": "Oren Alexander arrested on sex trafficking charges (2024); brothers also facing allegations",
        "source_url": "https://en.wikipedia.org/wiki/Oren_Alexander"
    },
    "Ramsey Elkholy": {
        "consequence_tier": 2,
        "consequence_description": "Indicted on sex trafficking charges related to Epstein network",
        "source_url": "https://en.wikipedia.org/wiki/Ramsey_Elkholy"
    },

    # === TIER 1: Soft consequences (resigned, settled, lost position) ===
    "Prince Andrew": {
        "consequence_tier": 1,
        "consequence_description": "Stripped of military titles and royal patronages; settled civil lawsuit with Virginia Giuffre (2022)",
        "source_url": "https://en.wikipedia.org/wiki/Prince_Andrew,_Duke_of_York"
    },
    "Jes Staley": {
        "consequence_tier": 1,
        "consequence_description": "Resigned as Barclays CEO (2021); fined by UK regulators over Epstein ties; sued by JPMorgan",
        "source_url": "https://en.wikipedia.org/wiki/Jes_Staley"
    },
    "Leon Black": {
        "consequence_tier": 1,
        "consequence_description": "Stepped down as CEO of Apollo Global Management (2021) following scrutiny of Epstein payments",
        "source_url": "https://en.wikipedia.org/wiki/Leon_Black"
    },
    "Les Wexner": {
        "consequence_tier": 1,
        "consequence_description": "Stepped down as L Brands CEO (2020); severed ties with Epstein; admitted Epstein misappropriated funds",
        "source_url": "https://en.wikipedia.org/wiki/Les_Wexner"
    },
    "Bill Gates": {
        "consequence_tier": 1,
        "consequence_description": "Epstein meetings cited as factor in divorce from Melinda Gates (2021); faced sustained reputational damage",
        "source_url": "https://en.wikipedia.org/wiki/Bill_Gates"
    },
    "Sarah Ferguson": {
        "consequence_tier": 1,
        "consequence_description": "Publicly distanced from Prince Andrew; reputational damage from association",
        "source_url": "https://en.wikipedia.org/wiki/Sarah,_Duchess_of_York"
    },
    "Eva Andersson-Dubin": {
        "consequence_tier": 1,
        "consequence_description": "Named in civil lawsuits; faced public scrutiny over relationship with Epstein",
        "source_url": "https://en.wikipedia.org/wiki/Glenn_Dubin"
    },
    "Glenn Dubin": {
        "consequence_tier": 1,
        "consequence_description": "Named in civil lawsuits; faced public scrutiny and allegations related to Epstein",
        "source_url": "https://en.wikipedia.org/wiki/Glenn_Dubin"
    },
    "Leslie Groff": {
        "consequence_tier": 1,
        "consequence_description": "Named as co-conspirator in civil lawsuits; identified as Epstein's assistant who facilitated scheduling",
        "source_url": ""
    },
    "Peggy Siegal": {
        "consequence_tier": 1,
        "consequence_description": "Dropped by clients and socially ostracized after Epstein connections became public",
        "source_url": "https://en.wikipedia.org/wiki/Peggy_Siegal"
    },

    # === TIER 0: No known Epstein-related consequence ===
    "Donald Trump": {
        "consequence_tier": 0,
        "consequence_description": "No legal consequences related to Epstein; served as 45th and 47th President",
        "source_url": "https://en.wikipedia.org/wiki/Donald_Trump"
    },
    "Bill Clinton": {
        "consequence_tier": 0,
        "consequence_description": "No legal consequences related to Epstein; denied wrongdoing",
        "source_url": "https://en.wikipedia.org/wiki/Bill_Clinton"
    },
    "Elon Musk": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein; denied close relationship",
        "source_url": "https://en.wikipedia.org/wiki/Elon_Musk"
    },
    "Ehud Barak": {
        "consequence_tier": 0,
        "consequence_description": "Faced media scrutiny over Epstein visits; no formal consequences",
        "source_url": "https://en.wikipedia.org/wiki/Ehud_Barak"
    },
    "Bill Richardson": {
        "consequence_tier": 0,
        "consequence_description": "Denied allegations; died September 2023 before further proceedings",
        "source_url": "https://en.wikipedia.org/wiki/Bill_Richardson"
    },
    "Larry Summers": {
        "consequence_tier": 0,
        "consequence_description": "Acknowledged meeting Epstein; no consequences",
        "source_url": "https://en.wikipedia.org/wiki/Lawrence_Summers"
    },
    "Peter Mandelson": {
        "consequence_tier": 0,
        "consequence_description": "Named in documents; denied wrongdoing; no formal consequences",
        "source_url": "https://en.wikipedia.org/wiki/Peter_Mandelson"
    },
    "Steve Bannon": {
        "consequence_tier": 0,
        "consequence_description": "No Epstein-related consequences; separate legal issues unrelated to Epstein",
        "source_url": "https://en.wikipedia.org/wiki/Steve_Bannon"
    },
    "Howard Lutnick": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein",
        "source_url": "https://en.wikipedia.org/wiki/Howard_Lutnick"
    },
    "Jeff Bezos": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein; minimal documented connection",
        "source_url": "https://en.wikipedia.org/wiki/Jeff_Bezos"
    },
    "Mark Zuckerberg": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein; minimal documented connection",
        "source_url": "https://en.wikipedia.org/wiki/Mark_Zuckerberg"
    },
    "Ron Lauder": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein",
        "source_url": "https://en.wikipedia.org/wiki/Ronald_Lauder"
    },
    "Oprah": {
        "consequence_tier": 0,
        "consequence_description": "No consequences; minimal documented connection to Epstein",
        "source_url": "https://en.wikipedia.org/wiki/Oprah_Winfrey"
    },
    "Tom Hanks": {
        "consequence_tier": 0,
        "consequence_description": "No credible connection to Epstein; subject of conspiracy theories only",
        "source_url": "https://en.wikipedia.org/wiki/Tom_Hanks"
    },

    # === Additional overrides to correct automated false positives ===
    "Karyna Shuliak": {
        "consequence_tier": 0,
        "consequence_description": "Epstein's girlfriend; not charged with any crimes",
        "source_url": ""
    },
    "Sultan Bin Sulayem": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein",
        "source_url": ""
    },
    "William Riley": {
        "consequence_tier": 0,
        "consequence_description": "No publicly reported consequences",
        "source_url": ""
    },
    "Michael Wolff": {
        "consequence_tier": 0,
        "consequence_description": "Journalist/author; no consequences related to Epstein",
        "source_url": "https://en.wikipedia.org/wiki/Michael_Wolff_(journalist)"
    },
    "Corina Tarnita": {
        "consequence_tier": 0,
        "consequence_description": "Princeton professor; no consequences related to Epstein",
        "source_url": ""
    },
    "Lana Zakocela": {
        "consequence_tier": 0,
        "consequence_description": "Model; no Epstein-related consequences",
        "source_url": ""
    },
    "Susan Hamblin": {
        "consequence_tier": 0,
        "consequence_description": "No publicly reported consequences",
        "source_url": ""
    },
    "Peter Theil": {
        "consequence_tier": 0,
        "consequence_description": "Named in Epstein-related documents; no formal consequences",
        "source_url": "https://en.wikipedia.org/wiki/Peter_Thiel"
    },
    "Francis Derby": {
        "consequence_tier": 0,
        "consequence_description": "No publicly reported consequences",
        "source_url": ""
    },
    "Leon Botstein": {
        "consequence_tier": 0,
        "consequence_description": "Bard College president; received Epstein donations; no formal consequences",
        "source_url": "https://en.wikipedia.org/wiki/Leon_Botstein"
    },
    "Casey Wasserman": {
        "consequence_tier": 1,
        "consequence_description": "Artists and clients publicly departed his talent agency after Epstein file revelations (2026)",
        "source_url": ""
    },
    "Kathryn Ruemmler": {
        "consequence_tier": 0,
        "consequence_description": "Former Obama White House Counsel; met with Epstein; no formal consequences",
        "source_url": "https://en.wikipedia.org/wiki/Kathryn_Ruemmler"
    },
    "Peter Attia": {
        "consequence_tier": 1,
        "consequence_description": "Resigned from CBS News contributor role after Epstein file revelations (2026)",
        "source_url": "https://en.wikipedia.org/wiki/Peter_Attia"
    },
    "Jess Ting": {
        "consequence_tier": 0,
        "consequence_description": "No publicly reported consequences",
        "source_url": ""
    },
    "Paolo Zampolli": {
        "consequence_tier": 0,
        "consequence_description": "No Epstein-related legal consequences",
        "source_url": "https://en.wikipedia.org/wiki/Paolo_Zampolli"
    },
    "Chuck Schumer": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein; minimal documented connection",
        "source_url": "https://en.wikipedia.org/wiki/Chuck_Schumer"
    },
    "Rupert Murdoch": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein",
        "source_url": "https://en.wikipedia.org/wiki/Rupert_Murdoch"
    },
    "Stefan Krause": {
        "consequence_tier": 0,
        "consequence_description": "No publicly reported consequences",
        "source_url": ""
    },
    "Sheikh Sultan Bin Jassim Al Thani": {
        "consequence_tier": 0,
        "consequence_description": "No publicly reported consequences",
        "source_url": ""
    },
    "Hillary Clinton": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein; minimal documented connection",
        "source_url": "https://en.wikipedia.org/wiki/Hillary_Clinton"
    },
    "Kevin O'Leary": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein; minimal documented connection",
        "source_url": "https://en.wikipedia.org/wiki/Kevin_O%27Leary"
    },
    "Pam Bondi": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein",
        "source_url": "https://en.wikipedia.org/wiki/Pam_Bondi"
    },
    "George H.W. Bush and George W. Bush": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein; minimal documented connection",
        "source_url": ""
    },
    "Stanislas Nordey": {
        "consequence_tier": 0,
        "consequence_description": "No publicly reported consequences",
        "source_url": ""
    },
    "Bill O'Reilly": {
        "consequence_tier": 0,
        "consequence_description": "Fired from Fox News for separate sexual harassment allegations (not Epstein-related)",
        "source_url": "https://en.wikipedia.org/wiki/Bill_O%27Reilly_(political_commentator)"
    },
    "Kash Patel": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein",
        "source_url": "https://en.wikipedia.org/wiki/Kash_Patel"
    },
    "Betsy DeVos": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein",
        "source_url": "https://en.wikipedia.org/wiki/Betsy_DeVos"
    },
    "Benjamin Netanyahu": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein; separate domestic legal issues unrelated",
        "source_url": "https://en.wikipedia.org/wiki/Benjamin_Netanyahu"
    },
    "Bill Maher": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein; minimal documented connection",
        "source_url": "https://en.wikipedia.org/wiki/Bill_Maher"
    },
    "Justice Clarence Thomas": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein",
        "source_url": "https://en.wikipedia.org/wiki/Clarence_Thomas"
    },
    "Ellen DeGeneres": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein; retired from TV independently",
        "source_url": "https://en.wikipedia.org/wiki/Ellen_DeGeneres"
    },
    "John and Tony Podesta": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein; subject of conspiracy theories",
        "source_url": ""
    },
    "Justin Trudeau": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein; resigned as PM for unrelated political reasons",
        "source_url": "https://en.wikipedia.org/wiki/Justin_Trudeau"
    },
    "Larry Silverstein": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein",
        "source_url": "https://en.wikipedia.org/wiki/Larry_Silverstein"
    },
    "Mark Landon": {
        "consequence_tier": 0,
        "consequence_description": "No publicly reported consequences",
        "source_url": ""
    },
    "Barack Obama": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein; minimal documented connection",
        "source_url": "https://en.wikipedia.org/wiki/Barack_Obama"
    },
    "RFK": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein",
        "source_url": "https://en.wikipedia.org/wiki/Robert_F._Kennedy_Jr."
    },
    "Savannah Guthrie": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein; minimal documented connection",
        "source_url": "https://en.wikipedia.org/wiki/Savannah_Guthrie"
    },
    "Ted Leonsis": {
        "consequence_tier": 0,
        "consequence_description": "No consequences related to Epstein",
        "source_url": "https://en.wikipedia.org/wiki/Ted_Leonsis"
    },
}


class ConsequenceScraper:
    """Scraper for consequence data from multiple sources."""

    def __init__(self) -> None:
        """Initialize the consequence scraper."""
        self.wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='EpsteinAccountabilityIndex/1.0 (Educational Research)'
        )

    def _is_epstein_related(self, text: str, keyword: str) -> bool:
        """
        Check if a consequence keyword appears in Epstein-related context.

        Looks for Epstein context keywords within 500 characters of the
        consequence keyword to reduce false positives.

        Args:
            text: Full text to search
            keyword: The consequence keyword found

        Returns:
            True if the keyword appears near Epstein-related context
        """
        text_lower = text.lower()
        keyword_positions = [
            m.start() for m in re.finditer(re.escape(keyword), text_lower)
        ]

        for pos in keyword_positions:
            window_start = max(0, pos - 500)
            window_end = min(len(text_lower), pos + 500)
            window = text_lower[window_start:window_end]

            for ctx in EPSTEIN_CONTEXT:
                if ctx in window:
                    return True

        return False

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

            # Check for hard consequences (Epstein-related context preferred)
            for keyword in CONSEQUENCE_KEYWORDS_HARD:
                if keyword in text:
                    idx = text.index(keyword)
                    context_start = max(0, idx - 150)
                    context_end = min(len(text), idx + 150)
                    context = text[context_start:context_end]

                    if self._is_epstein_related(text, keyword):
                        return (
                            2,
                            f"Hard consequence (Epstein-related): {keyword}. Context: ...{context}...",
                            url
                        )

            # Check for soft consequences (Epstein-related context preferred)
            for keyword in CONSEQUENCE_KEYWORDS_SOFT:
                if keyword in text:
                    idx = text.index(keyword)
                    context_start = max(0, idx - 150)
                    context_end = min(len(text), idx + 150)
                    context = text[context_start:context_end]

                    if self._is_epstein_related(text, keyword):
                        return (
                            1,
                            f"Soft consequence (Epstein-related): {keyword}. Context: ...{context}...",
                            url
                        )

            # Fallback: check keywords without Epstein context
            for keyword in CONSEQUENCE_KEYWORDS_HARD:
                if keyword in text:
                    idx = text.index(keyword)
                    context_start = max(0, idx - 150)
                    context_end = min(len(text), idx + 150)
                    context = text[context_start:context_end]
                    return (
                        2,
                        f"Hard consequence (general): {keyword}. Context: ...{context}...",
                        url
                    )

            for keyword in CONSEQUENCE_KEYWORDS_SOFT:
                if keyword in text:
                    idx = text.index(keyword)
                    context_start = max(0, idx - 150)
                    context_end = min(len(text), idx + 150)
                    context = text[context_start:context_end]
                    return (
                        1,
                        f"Soft consequence (general): {keyword}. Context: ...{context}...",
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
            items = soup.find_all("item", limit=5)

            if not items:
                logger.debug(f"No news articles found for {name}")
                return None

            for item in items:
                title = item.title.text.lower() if item.title else ""
                description = item.description.text.lower() if item.description else ""
                link = item.link.text if item.link else ""

                combined_text = f"{title} {description}"

                for keyword in CONSEQUENCE_KEYWORDS_HARD:
                    if keyword in combined_text:
                        return (
                            2,
                            f"News: Hard consequence - {title[:100]}",
                            link
                        )

                for keyword in CONSEQUENCE_KEYWORDS_SOFT:
                    if keyword in combined_text:
                        return (
                            1,
                            f"News: Soft consequence - {title[:100]}",
                            link
                        )

            return (0, "No consequences detected in news", rss_url)

        except Exception as e:
            logger.warning(f"Error searching Google News for {name}: {e}")
            return None

    def get_consequence_info(self, name: str) -> Dict[str, Any]:
        """
        Get consequence information for a person from all sources.

        Priority: manual overrides > Wikipedia > Google News.

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
    # Resolve output path relative to project root
    project_root = Path(__file__).resolve().parent.parent
    out = Path(output_path)
    if not out.is_absolute():
        out = project_root / out

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
            out.parent.mkdir(parents=True, exist_ok=True)
            temp_df.to_csv(str(out) + ".tmp", index=False)
            logger.info(f"Saved temporary results ({i}/{len(names)})")

    df = pd.DataFrame(results)

    # Create output directory if needed
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Saved {len(df)} consequence records to {out}")

    # Log statistics
    tier_counts = df["consequence_tier"].value_counts().sort_index()
    logger.info(f"Consequence tier distribution:\n{tier_counts}")

    # Clean up temp file
    temp_file = Path(str(out) + ".tmp")
    if temp_file.exists():
        temp_file.unlink()

    return df


def load_names_from_severity_file(
    severity_path: str = "data/processed/severity_scores.csv"
) -> List[str]:
    """
    Load person names from severity scores CSV or scraped JSON.

    Checks the processed CSV first, then falls back to the scraped JSON.

    Args:
        severity_path: Path to severity scores CSV

    Returns:
        List of person names
    """
    # Resolve paths relative to the project root (parent of scripts/)
    project_root = Path(__file__).resolve().parent.parent

    csv_path = Path(severity_path)
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df["name"].tolist()

    # Fallback: load from scraped JSON
    json_path = project_root / "data" / "scraped" / "epsteinoverview_scores.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            topics = json.load(f)
        # Filter non-person topics using the same set as scrape_severity.py
        non_person_topics = {
            "dentist", "gynecologist", "pregnant", "whoops",
            "beef jerky", "pizza", "cream cheese",
            "drugs", "bitcoin", "9/11", "zorro ranch",
            "baal and occult references", "israel and mossad",
            "dangene and jennie enterprise", "epstein suicide",
            "qatar", "lifetouch",
        }
        names = [
            t["name"] for t in topics
            if t["name"].lower().strip() not in non_person_topics
        ]
        logger.info(f"Loaded {len(names)} names from scraped JSON (CSV not found)")
        return names

    raise FileNotFoundError(
        f"No severity data found at {csv_path} or {json_path}. "
        "Run 'python main.py scrape-severity' first."
    )


def main() -> None:
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape consequence data for Epstein-linked individuals"
    )
    parser.add_argument(
        "--input", default=None,
        help="Input CSV with names (defaults to severity_scores.csv)"
    )
    parser.add_argument(
        "--output", default="data/processed/consequences.csv",
        help="Output CSV file path"
    )
    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input)
        names = df["name"].tolist()
    else:
        names = load_names_from_severity_file()

    logger.info(f"Loaded {len(names)} names")
    scrape_consequences(names, args.output)


if __name__ == "__main__":
    main()
