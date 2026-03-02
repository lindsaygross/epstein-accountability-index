# Attribution: Scaffolded with AI assistance (Claude, Anthropic)

"""
Scrape severity scores from epsteinoverview.com.

This script extracts person names, severity scores, and profile URLs
from the Epstein Overview website.
"""

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


def scrape_person_list(base_url: str = BASE_URL) -> List[Dict[str, str]]:
    """
    Scrape the main person list from epsteinoverview.com.

    Args:
        base_url: Base URL of the website

    Returns:
        List of dictionaries with person data
    """
    logger.info(f"Scraping person list from {base_url}")

    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {base_url}: {e}")
        raise

    soup = BeautifulSoup(response.content, "html.parser")
    people = []

    # TODO: Update selectors based on actual website structure
    # This is a placeholder implementation
    person_cards = soup.find_all("div", class_="person-card")

    for card in person_cards:
        try:
            name_elem = card.find("h3", class_="person-name")
            score_elem = card.find("span", class_="severity-score")
            link_elem = card.find("a", class_="profile-link")

            if name_elem and score_elem:
                person = {
                    "name": name_elem.text.strip(),
                    "severity_score": float(score_elem.text.strip()),
                    "profile_url": (
                        f"{base_url}{link_elem['href']}"
                        if link_elem and "href" in link_elem.attrs
                        else None
                    )
                }
                people.append(person)
                logger.debug(f"Found person: {person['name']} (score: {person['severity_score']})")
        except (AttributeError, ValueError) as e:
            logger.warning(f"Failed to parse person card: {e}")
            continue

        # Polite delay
        time.sleep(REQUEST_DELAY)

    logger.info(f"Scraped {len(people)} people")
    return people


def check_pagination(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    """
    Check if there is a next page and return its URL.

    Args:
        soup: BeautifulSoup object of current page
        base_url: Base URL of the website

    Returns:
        URL of next page or None
    """
    next_button = soup.find("a", class_="next-page")
    if next_button and "href" in next_button.attrs:
        return f"{base_url}{next_button['href']}"
    return None


def scrape_all_pages(base_url: str = BASE_URL) -> List[Dict[str, str]]:
    """
    Scrape all pages with pagination support.

    Args:
        base_url: Base URL of the website

    Returns:
        Combined list of all people across pages
    """
    all_people = []
    current_url = base_url
    page_num = 1

    while current_url:
        logger.info(f"Scraping page {page_num}")

        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            # Scrape people from current page
            people = scrape_person_list(current_url)
            all_people.extend(people)

            # Check for next page
            current_url = check_pagination(soup, base_url)
            page_num += 1

            if current_url:
                time.sleep(REQUEST_DELAY)

        except requests.RequestException as e:
            logger.error(f"Failed to scrape page {page_num}: {e}")
            break

    return all_people


def scrape_severity_scores(output_path: str = "data/processed/severity_scores.csv") -> pd.DataFrame:
    """
    Main function to scrape severity scores and save to CSV.

    Args:
        output_path: Path to save the output CSV file

    Returns:
        DataFrame with severity scores
    """
    logger.info("Starting severity score scraping")

    # Scrape all people
    people = scrape_person_list()

    # Convert to DataFrame
    df = pd.DataFrame(people)

    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} severity scores to {output_path}")

    # Log statistics
    logger.info(f"Severity score statistics:\n{df['severity_score'].describe()}")

    return df


def main() -> None:
    """Main entry point for the script."""
    scrape_severity_scores()


if __name__ == "__main__":
    main()
