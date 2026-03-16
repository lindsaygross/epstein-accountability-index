# Project: The Impunity Index
# Authors: Lindsay Gross, Shreya Mendi, Andrew Jin
# Advisor: Brinnae Bent, PhD
# Claude chat: https://claude.ai/chat/f8744002-3279-48ab-9d9a-8efa1fdb1af1
# Built with Claude AI assistance

"""
Scrape headshot images for all tracked individuals from Wikipedia.

Uses the MediaWiki pageimages API to download 200x200 thumbnails.
Generates dark-themed placeholder PNGs with initials for people
without Wikipedia images.

Output:
    app/static/images/people/{slug}.jpg   (Wikipedia images)
    app/static/images/people/{slug}.png   (Placeholder images)
    app/static/images/people/images_manifest.json  (name -> filename mapping)
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, Optional
from io import BytesIO

import requests
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Wikipedia name overrides for compound/unusual names
WIKIPEDIA_NAME_OVERRIDES: Dict[str, str] = {
    "Oren, Alon, and Tal Alexander": "Oren Alexander",
    "George H.W. Bush and George W. Bush": "George W. Bush",
    "John and Tony Podesta": "John Podesta",
    "RFK": "Robert F. Kennedy Jr.",
    "Justice Clarence Thomas": "Clarence Thomas",
    "Oprah": "Oprah Winfrey",
    "Peter Theil": "Peter Thiel",
    "Bill O'Reilly": "Bill O'Reilly (political commentator)",
    "Sheikh Sultan Bin Jassim Al Thani": "Sultan bin Jassim Al Thani",
    "Sultan Bin Sulayem": "Sultan Ahmed bin Sulayem",
}

# Non-person topics to skip (from scrape_severity.py)
NON_PERSON_TOPICS = {
    "Dentist", "Pilot", "Pizza", "McDonald's", "Bitcoin",
    "Lolita Express", "Zorro Ranch", "Epstein Island",
    "Little St. James", "Bear Stearns", "Deutsche Bank",
    "JP Morgan", "Victoria's Secret", "MIT Media Lab",
    "Harvard University", "Ohio State", "Dalton School"
}

REQUEST_DELAY = 0.5  # seconds between Wikipedia API calls


def slugify_name(name: str) -> str:
    """Convert 'Donald Trump' -> 'donald-trump' for filenames."""
    slug = name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    return slug


def get_wikipedia_image_url(name: str, thumb_size: int = 200) -> Optional[str]:
    """
    Query Wikipedia pageimages API for a person's primary image.
    Returns thumbnail URL or None.
    """
    wiki_name = WIKIPEDIA_NAME_OVERRIDES.get(name, name)

    params = {
        "action": "query",
        "titles": wiki_name,
        "prop": "pageimages",
        "format": "json",
        "pithumbsize": thumb_size,
        "redirects": 1,
    }

    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params=params,
            headers={"User-Agent": "AccountabilityGapBot/1.0 (academic project)"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return None
            thumb = page_data.get("thumbnail", {})
            return thumb.get("source")

    except Exception as e:
        logger.warning(f"Wikipedia API error for '{name}': {e}")

    return None


def download_image(url: str, output_path: Path) -> bool:
    """Download image from URL, crop to square, save as JPEG."""
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "AccountabilityGapBot/1.0 (academic project)"
        })
        resp.raise_for_status()

        img = Image.open(BytesIO(resp.content))
        img = img.convert("RGB")

        # Crop to center square
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))

        # Resize to 200x200
        img = img.resize((200, 200), Image.LANCZOS)
        img.save(output_path, "JPEG", quality=85)
        return True

    except Exception as e:
        logger.warning(f"Download failed for {url}: {e}")
        return False


def generate_placeholder(name: str, output_path: Path, size: int = 200) -> None:
    """Generate a dark-themed placeholder PNG with initials."""
    # Get initials (max 2 characters)
    parts = name.split()
    if len(parts) >= 2:
        initials = (parts[0][0] + parts[-1][0]).upper()
    elif parts:
        initials = parts[0][0].upper()
    else:
        initials = "?"

    # Create image
    img = Image.new("RGB", (size, size), color=(26, 26, 26))  # #1a1a1a
    draw = ImageDraw.Draw(img)

    # Try to use a decent font, fall back to default
    font_size = size // 3
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Center the text
    bbox = draw.textbbox((0, 0), initials, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size - text_w) // 2
    y = (size - text_h) // 2 - bbox[1]

    draw.text((x, y), initials, fill=(136, 136, 136), font=font)  # #888888
    img.save(output_path, "PNG")


def scrape_all_images(
    scores_path: str = "data/scraped/epsteinoverview_scores.json",
    output_dir: str = "app/static/images/people",
) -> Dict[str, str]:
    """
    Scrape images for all tracked individuals.
    Returns mapping of name -> filename.
    """
    base_path = Path(__file__).parent.parent
    scores_file = base_path / scores_path
    out_dir = base_path / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load names
    with open(scores_file) as f:
        scores = json.load(f)

    names = [s["name"] for s in scores if s["name"] not in NON_PERSON_TOPICS]
    logger.info(f"Processing {len(names)} individuals for images")

    manifest: Dict[str, str] = {}
    found = 0
    placeholder_count = 0

    for i, name in enumerate(names):
        slug = slugify_name(name)
        jpg_path = out_dir / f"{slug}.jpg"
        png_path = out_dir / f"{slug}.png"

        # Check if already downloaded
        if jpg_path.exists():
            manifest[name] = f"{slug}.jpg"
            found += 1
            continue
        if png_path.exists():
            manifest[name] = f"{slug}.png"
            placeholder_count += 1
            continue

        # Try Wikipedia
        logger.info(f"[{i+1}/{len(names)}] Fetching image for {name}...")
        url = get_wikipedia_image_url(name)

        if url:
            if download_image(url, jpg_path):
                manifest[name] = f"{slug}.jpg"
                found += 1
                logger.info(f"  ✓ Downloaded from Wikipedia")
            else:
                generate_placeholder(name, png_path)
                manifest[name] = f"{slug}.png"
                placeholder_count += 1
                logger.info(f"  ⊘ Download failed, placeholder generated")
        else:
            generate_placeholder(name, png_path)
            manifest[name] = f"{slug}.png"
            placeholder_count += 1
            logger.info(f"  ⊘ No Wikipedia image, placeholder generated")

        time.sleep(REQUEST_DELAY)

    # Save manifest
    manifest_path = out_dir / "images_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\nDone! {found} Wikipedia images, {placeholder_count} placeholders")
    logger.info(f"Manifest saved to {manifest_path}")

    return manifest


if __name__ == "__main__":
    scrape_all_images()
