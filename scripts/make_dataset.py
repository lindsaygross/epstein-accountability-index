# Attribution: Scaffolded with AI assistance (Claude, Anthropic)

"""
Download raw aggregated JSON files from Google Drive using gdown.

This script downloads the processed Epstein document files (ds*_agg.json)
from Google Drive and saves them to the data/raw/ directory.
"""

import logging
import os
from pathlib import Path
from typing import Dict

import gdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# TODO: Replace with actual Google Drive file IDs
GDRIVE_FILE_IDS: Dict[str, str] = {
    "ds8_agg.json": "TODO_FILE_ID_1",
    "ds9_agg.json": "TODO_FILE_ID_2",
    # Add more file IDs as needed
}


def download_datasets(output_dir: str = "data/raw") -> None:
    """
    Download all dataset files from Google Drive.

    Args:
        output_dir: Directory to save downloaded files

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading datasets to {output_dir}")

    for filename, file_id in GDRIVE_FILE_IDS.items():
        output_file = output_path / filename

        # Skip if file already exists
        if output_file.exists():
            logger.info(f"File {filename} already exists, skipping download")
            continue

        # Check for TODO placeholder
        if file_id.startswith("TODO"):
            logger.warning(
                f"File ID for {filename} is a placeholder. "
                "Please update GDRIVE_FILE_IDS with actual file IDs."
            )
            continue

        # Download file
        try:
            logger.info(f"Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(output_file), quiet=False)
            logger.info(f"Successfully downloaded {filename}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise

    logger.info("Dataset download complete")


def main() -> None:
    """Main entry point for the script."""
    download_datasets()


if __name__ == "__main__":
    main()
