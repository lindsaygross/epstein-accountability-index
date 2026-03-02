# Attribution: Scaffolded with AI assistance (Claude, Anthropic)

"""
Download raw aggregated JSON files from Google Drive using gdown,
or aggregate them locally from EpsteinProcessor output.

This script downloads the processed Epstein document files (ds*_agg.json)
from Google Drive and saves them to the data/raw/ directory. Alternatively,
it can aggregate data directly from a local EpsteinProcessor output directory.
"""

import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# TODO: Replace with actual Google Drive file IDs after uploading
GDRIVE_FILE_IDS: Dict[str, str] = {
    "ds8_agg.json": "TODO_FILE_ID_1",
    "ds9_agg.json": "TODO_FILE_ID_2",
    # Add more file IDs as needed
}

def _find_processor_dir() -> str:
    """
    Locate the EpsteinProcessor directory relative to this project.

    Walks up from the script location looking for a sibling directory
    named EpsteinProcessor. Works from both the main repo and git worktrees.

    Returns:
        Absolute path to EpsteinProcessor directory, or a placeholder if not found.
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):
        current = current.parent
        candidate = current / "EpsteinProcessor"
        if candidate.is_dir():
            return str(candidate)
    return str(Path(__file__).resolve().parent.parent.parent / "EpsteinProcessor")


DEFAULT_PROCESSOR_DIR = _find_processor_dir()


def aggregate_from_local(
    processor_dir: str = DEFAULT_PROCESSOR_DIR,
    output_dir: str = "data/raw"
) -> None:
    """
    Aggregate data from a local EpsteinProcessor output directory.

    Reads scan_results.json from each topic subdirectory, groups entries
    by their dataset field, and writes ds*_agg.json files in the format
    expected by build_features.py.

    Args:
        processor_dir: Path to the EpsteinProcessor root directory
        output_dir: Directory to save aggregated JSON files
    """
    processor_path = Path(processor_dir)
    topics_dir = processor_path / "topics"

    if not topics_dir.is_dir():
        raise FileNotFoundError(
            f"EpsteinProcessor topics directory not found at {topics_dir}. "
            "Provide the correct path with --processor-dir."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all scan results grouped by dataset
    dataset_docs: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    total_entries = 0

    for topic_dir in sorted(topics_dir.iterdir()):
        if not topic_dir.is_dir():
            continue

        scan_file = topic_dir / "scan_results.json"
        if not scan_file.exists():
            logger.warning(f"No scan_results.json in {topic_dir.name}, skipping")
            continue

        logger.info(f"Reading scan results from topic: {topic_dir.name}")
        with open(scan_file, "r") as f:
            entries = json.load(f)

        for entry in entries:
            dataset = entry.get("dataset", "unknown")
            filename = entry.get("filename", "")
            text = entry.get("text", "")

            if not filename or not text:
                continue

            # Transform to the format expected by build_features.py
            dataset_docs[dataset][filename] = {
                "text": text,
                "success": True,
                "pages": 1,
                "status_reason": "ok"
            }
            total_entries += 1

    if not dataset_docs:
        logger.warning("No documents found in EpsteinProcessor output")
        return

    # Write one ds*_agg.json file per dataset
    for dataset_name, docs in sorted(dataset_docs.items()):
        out_file = output_path / f"{dataset_name}_agg.json"

        if out_file.exists():
            logger.info(f"{out_file.name} already exists, skipping")
            continue

        with open(out_file, "w") as f:
            json.dump(docs, f, indent=2)
        logger.info(f"Wrote {out_file.name} with {len(docs)} documents")

    # Also copy final_topic_data.json if it exists (useful for severity/summaries)
    topic_data_src = processor_path / "final_topic_data.json"
    topic_data_dst = output_path / "final_topic_data.json"
    if topic_data_src.exists() and not topic_data_dst.exists():
        shutil.copy2(topic_data_src, topic_data_dst)
        logger.info("Copied final_topic_data.json")

    logger.info(
        f"Local aggregation complete: {total_entries} documents "
        f"across {len(dataset_docs)} datasets"
    )


def download_datasets(output_dir: str = "data/raw") -> None:
    """
    Download all dataset files from Google Drive.

    Args:
        output_dir: Directory to save downloaded files
    """
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
            import gdown  # Lazy import: only needed for GDrive mode
            logger.info(f"Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(output_file), quiet=False)
            logger.info(f"Successfully downloaded {filename}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise

    logger.info("Dataset download complete")


def load_datasets(
    output_dir: str = "data/raw",
    local: bool = False,
    processor_dir: Optional[str] = None
) -> None:
    """
    Load datasets using either local aggregation or Google Drive download.

    Args:
        output_dir: Directory to save data files
        local: If True, aggregate from local EpsteinProcessor output
        processor_dir: Path to EpsteinProcessor directory (local mode only)
    """
    if local:
        src = processor_dir or DEFAULT_PROCESSOR_DIR
        logger.info(f"Aggregating data locally from {src}")
        aggregate_from_local(processor_dir=src, output_dir=output_dir)
    else:
        download_datasets(output_dir=output_dir)


def main() -> None:
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download or aggregate Epstein case file datasets"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Aggregate from local EpsteinProcessor output instead of Google Drive"
    )
    parser.add_argument(
        "--processor-dir", default=None,
        help=f"Path to EpsteinProcessor directory (default: {DEFAULT_PROCESSOR_DIR})"
    )
    parser.add_argument(
        "--output-dir", default="data/raw",
        help="Output directory for data files"
    )
    args = parser.parse_args()

    load_datasets(
        output_dir=args.output_dir,
        local=args.local,
        processor_dir=args.processor_dir
    )


if __name__ == "__main__":
    main()
