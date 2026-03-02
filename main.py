# Attribution: Scaffolded with AI assistance (Claude, Anthropic)

"""
Main CLI entry point for the Epstein Accountability Index.

This script provides a command-line interface for all pipeline operations.
"""

import argparse
import logging
import sys

from scripts.make_dataset import load_datasets
from scripts.scrape_severity import scrape_severity_scores
from scripts.scrape_consequences import scrape_consequences, load_names_from_severity_file
from scripts.build_features import build_feature_matrix
from scripts.model import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cmd_download_data(args: argparse.Namespace) -> None:
    """
    Download raw data from Google Drive or aggregate from local EpsteinProcessor.

    Args:
        args: Command-line arguments
    """
    logger.info("Loading raw data...")
    load_datasets(
        output_dir=args.output_dir,
        local=args.local,
        processor_dir=getattr(args, 'processor_dir', None)
    )
    logger.info("Data loading complete")


def cmd_scrape_severity(args: argparse.Namespace) -> None:
    """
    Scrape severity scores from epsteinoverview.com.

    Args:
        args: Command-line arguments
    """
    logger.info("Scraping severity scores...")
    scrape_severity_scores(args.output)
    logger.info("Scraping complete")


def cmd_scrape_consequences(args: argparse.Namespace) -> None:
    """
    Scrape consequence labels from Wikipedia and news.

    Args:
        args: Command-line arguments
    """
    logger.info("Scraping consequences...")

    if args.input:
        # Load names from file
        import pandas as pd
        df = pd.read_csv(args.input)
        names = df['name'].tolist()
    else:
        # Load from severity scores
        names = load_names_from_severity_file()

    scrape_consequences(names, args.output)
    logger.info("Scraping complete")


def cmd_build_features(args: argparse.Namespace) -> None:
    """
    Build feature matrix from document corpus.

    Args:
        args: Command-line arguments
    """
    logger.info("Building feature matrix...")
    build_feature_matrix(
        raw_data_dir=args.raw_data_dir,
        severity_path=args.severity_path,
        output_path=args.output
    )
    logger.info("Feature extraction complete")


def cmd_train_models(args: argparse.Namespace) -> None:
    """
    Train all models.

    Args:
        args: Command-line arguments
    """
    logger.info("Training models...")
    trainer = ModelTrainer(
        features_path=args.features_path,
        consequences_path=args.consequences_path
    )

    # Train all models
    trainer.train_all_models()

    # Evaluate
    trainer.evaluate_all_models()

    # Run experiment
    if args.run_experiment:
        trainer.run_experiment()

    logger.info("Training complete")


def cmd_run_all(args: argparse.Namespace) -> None:
    """
    Run complete pipeline.

    Args:
        args: Command-line arguments
    """
    logger.info("Running complete pipeline...")

    # Step 1: Download data
    logger.info("Step 1/5: Loading data...")
    load_datasets(local=getattr(args, 'local', False))

    # Step 2: Scrape severity scores
    logger.info("Step 2/5: Scraping severity scores...")
    scrape_severity_scores()

    # Step 3: Scrape consequences
    logger.info("Step 3/5: Scraping consequences...")
    names = load_names_from_severity_file()
    scrape_consequences(names)

    # Step 4: Build features
    logger.info("Step 4/5: Building features...")
    build_feature_matrix()

    # Step 5: Train models
    logger.info("Step 5/5: Training models...")
    trainer = ModelTrainer()
    trainer.train_all_models()
    trainer.evaluate_all_models()
    trainer.run_experiment()

    logger.info("Pipeline complete!")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Epstein Accountability Index - ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download raw data
  python main.py download-data

  # Run complete pipeline
  python main.py run-all

  # Train models only
  python main.py train-models

For more information, see README.md
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Download data command
    parser_download = subparsers.add_parser(
        'download-data',
        help='Download raw data from Google Drive'
    )
    parser_download.add_argument(
        '--output-dir',
        default='data/raw',
        help='Output directory for downloaded files'
    )
    parser_download.add_argument(
        '--local', action='store_true',
        help='Aggregate from local EpsteinProcessor output instead of Google Drive'
    )
    parser_download.add_argument(
        '--processor-dir', default=None,
        help='Path to EpsteinProcessor directory (used with --local)'
    )
    parser_download.set_defaults(func=cmd_download_data)

    # Scrape severity command
    parser_severity = subparsers.add_parser(
        'scrape-severity',
        help='Scrape severity scores from epsteinoverview.com'
    )
    parser_severity.add_argument(
        '--output',
        default='data/processed/severity_scores.csv',
        help='Output CSV file path'
    )
    parser_severity.set_defaults(func=cmd_scrape_severity)

    # Scrape consequences command
    parser_consequences = subparsers.add_parser(
        'scrape-consequences',
        help='Scrape consequence labels from Wikipedia and news'
    )
    parser_consequences.add_argument(
        '--input',
        help='Input CSV with names (optional, defaults to severity_scores.csv)'
    )
    parser_consequences.add_argument(
        '--output',
        default='data/processed/consequences.csv',
        help='Output CSV file path'
    )
    parser_consequences.set_defaults(func=cmd_scrape_consequences)

    # Build features command
    parser_features = subparsers.add_parser(
        'build-features',
        help='Build feature matrix from document corpus'
    )
    parser_features.add_argument(
        '--raw-data-dir',
        default='data/raw',
        help='Directory containing raw JSON files'
    )
    parser_features.add_argument(
        '--severity-path',
        default='data/processed/severity_scores.csv',
        help='Path to severity scores CSV'
    )
    parser_features.add_argument(
        '--output',
        default='data/processed/features.csv',
        help='Output CSV file path'
    )
    parser_features.set_defaults(func=cmd_build_features)

    # Train models command
    parser_train = subparsers.add_parser(
        'train-models',
        help='Train all classification models'
    )
    parser_train.add_argument(
        '--features-path',
        default='data/processed/features.csv',
        help='Path to features CSV'
    )
    parser_train.add_argument(
        '--consequences-path',
        default='data/processed/consequences.csv',
        help='Path to consequences CSV'
    )
    parser_train.add_argument(
        '--run-experiment',
        action='store_true',
        help='Run power tier experiment'
    )
    parser_train.set_defaults(func=cmd_train_models)

    # Run all command
    parser_all = subparsers.add_parser(
        'run-all',
        help='Run complete pipeline'
    )
    parser_all.add_argument(
        '--local', action='store_true',
        help='Aggregate from local EpsteinProcessor output instead of Google Drive'
    )
    parser_all.set_defaults(func=cmd_run_all)

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except Exception as e:
            logger.error(f"Error executing command: {e}", exc_info=True)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
