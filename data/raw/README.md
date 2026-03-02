# Raw Data Directory

This directory contains the raw aggregated JSON files from EpsteinProcessor.

## Download Instructions

Run the following command to download the raw data from Google Drive:

```bash
python main.py download-data
```

Or directly:

```bash
python scripts/make_dataset.py
```

## File Format

Each `ds*_agg.json` file contains:
- **Key**: Document ID
- **Value**: Dictionary with:
  - `pages`: Number of pages
  - `text`: Full extracted text
  - `success`: Boolean indicating successful extraction
  - `status_reason`: Extraction status message

## Note

These files are NOT committed to git due to their size. They are automatically downloaded when needed.
