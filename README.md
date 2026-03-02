# The Accountability Gap

## Overview
This project investigates whether the severity of an individual's mention in the DOJ-released Epstein case files correlates with real-world consequences they faced (resignations, arrests, convictions, etc.).

## Project Structure
```
epstein-accountability-index/
├── scripts/          # Data processing and model training scripts
├── models/           # Trained models (stored on Google Drive)
├── data/
│   ├── raw/         # Raw JSON files from EpsteinProcessor
│   ├── processed/   # Feature matrices and labels
│   └── outputs/     # Model predictions and experiment results
├── notebooks/       # Exploratory analysis notebooks
└── app/            # Flask web application
```

## Setup

### Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader vader_lexicon punkt
```

### Download Data
```bash
python main.py download-data
```

## Usage

### CLI Commands
```bash
# Download raw data from Google Drive
python main.py download-data

# Scrape severity scores
python main.py scrape-severity

# Scrape consequence labels
python main.py scrape-consequences

# Build feature matrix
python main.py build-features

# Train all models
python main.py train-models

# Run complete pipeline
python main.py run-all
```

### Web Application
```bash
cd app
python main.py
# Visit http://localhost:5000
```

## Models
1. **Naive Baseline**: Most frequent class predictor
2. **XGBoost**: Gradient boosting classifier with hyperparameter tuning
3. **DistilBERT**: Fine-tuned transformer model on context windows

## Data Sources
- **Document Corpus**: Epstein case files (via EpsteinProcessor)
- **Severity Scores**: epsteinoverview.com
- **Consequence Labels**: Wikipedia + Google News

## Features
- `mention_count`: Documents mentioning the individual
- `total_mentions`: Total name appearances
- `mean_context_sentiment`: VADER sentiment around mentions
- `cooccurrence_score`: Co-occurrence with incriminating keywords
- `doc_type_diversity`: Document category diversity
- `severity_score`: Scraped severity rating
- `name_in_subject_line`: Email subject line appearances

## License
Educational use only - Duke AIPI Graduate ML Course

## Attribution
Scaffolded with AI assistance (Claude, Anthropic)
