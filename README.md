# The Accountability Gap

An NLP project investigating whether the severity of an individual's mention in the DOJ-released Epstein case files correlates with real-world consequences they faced (resignations, arrests, convictions, etc.).

**Author:** Lindsay Gross | Duke AIPI Graduate ML Course

## Pipeline Status

| Step | Script | Status | Output |
|------|--------|--------|--------|
| 1. Download data | `scripts/make_dataset.py` | Done | `data/raw/ds{8,9,10,12}_agg.json` (2,935 docs) |
| 2. Scrape severity | `scripts/scrape_severity.py` | Done | `data/scraped/epsteinoverview_scores.json` (66 people) |
| 3. Scrape consequences | `scripts/scrape_consequences.py` | Not started | `data/processed/consequences.csv` |
| 4. Build features | `scripts/build_features.py` | Not started | `data/processed/features.csv` |
| 5. Train models | `scripts/model.py` | Not started | `models/`, `data/outputs/` |
| 6. Web app | `app/main.py` | Scaffolded | http://localhost:5000 |

## What's Been Built

### Step 1 - Data Ingestion (`make_dataset.py`)
Aggregates raw Epstein case documents from the local [EpsteinProcessor](../EpsteinProcessor) output into the `ds*_agg.json` format expected by the pipeline. Also supports Google Drive download for deployment.

- **Local mode** (`--local`): Reads `scan_results.json` from each EpsteinProcessor topic directory, groups by dataset, and writes `ds*_agg.json` files
- **Google Drive mode** (default): Downloads via `gdown` using file IDs (placeholder TODOs until files are uploaded)
- **Output**: 4 dataset files with 2,935 total documents across ds8, ds9, ds10, and ds12

### Step 2 - Severity Scores (`scrape_severity.py`)
Extracts concern scores from [epsteinoverview.com](https://epsteinoverview.com), a React SPA displaying AI-generated summaries of key people and topics from the Epstein files.

- Scraped all **83 topics** from the live site, saved to `data/scraped/epsteinoverview_scores.json`
- Filtered to **66 named individuals** (removed non-person topics like "Dentist", "Pizza", "Bitcoin")
- Scores range from 0 to 10 on a concern scale (Critical / Very High / High / Moderate / Low)
- Top scored: Donald Trump (10.0), Ghislaine Maxwell (10.0), Jes Staley (10.0), Prince Andrew (9.9), Leon Black (9.9)

### Remaining Steps (Scaffolded, Not Yet Run)
- **Step 3**: Scrape Wikipedia + Google News for consequence labels (resigned, arrested, convicted, etc.)
- **Step 4**: NER-based feature extraction using spaCy + VADER sentiment
- **Step 5**: Train Naive Baseline, XGBoost, and fine-tuned DistilBERT classifiers
- **Step 6**: Flask web app with search, scatter plot, and model comparison dashboard

## Project Structure
```
epstein-accountability-index/
├── main.py                          <- CLI entry point (argparse)
├── requirements.txt
├── setup.py
├── scripts/
│   ├── make_dataset.py              <- Download/aggregate raw data
│   ├── scrape_severity.py           <- Severity scores from epsteinoverview.com
│   ├── scrape_consequences.py       <- Consequence labels from Wikipedia/news
│   ├── build_features.py            <- NER + feature extraction
│   └── model.py                     <- Train/evaluate 3 models
├── data/
│   ├── raw/                         <- ds*_agg.json files (gitignored)
│   ├── scraped/                     <- Pre-scraped site data (committed)
│   ├── processed/                   <- Feature matrices and labels
│   └── outputs/                     <- Predictions and experiment results
├── models/                          <- Trained models (on GDrive, not in git)
├── notebooks/                       <- Exploratory analysis
└── app/                             <- Flask web application
```

## Setup

### Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### CLI Commands
```bash
# Step 1: Aggregate data from local EpsteinProcessor
python3 main.py download-data --local

# Step 2: Extract severity scores (reads from pre-scraped snapshot)
python3 main.py scrape-severity

# Step 3: Scrape consequence labels
python3 main.py scrape-consequences

# Step 4: Build feature matrix
python3 main.py build-features

# Step 5: Train all models
python3 main.py train-models --run-experiment

# Run complete pipeline
python3 main.py run-all --local
```

### Web Application
```bash
cd app
python3 main.py
# Visit http://localhost:5000
```

## Models
1. **Naive Baseline**: Most frequent class predictor (sklearn DummyClassifier)
2. **XGBoost**: Gradient boosting with GridSearchCV hyperparameter tuning
3. **DistilBERT**: Fine-tuned transformer on context windows around name mentions

## Data Sources
- **Document Corpus**: 2,935 Epstein case files via EpsteinProcessor (DOJ releases)
- **Severity Scores**: 66 individuals scraped from epsteinoverview.com (0-10 concern scale)
- **Consequence Labels**: Wikipedia + Google News RSS (planned)

## NLP Features (per person)
| Feature | Description |
|---------|-------------|
| `mention_count` | Number of documents mentioning the individual |
| `total_mentions` | Total name appearances across all documents |
| `mean_context_sentiment` | Average VADER sentiment of surrounding sentences |
| `cooccurrence_score` | Co-occurrence with keywords: minor, massage, flight, island, etc. |
| `doc_type_diversity` | Number of distinct document types (email, deposition, etc.) |
| `severity_score` | Concern score from epsteinoverview.com |
| `name_in_subject_line` | Whether name appears in email subject lines |

## Research Question
> **Does power protect?** Does the correlation between severity of mention and real-world consequences weaken for high-power individuals (politicians, CEOs, royalty)?

## License
Educational use only - Duke AIPI Graduate ML Course

## Attribution
Scaffolded with AI assistance (Claude, Anthropic)
