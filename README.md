# The Impunity Index

An NLP-driven investigation analyzing 2,849 Epstein case documents with machine learning to measure the gap between documentary evidence and real-world consequences across 66 individuals.

**Author:** Lindsay Gross | Duke AIPI Graduate ML Course

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1. Download data | `scripts/make_dataset.py` | `data/raw/ds{8,9,10,12}_agg.json` (2,849 docs) |
| 2. Scrape severity | `scripts/scrape_severity.py` | `data/processed/severity_scores.csv` (66 people) |
| 3. Scrape consequences | `scripts/scrape_consequences.py` | `data/processed/consequences.csv` (66 people, 3 tiers) |
| 4. Build features | `scripts/build_features.py` | `data/processed/features.csv` (66 x 7 NLP features) |
| 5. Train models | `scripts/model.py` | `models/`, `data/outputs/model_metrics.json` |
| 6. Web app | `app/main.py` | http://localhost:5001 |

## NLP Approach

### Step 1 — Feature Extraction

spaCy NER identifies person mentions across 2,849 documents. For each of the 66 individuals, we extract:

| Feature | Description | Range |
|---------|-------------|-------|
| `mention_count` | Number of documents mentioning the individual | 0–578 |
| `total_mentions` | Total name appearances across all documents | 0–744 |
| `cooccurrence_score` | Co-occurrence with incriminating terms (trafficking, minor, abuse) | 0–126 |
| `doc_type_diversity` | Number of distinct document types (email, deposition, legal filing) | 0–5 |
| `name_in_subject_line` | Whether name appears in email subject lines (binary) | 0 or 1 |
| `mean_context_sentiment` | Average VADER sentiment of sentences surrounding mentions | -1 to 1 |

### Step 2 — Classification

Binary classification task: predict whether an individual faced real-world consequences (resignation, arrest, conviction) based on their NLP feature profile. Three models trained with 80/20 stratified split:

1. **Logistic Regression** (baseline) — L2-regularized with balanced class weights on 7 tabular NLP features
2. **Random Forest + TF-IDF** (best) — Combines 7 tabular features with TF-IDF bigrams (500 features) from document text
3. **Legal-BERT** (transformer) — Fine-tuned `nlpaueb/legal-bert-base-uncased` on document-level text classification with per-person aggregation

### Step 3 — Impunity Scoring

The **Impunity Index** replaces the previously scraped severity score from epsteinoverview.com. It is computed entirely from our NLP features and consequence data:

#### Evidence Index (0–10)

A weighted combination of normalized NLP features:

```
evidence_index = (
    0.30 × mention_count_norm +
    0.20 × cooccurrence_score_norm +
    0.15 × total_mentions_norm +
    0.15 × doc_type_diversity_norm +
    0.10 × name_in_subject_line_norm +
    0.10 × sentiment_norm
) × 10
```

Each feature is min-max normalized to [0, 1] across all 66 individuals. Sentiment is inverted (more negative = higher score) since negative sentiment around a person's mentions suggests more incriminating context.

#### Consequence Modifier

The evidence index is then adjusted based on what actually happened to the person:

| Consequence Tier | Modifier | Rationale |
|-----------------|----------|-----------|
| **Tier 0** — No consequence | × 1.3 (capped at 10) | *High impunity*: strong evidence but no justice = higher score |
| **Tier 1** — Soft consequence (resigned, sued, reputational damage) | × 1.0 | Neutral: partial consequence |
| **Tier 2** — Hard consequence (arrested, convicted, imprisoned) | × 0.7 | *Low impunity*: justice served, system worked |

#### Final Score

```
impunity_index = evidence_index × consequence_modifier
```

**Interpretation**: A high impunity index means strong documentary evidence with little consequence — the person "got away with it." A low score means either minimal evidence or the justice system responded appropriately.

#### Examples

| Individual | Evidence Index | Consequence Tier | Modifier | Impunity Index | Level |
|-----------|---------------|-----------------|----------|---------------------|-------|
| Donald Trump | 7.2 | 0 (None) | ×1.3 | **9.4** | Critical |
| Ghislaine Maxwell | 5.5 | 2 (Convicted) | ×0.7 | **3.8** | Moderate |
| Bill Gates | 3.4 | 1 (Soft) | ×1.0 | **3.4** | Moderate |
| Bill Clinton | 5.4 | 0 (None) | ×1.3 | **7.0** | High |

#### Impunity Levels

| Level | Score Range | Color |
|-------|-----------|-------|
| Critical | ≥ 7.5 | Red |
| High | 5.0–7.5 | Orange |
| Moderate | 2.5–5.0 | Blue |
| Low | 1.0–2.5 | Cyan |
| Minimal | < 1.0 | Gray |

### Step 4 — Evaluation

#### Metrics

Five complementary metrics evaluated per model:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Accuracy** | Overall correct predictions | Baseline reference, but misleading with imbalanced classes |
| **F1 Macro** | Harmonic mean of precision/recall, averaged across classes | Balances both classes equally regardless of size |
| **MCC** | Matthews Correlation Coefficient (-1 to +1) | Best single metric for imbalanced binary classification; accounts for all 4 confusion matrix quadrants |
| **Precision** | Of those predicted positive, how many are correct | Measures false alarm rate |
| **Recall** | Of actual positives, how many were found | Measures miss rate |

#### Model Results

| Model | Accuracy | F1 Macro | MCC | Precision | Recall | Test Size |
|-------|----------|----------|-----|-----------|--------|-----------|
| Logistic Regression | 71.4% | 65.0% | 0.337 | 40.0% | 66.7% | 14 |
| **Random Forest + TF-IDF** | **78.6%** | **63.5%** | **0.284** | **50.0%** | **33.3%** | **14** |
| Legal-BERT | 45.7% | 41.8% | 0.135 | 40.9% | 93.1% | 151 |

**Best model**: Random Forest + TF-IDF achieves the highest accuracy (78.6%) and precision (50.0%), though Logistic Regression has the best F1 and MCC due to better recall balance.

**Legal-BERT underperformance**: Despite being the most sophisticated model, Legal-BERT achieves only 45.7% accuracy (below the 77.3% majority-class baseline). It over-predicts the positive class (93.1% recall but only 16.1% specificity), likely because fine-tuning on only 66 person-level examples is insufficient for a 110M-parameter transformer.

#### Stress Tests

**Class Imbalance Analysis**:
- 51 individuals face no consequences vs only 15 with consequences (3.4:1 ratio)
- A naive majority-class baseline achieves 77.3% accuracy — our models must beat this to be useful
- With only ~3 positive samples per test fold, single misclassifications swing F1 by ±15-20%

**Feature Ablation Study**:
- Removing `cooccurrence_score` causes the largest F1 drop (-7.4%), confirming it as the most predictive feature
- NLP-only features (without severity score) achieve F1 of 71.4%, showing the model doesn't rely on external scores
- Most individual features show minimal impact when removed, suggesting redundancy across mention-based features

**Power Tier Experiment**:
- Individuals grouped by power level (political, financial, celebrity, private)
- Tests whether the evidence→consequence correlation weakens for high-power individuals

#### Limitations and Improvement Paths

| Limitation | Impact | Improvement Path |
|-----------|--------|-----------------|
| **Class Imbalance** (3.4:1) | Models biased toward majority class; accuracy is misleading | SMOTE oversampling or collecting more positive examples through expanded document sources |
| **Small Sample** (n=66) | Insufficient data for transformer fine-tuning; unstable cross-validation | Expand to associates, witnesses, and unnamed individuals in documents |
| **Legal-BERT Underperformance** | 45.7% accuracy, below majority baseline | Use document-level pre-training, increase training data, or use as feature extractor instead of classifier |
| **Feature Correlation** | Mention count and total mentions are highly correlated (r=0.95) | Apply PCA or drop redundant features; focus on co-occurrence and doc diversity |

## Project Structure

```
epstein-accountability-index/
├── main.py                          <- CLI entry point (argparse)
├── requirements.txt
├── scripts/
│   ├── make_dataset.py              <- Download/aggregate raw data
│   ├── scrape_severity.py           <- Severity scores from epsteinoverview.com
│   ├── scrape_consequences.py       <- Consequence labels (tier 0/1/2)
│   ├── build_features.py            <- NER + feature extraction
│   └── model.py                     <- Train/evaluate 3 models
├── data/
│   ├── raw/                         <- ds*_agg.json files (gitignored)
│   ├── scraped/                     <- Pre-scraped site data
│   ├── processed/                   <- features.csv, consequences.csv, severity_scores.csv
│   └── outputs/                     <- model_metrics.json, predictions.csv, experiment/ablation results
├── models/                          <- Trained models (on GDrive, not in git)
├── notebooks/                       <- Exploratory analysis
└── app/                             <- Flask web application
    ├── main.py                      <- API endpoints + impunity score computation
    ├── templates/index.html         <- Single-page app
    └── static/
        ├── css/style.css
        ├── js/app.js                <- D3 network, Plotly charts, interactive UI
        └── images/people/           <- 80 person photos + placeholder
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### CLI — Full Pipeline
```bash
# Step 1: Aggregate data from local EpsteinProcessor
python3 main.py download-data --local

# Step 2: Extract severity scores
python3 main.py scrape-severity

# Step 3: Scrape consequence labels
python3 main.py scrape-consequences

# Step 4: Build feature matrix
python3 main.py build-features

# Step 5: Train all models
python3 main.py train-models --run-experiment

# Or run complete pipeline
python3 main.py run-all --local
```

### Web Application
```bash
python3 app/main.py
# Visit http://localhost:5001
```

The web app provides:
- **Connection Network**: D3.js force graph showing co-occurrence relationships (node size = impunity index)
- **People Grid**: Searchable, filterable, sortable cards for all 66 individuals
- **Person Modal**: Impunity gauge with score breakdown, NLP features, model predictions, document citations with Bates numbers
- **Models & Evaluation**: 4-tab interface with model comparison, confusion matrices, stress tests, limitations
- **Impunity Gap Analysis**: Scatter plot of evidence index vs. consequence tier

## Data Sources

- **Document Corpus**: 2,849 Epstein case files via DOJ releases (emails, depositions, legal filings, documents)
- **Consequence Labels**: Manually researched from Wikipedia and news sources, categorized into 3 tiers
- **Person Images**: Scraped from public sources, with placeholder fallback

## Research Question

> **Does power protect?** Does the correlation between documentary evidence and real-world consequences weaken for high-power individuals?

The Impunity Index quantifies this gap: individuals with high evidence indices but no consequences receive boosted scores, making the disparity visible and measurable.

## License

Educational use only — Duke AIPI Graduate ML Course

## Attribution

Scaffolded with AI assistance (Claude, Anthropic)
