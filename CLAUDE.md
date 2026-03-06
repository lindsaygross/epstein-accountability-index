# Epstein Accountability Index — CLAUDE.md

Project assistant context for Claude Code. Read this at the start of every session.

---

## What This Project Is

**The Impunity Index** — an NLP-driven accountability dashboard over the Epstein case files.
The core idea: use NLP to extract evidence strength per individual from the documents,
compare that to real-world consequences they faced, and compute an **impunity score** =
the gap between what the documents show and what justice delivered.

This is an NLP course project (MODULE PROJECT RUBRIC requirements must be met).

**Live app target:** Flask backend at `app/main.py`, port 5001.
**Deployed URL:** TODO — needs public deployment (Render / Railway / HuggingFace Spaces).

---

## Two Sibling Projects — Understand Both

### 1. `epstein-accountability-index/` (THIS REPO)
- **Purpose:** The graded course project. Impunity Index dashboard, 4 ML models, rubric compliance.
- **Stack:** Flask, pandas, sklearn, sentence-transformers, spaCy, VADER, TF-IDF
- **Data:** 66 people (hand-labeled) + 1,264 person registry from kaggle CSV
- **Models trained:** MajorityClassifier + LR + RF+TF-IDF + ST+SVC (all saved as .pkl)
- **Model artifacts:** `models/` has 4 .pkl files as of 2026-03-04

### 2. `../epstein-paper-trail/` (SIBLING REPO)
- **Purpose:** A more complete RAG + NER system with ChromaDB vector store and Claude API backend
- **Stack:** FastAPI, ChromaDB, sentence-transformers (all-MiniLM-L6-v2), Anthropic Claude API
- **Data:** 2,492 docs in `raw_corpus.jsonl` (961 DOJ PDFs + 1,264 kaggle_persons + Wikipedia + court docs)
- **Key asset:** `chroma_db/` — **4,376 chunks already embedded with all-MiniLM-L6-v2** (ChromaDB, cosine similarity)
- **Key asset:** `data/raw/doj_datasets/ds10/` (962 EFTA docs, 101MB) + `ds11/` (962 docs, 58MB)
- **Key asset:** `data/raw/epstein-persons-2026-02-13_cleaned.csv` — **1,264 people with nationality field**
- **Models:** `models/baseline/logreg_model.pkl` trained
- **RAG:** `app/backend/rag.py` — full RAG pipeline using Claude as backbone, already works

---

## Data Reality (Current State)

### What We Have
| Source | Location | Size | Content |
|--------|----------|------|---------|
| DOJ PDFs (ds10) | `../epstein-paper-trail/data/raw/doj_datasets/ds10/` | 101MB | 962 EFTA docs (.pdf + .txt) |
| DOJ PDFs (ds11) | `../epstein-paper-trail/data/raw/doj_datasets/ds11/` | 58MB | 962 EFTA docs |
| Raw corpus JSONL | `../epstein-paper-trail/data/raw/raw_corpus.jsonl` | 2,492 docs | Kaggle persons, DOJ, Wikipedia, court |
| ChromaDB | `../epstein-paper-trail/chroma_db/` | 69MB | 4,376 embedded chunks |
| Person corpora | `data/processed/person_corpora.json` | 1.8MB | 49 people with text (capped 50K chars each) |
| JMail API | `https://data.jmail.world/v1/` | ~700MB total | Full email + document corpus, free, no auth |

### What's Missing / Incomplete
- **No raw JSON files** in `data/raw/` — the `ds*_agg.json` files were never committed (too large for git)
- **20 of 66 people have zero mentions** in the current subset corpus
- **30 of 79 people have <50 chars** in their person corpus (no usable text)
- **JMail full dataset not downloaded** — `emails.parquet` (334MB) + 5 document volumes not cached locally
  - Use the sibling repo's `raw_corpus.jsonl` (2,492 docs) as full corpus — it's already available
- **jmail_cache/** directory not needed — corpus already in paper-trail sibling

### The Corpus Gap Problem
The current `person_corpora.json` is built from a subset of documents. Many high-profile people
(Bill Maher, Ellen DeGeneres, Clarence Thomas, etc.) show zero mentions because those documents
weren't in the subset. jmail.world has the full set.

---

## Architecture & How Things Connect

```
jmail.world API (full corpus, 700MB+)
    ↓ scripts/download_jmail.py
data/jmail_cache/*.parquet
    ↓ scripts/data_loader.py
data/processed/person_corpora.json  (text per person)
data/processed/doc_level_dataset.csv (text windows per person-document pair)
    ↓ scripts/build_features.py
data/processed/features.csv  (7 NLP features per person)
    ↓ scripts/model.py
models/*.pkl  (trained models)
    ↓ app/main.py
http://localhost:5001  (Flask dashboard)
```

The **paper-trail** project has a parallel architecture using ChromaDB + Claude RAG.
The goal is to **merge these**: use paper-trail's ChromaDB/RAG for citations and summaries,
use this project's ML models for the impunity score.

---

## Model Design Philosophy (CONFIRMED — CORRECT)

**We are building the semantic space from scratch, not using pre-trained classifiers.**

What this means concretely:
- We use `all-MiniLM-L6-v2` purely as an **encoder** (feature extractor), not a classifier
- The classification head (SVM/LR) is trained fresh on our labeled data (66 people)
- The embedding model itself is pre-trained (unavoidable — you need language understanding)
  but we do NOT fine-tune it; we treat it as a fixed feature extractor
- This is analogous to using ImageNet weights for CV features then training a custom head
- Legal-BERT failed because we tried to FINE-TUNE a 110M param model on 50 training examples
- The sentence-transformers approach avoids this by NOT fine-tuning

**The distinction matters for the paper:** we claim novelty in applying semantic embeddings
to build a person-level evidence space from legal documents, then training accountability
prediction on top. That semantic space IS our contribution.

---

## Impunity Score Formula (Current)

```python
evidence_index = (
    0.30 * mention_count_norm +
    0.20 * cooccurrence_score_norm +
    0.15 * total_mentions_norm +
    0.15 * doc_type_diversity_norm +
    0.10 * name_in_subject_line_norm +
    0.10 * sentiment_norm
) * 10

# Consequence modifier
tier_0 (no consequence):   impunity = evidence_index * 1.3  (capped 10)
tier_1 (soft consequence): impunity = evidence_index * 1.0
tier_2 (hard consequence): impunity = evidence_index * 0.7
```

**Problem with current formula:** weights are hand-tuned, not learned. The sentence-transformer
approach should replace/augment the evidence_index with a model-learned similarity score.

**Better formula (to implement):**
```
impunity = learned_evidence_score * (1 - consequence_weight[tier])
consequence_weight = {0: 0.0, 1: 0.4, 2: 0.9}
```
Where `learned_evidence_score` comes from the SVM probability output.

---

## Current Model Results

| Model | Accuracy | F1 Macro | Notes |
|-------|----------|----------|-------|
| MajorityClassifier (naive) | 78.6% | 0.44 | Rubric baseline — always predicts "no consequence" |
| Logistic Regression | 71.4% | 0.65 | **Best** — 7 tabular NLP features, balanced class weight |
| Random Forest + TF-IDF | 78.6% | 0.635 | Tabular + 500 TF-IDF bigrams |
| SentenceTransformer+SVC | 78.6% | 0.44 | all-MiniLM-L6-v2 (384-dim) + tabular (391-dim) → LinearSVC |
| Legal-BERT (deprecated) | 45.7% | 0.418 | FAILED — fine-tuning 110M params on 50 samples |

**All 4 model artifacts saved to `models/`:**
- `models/majority_classifier.pkl`
- `models/logistic_regression.pkl`
- `models/random_forest_tfidf.pkl`
- `models/stsvc.pkl`

**Why ST+SVC matches majority classifier:** Only 3 positives in test set (14 total). With such
a small test set, any single misclassification drastically swings F1. The model IS correctly
implemented; it just can't outperform the majority baseline on 3 positive test samples.
Training on more data (1,264 people once fully labeled) will fix this.

**Why Legal-BERT failed:** Fine-tuning 110M parameters on 50 training examples. Collapsed to
predicting majority class. The failure IS a finding — documented in model_metrics.json.

**Class imbalance:** 51 no-consequence vs 15 with-consequence (3.4:1 ratio).

---

## Completed Work (2026-03-04)

### DONE — Data
- [x] `scripts/expand_persons.py` — merges 1,264 kaggle persons with consequences + severity scores
- [x] `data/processed/people_registry.csv` — full 1,264 person registry with sector, country, jurisdiction
- [x] `data/processed/consequences_enriched.csv` — 1,264 rows with geographic metadata

### DONE — Models
- [x] `MajorityClassifier` naive baseline added to `scripts/model.py`
- [x] `SentenceTransformer + LinearSVC` deep learning model added to `scripts/model.py`
- [x] All 4 model artifacts saved to `models/`

### DONE — API Endpoints
- [x] `POST /api/predict` — live inference from all trained models + consensus
- [x] `GET /api/person/<name>/citations` — ChromaDB citation lookup from sibling repo
- [x] `GET /api/registry` — full 1,264 person registry with country/sector/tier filters
- [x] `load_models()` in `app/main.py` — loads all 4 model artifacts correctly

---

## Remaining Work

### Priority 1 — Data Completeness
- [ ] Re-run `python scripts/build_features.py` against full corpus (sibling raw_corpus.jsonl)
  to fix the 20 people with zero mentions
- [ ] Pool doj_datasets (ds10 + ds11) for local feature extraction if needed

### Priority 2 — App Improvements
- [ ] Add `GET /api/person/<name>/similar` — top-5 similar people by embedding distance
- [ ] Add world map visualization to app (Plotly choropleth)
- [ ] One-hot encode `sector` as model feature (after re-training on 1,264 labeled people)

### Priority 3 — Rubric Compliance Gaps
- [ ] **Error analysis** — identify 5 specific mispredictions with root cause explanation
- [ ] **Experiment write-up** — the ablation study (already coded) needs written interpretation
- [ ] **Ethics statement** — required in paper
- [ ] **Commercial viability statement** — required in paper
- [ ] **Related work** — cite prior Epstein NLP work, accountability index literature

### Priority 6 — Deployment
- [ ] **Recommendation: Render.com free tier** — supports Flask, persistent disk, free SSL
  - Alternative: Railway.app (better free tier for Python)
  - Alternative: HuggingFace Spaces (if we switch to Gradio/Streamlit — but rubric discourages basic Streamlit)
- [ ] Model artifacts must be committed to git LFS or stored on HuggingFace Hub
  - `models/logistic_regression.pkl` and `models/random_forest_tfidf.pkl` are small enough for git
  - SVM on sentence embeddings (~10MB) fits in git LFS
  - DO NOT commit Legal-BERT (400MB+) — load from HuggingFace Hub at inference time if needed
- [ ] ChromaDB from paper-trail can be deployed as a separate service or embedded
- [ ] Environment variables needed: `ANTHROPIC_API_KEY` (for RAG summaries)

---

## File Map

```
epstein-accountability-index/
├── main.py                          ← CLI entry point for full pipeline
├── app/main.py                      ← Flask server (PORT 5001)
│   ├── compute_impunity_scores()    ← Derives impunity index from NLP features
│   ├── GET /api/people              ← All 66 people with impunity scores
│   ├── GET /api/person/<name>       ← Individual profile + connections
│   ├── GET /api/person/<name>/summary ← Template-based summary with citations
│   ├── GET /api/edges               ← Co-occurrence network edges
│   ├── GET /api/chart-data          ← Scatter plot data (evidence vs consequence)
│   ├── GET /api/model-results       ← Model metrics JSON
│   ├── GET /api/search              ← Name search
│   ├── POST /api/predict            ← Live inference (all 4 models + consensus)
│   ├── GET /api/person/<name>/citations ← ChromaDB citation lookup (paper-trail)
│   └── GET /api/registry            ← 1264 person registry (country/sector/tier filters)
├── scripts/
│   ├── model.py                     ← ModelTrainer class (4 models + ablation + experiment)
│   ├── expand_persons.py            ← Builds people_registry.csv from 1264-person kaggle CSV
│   ├── build_features.py            ← FeatureExtractor (spaCy NER, VADER, TF-IDF)
│   ├── data_loader.py               ← Builds person_corpora.json + doc_level_dataset.csv
│   ├── download_jmail.py            ← Downloads from data.jmail.world/v1/ API
│   ├── generate_summaries.py        ← Template summaries with EFTA citations
│   ├── scrape_severity.py           ← Scrapes epsteinoverview.com scores
│   ├── scrape_consequences.py       ← Scrapes Wikipedia/news consequence data
│   └── build_edges.py               ← Co-occurrence network edges
├── data/
│   ├── processed/
│   │   ├── features.csv             ← 66 people × 7 NLP features
│   │   │   ├── consequences.csv         ← 66 people × tier + description + source_url
│   │   ├── people_registry.csv      ← 1264 people with sector, country, jurisdiction (NEW)
│   │   ├── consequences_enriched.csv ← 1264 rows with geographic metadata (NEW)
│   │   ├── person_corpora.json      ← 79 people → concatenated text (49 non-empty)
│   │   ├── edges.csv                ← 334 co-occurrence edges
│   │   ├── summaries.json           ← 79 template summaries with citations
│   │   └── severity_scores.csv      ← Raw severity from epsteinoverview.com
│   ├── outputs/
│   │   ├── model_metrics.json       ← Training results (accuracy, F1, confusion matrix)
│   │   ├── predictions.csv          ← Per-person model predictions
│   │   ├── experiment_results.csv   ← Power tier experiment
│   │   └── ablation_results.csv     ← Feature ablation study
│   ├── raw/                         ← ds*_agg.json files (NOT in git, too large)
│   └── scraped/
│       └── epsteinoverview_scores.json  ← 83 entries with severity scores
└── models/                          ← Trained model artifacts (saved with sklearn 1.7.2)
    ├── majority_classifier.pkl      ← DummyClassifier (most_frequent)
    ├── logistic_regression.pkl      ← LR + StandardScaler
    ├── random_forest_tfidf.pkl      ← RF + TF-IDF + StandardScaler
    └── stsvc.pkl                    ← SentenceTransformer+SVC (encoder_name, scaler, name_to_text)

../epstein-paper-trail/             ← SIBLING REPO — key assets
├── data/raw/
│   ├── raw_corpus.jsonl            ← 2,492 docs (kaggle + DOJ + Wikipedia + court)
│   ├── doj_datasets/ds10/          ← 962 EFTA PDFs + TXTs (101MB)
│   ├── doj_datasets/ds11/          ← 962 EFTA PDFs + TXTs (58MB)
│   └── epstein-persons-2026-02-13_cleaned.csv  ← 1,264 people WITH NATIONALITY
├── chroma_db/                      ← 4,376 chunks embedded with all-MiniLM-L6-v2
└── app/backend/rag.py              ← RAG pipeline (ChromaDB + Claude API)
```

---

## How to Run

```bash
# Start Flask server (now uses ~/.venv — sklearn + sentence-transformers installed)
/Users/shreyamendi/.venv/bin/python3 app/main.py
# → http://localhost:5001

# Train/retrain all models (use ~/.venv — has sklearn + sentence-transformers)
/Users/shreyamendi/.venv/bin/python3 scripts/model.py --run-experiment

# Expand people registry (run once when kaggle CSV changes)
/Users/shreyamendi/.venv/bin/python3 scripts/expand_persons.py

# Test POST /api/predict
curl -s -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"name": "Ghislaine Maxwell"}' | python3 -m json.tool

# Test /api/registry with filters
curl "http://localhost:5001/api/registry?sector=finance&limit=10"

# Test /api/citations (requires chromadb + paper-trail sibling)
curl "http://localhost:5001/api/person/Ghislaine%20Maxwell/citations"
```

**Python environment notes (UPDATED):**
- `~/.venv/` — now has flask, pandas, sklearn 1.7.2, sentence-transformers, torch, spaCy
- `../epstein-paper-trail/.venv/` — has sklearn 1.8.0, chromadb, FastAPI (use for paper-trail)
- System python3 (3.13) — do NOT use, no packages
- Models are saved with sklearn 1.7.2 (flask venv) — reload with same venv to avoid warnings

---

## Key Decisions Made

1. **Geographic data:** Pull from `../epstein-paper-trail/data/raw/epstein-persons-2026-02-13_cleaned.csv`
   which has nationality for 1,264 people. Map to consequences.csv manually for our 66.

2. **Model architecture:** sentence-transformers (all-MiniLM-L6-v2) as fixed encoder → LinearSVC.
   This is the "deep learning" model for the rubric. Legal-BERT remains as a documented failure
   with root cause analysis. Do NOT try to fix Legal-BERT — the failure is the finding.

3. **Citations/document linking:** Use paper-trail's ChromaDB + EFTA IDs. When app shows a person
   profile, query ChromaDB for top chunks mentioning them → return EFTA IDs → link to
   `https://jmail.world/drive` (search by EFTA ID). This gives real document citations.

4. **Impunity score:** Keep the current formula for now but make evidence_index a blend of
   (a) current hand-tuned NLP features and (b) SVM probability score from sentence embeddings.
   The SVM probability IS the learned component.

5. **Deployment:** Use Render.com (free tier). Store model .pkl files in git or HuggingFace Hub.
   No GPU needed — LinearSVC inference is CPU-only and fast. Embeddings pre-computed at startup.

6. **Training compute:** Laptop is not enough for Legal-BERT fine-tuning.
   For sentence-transformers + LinearSVC: runs fine on MacBook MPS (no GPU needed, it's sklearn).
   The embedding step (encoding ~50K chars per person × 66 people) takes ~2 min on CPU.
   Use Google Colab free tier for anything heavier.

7. **Data pooling:** Run `scripts/download_jmail.py` to get full jmail corpus. Also symlink or
   copy `../epstein-paper-trail/data/raw/doj_datasets/` into `data/raw/` for the full EFTA set.
   The `data_loader.py` already handles jmail_cache parquet files.

---

## What NOT to Do

- Do NOT try to re-fine-tune Legal-BERT on this dataset — it will fail the same way
- Do NOT commit large data files (>10MB) to git — use .gitignore
- Do NOT use the system python3 (3.13) — it has no packages
- Do NOT regenerate `summaries.json` or `person_corpora.json` unless raw data is available
- Do NOT change the impunity formula weights without re-running the ablation study
- Do NOT deploy with `debug=True` in Flask — switch to gunicorn for production
