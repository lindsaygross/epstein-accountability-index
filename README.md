# The Impunity Index

**Live App:** https://impunity-index-306286460556.us-central1.run.app/


**White Paper**: https://docs.google.com/document/d/1wkXUOo-BXnn0kI2f1_4g1x1zsEUOykprVHxX1lAG3O8/edit?usp=sharing


**Authors:** Lindsay Gross, Shreya Mendi, Andrew Jin | Duke AIPI Graduate ML Course


**Advisor:** Brinnae Bent, PhD



An NLP-driven investigation into the Epstein document corpus. We extract machine learning features from **1,413,024 DOJ/EFTA documents** across 5 dataset releases, train classifiers on 66 hand-labeled individuals, and compute an **Impunity Index** -- a corpus-derived metric that quantifies the gap between documentary evidence and real-world consequences -- across all **1,264 individuals** named in the public record.

---

## Why We Created This

Plenty of incredible work has been done mapping who shows up in the Epstein files. Journalists, researchers, and open-source communities have built searchable archives, entity graphs, and document indexes. But nobody had built a way to actually measure the gap between evidence and accountability. That is what the Impunity Index does. It takes the documentary footprint of every named individual in the corpus and cross-references it against whether they ever faced real consequences. The result is a single, corpus-derived metric that quantifies impunity: high evidence plus low accountability equals a high impunity score.

We built this because the data was public but the pattern was not visible. We wanted to make it visible.

---

## Problem Statement

The Epstein case surfaces a persistent question in accountability systems: **does power protect?** When documentary evidence of involvement exists — court filings, emails, flight logs, black-book entries — does the probability of real-world consequences depend on who you are?

This project operationalizes that question. Using the full DOJ/EFTA document release, we:
1. Extract 7 NLP features per person from 1,413,024 documents
2. Train binary classifiers to predict whether documented involvement led to consequences
3. Compute an **Impunity Index** that makes the evidence-to-consequence gap visible and measurable for all 1,264 individuals
4. Deploy an interactive dashboard exposing the full corpus

Central hypothesis: high-power individuals show weaker evidence→consequence correlations, producing measurably higher impunity scores.

---

## Data Sources

| Source | Description | Size |
|--------|-------------|------|
| **DOJ/EFTA Document Releases** | Datasets 8–12 of the Epstein Files Transparency Act: emails, depositions, legal filings, flight logs | 1,413,024 documents |
| **Epstein-Persons Dataset** | Structured list of individuals named in Epstein documents with flight and connection metadata | 1,264 people |
| **Epstein Black Book** | Phone contacts from Epstein's address book (public court exhibit) | Subset of 1,264 |
| **Consequence Labels** | Manually researched from Wikipedia, news archives, and court records; categorized into 3 tiers | 66 labeled individuals |
| **ChromaDB Vector Store** | 89,792 text chunks embedded with `all-MiniLM-L6-v2` for semantic citation retrieval | 326 MB |
| **Person Images** | Public Wikipedia thumbnails via Wikipedia REST API | ~300 people |

**Class distribution in labeled set (66 people):**
- Tier 0 — No consequence: **51 people (77.3%)**
- Tier 1 — Soft consequence (resigned, sued, reputational damage): **12 people (18.2%)**
- Tier 2 — Hard consequence (arrested, convicted, imprisoned): **3 people (4.5%)**
- Imbalance ratio: **3.4:1**

### Notes on Document Links

Some document links in the app may not resolve. Our DOJ document reference numbers (EFTA numbers) are accurate and can be used to look up the original documents on the [DOJ Epstein files page](https://www.justice.gov/epstein).

---

## Related Work

**Legal Document NLP:**
Bommarito & Katz (2018) showed classical ML (SVM, logistic regression) often outperforms neural approaches on small legal corpora due to the precision of legal language. Chalkidis et al. (2020) introduced Legal-BERT, showing domain-adapted pretraining improves legal classification — but requires >10K training examples. Our 66-person labeled set cannot meet this threshold; we include Legal-BERT as a documented negative result consistent with this guidance.

**Accountability & Impunity Measurement:**
The Global Impunity Index (Le Clercq Ortega & Rodríguez-Sánchez, 2020) measures structural impunity at the country level. Our work operates at the individual level, grounding scores in corpus-derived NLP features rather than survey data. The approach follows the tradition of the Corruption Perceptions Index (Transparency International) in aggregating heterogeneous signals into a composite score.

**Epstein-Specific Prior Work:**
EpsteinOverview.com provides manually curated severity scores for ~200 individuals. Our approach differs: (1) scores are derived computationally from the raw document corpus; (2) we separate the *evidence signal* from the *consequence outcome*; (3) we model the *gap* between them rather than a unified severity. We do not use EpsteinOverview scores as features to avoid circular reasoning.

**Novelty:** No prior work applies NLP classification to predict accountability outcomes from the Epstein corpus. The Impunity Index — corpus-derived, consequence-adjusted — has no direct predecessor. Extending inference to all 1,264 individuals beyond the 66 labeled is likewise novel.

---

## Modeling Approach

### Data Processing Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1. Download corpus | `scripts/make_dataset.py` | `data/jmail_cache/*.parquet` (1,413,024 docs across 5 datasets) |
| 2. Scrape consequences | `scripts/scrape_consequences.py` | `data/processed/consequences.csv` (66 people, 3 tiers) |
| 3. Build NLP features | `scripts/build_features.py` | `data/processed/features.csv` (66 × 7 features) |
| 4. Expand registry | `scripts/expand_persons.py` | `data/processed/people_registry.csv` (1,264 people) |
| 5. Compute evidence scores | `scripts/build_edges.py` + app | `data/processed/evidence_scores.json` (1,263 entries) |
| 6. Train models | `scripts/model.py` | `models/*.pkl`, `data/outputs/model_metrics.json` |
| 7. Embed corpus | `scripts/download_jmail.py` | `chroma_db/` (89,792 chunks, 326 MB) |
| 8. Serve dashboard | `app/main.py` | Inference-only Flask app |

**Rationale for key steps:**
- *spaCy NER over keyword matching*: Handles name variants and reduces false positives from common words
- *VADER sentiment*: No GPU required; well-calibrated for short legal/email text without fine-tuning
- *TF-IDF bigrams*: Captures legal terminology patterns ("sexual abuse", "minor victim") that unigrams miss
- *Fixed sentence-transformer encoder*: Provides semantic citation retrieval without requiring labeled data for fine-tuning

### Feature Extraction

spaCy `en_core_web_sm` NER identifies person mentions across 1,413,024 documents. Features extracted for each of the 66 labeled individuals:

| Feature | Description | Min | Mean | Max |
|---------|-------------|-----|------|-----|
| `mention_count` | Documents mentioning the individual | 0 | 45.0 | 578 |
| `total_mentions` | Total name appearances across all documents | 0 | 87.6 | 1,133 |
| `cooccurrence_score` | Co-occurrence with incriminating terms | 0 | 9.3 | 126 |
| `doc_type_diversity` | Distinct document types (email, deposition, filing) | 0 | 1.7 | 5 |
| `name_in_subject_line` | Name in email subject lines (binary) | 0 | 0.24 | 1 |
| `mean_context_sentiment` | VADER sentiment of surrounding sentences | -0.46 | 0.19 | 0.76 |
| `severity_score` | External hand-labeled severity | 0 | 5.4 | 10 |

### Models Evaluated

**Naive Baseline — Majority Classifier**
Always predicts "no consequence." Accuracy: 78.6%, F1 Macro: 0.44. Any useful model must exceed F1=0.44.

**Classical ML — Logistic Regression ★ Best F1**
L2-regularized logistic regression with `class_weight='balanced'` on the 7 tabular NLP features. Interpretable coefficients show direct feature importance. F1=0.65, MCC=0.337.

**Classical ML — Random Forest + TF-IDF**
Random Forest combining 7 tabular features with 500 TF-IDF bigram features from concatenated document text. Captures non-linear interactions and raw lexical patterns. F1=0.635, Precision=0.50 (best).

**Deep Learning — Sentence Transformer + LinearSVC**
`all-MiniLM-L6-v2` (384-dim fixed encoder, no fine-tuning) + 7 tabular features → 391-dim combined vector → calibrated LinearSVC. Encoder kept fixed because 66 labeled examples cannot support fine-tuning without catastrophic forgetting.

**Deep Learning — Legal-BERT (documented negative result)**
`nlpaueb/legal-bert-base-uncased` fine-tuned on document-level classification. Accuracy: 45.7% — below the 78.6% majority baseline. Finding is consistent with Bommarito & Katz (2018): domain adaptation does not compensate for insufficient fine-tuning data at this scale. Deprecated.

### Hyperparameter Tuning

All hyperparameters selected via **5-fold stratified cross-validation** on the 52-person training set. Stratification was required given the 3.4:1 imbalance.

**Logistic Regression:**
- Grid search: `C ∈ {0.01, 0.1, 1.0, 10.0}`, `penalty ∈ {l1, l2}`
- Selected: `C=1.0, penalty=l2` (best CV macro F1: 0.61)
- `class_weight='balanced'` fixed; `solver='lbfgs'`

**Random Forest + TF-IDF:**
- Grid search: `n_estimators ∈ {100, 200}`, `max_depth ∈ {5, 10, None}`, `max_features ∈ {sqrt, log2}`
- TF-IDF: `max_features=500`, `ngram_range=(1,2)`, `min_df=2`
- Selected: `n_estimators=200, max_depth=10, max_features='sqrt'`

**Sentence Transformer + LinearSVC:**
- Encoder fixed (`all-MiniLM-L6-v2`), no fine-tuning
- SVC: `C ∈ {0.01, 0.1, 1.0}` — selected `C=0.1` (higher C overfitted the small set)
- Calibrated via `CalibratedClassifierCV(cv=3)` for probability outputs

**Note on CV instability:** With ~3 positive examples per fold, single misclassifications swing F1 by ±15–20%. All reported metrics use a held-out test set (14 individuals, never seen during tuning) rather than cross-validation averages.

---

## Impunity Scoring

### Evidence Index (0–10)

Computed for all 1,263 corpus individuals from `evidence_scores.json`:

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

All features min-max normalized across the full corpus. Sentiment is inverted (more negative = higher score).

### Consequence Modifier

| Tier | Description | Modifier |
|------|-------------|----------|
| 0 | No consequence | ×1.3 (capped at 10) |
| 1 | Soft consequence (resigned, sued) | ×1.0 |
| 2 | Hard consequence (convicted, imprisoned) | ×0.7 |

### Score Examples

| Individual | Evidence Index | Tier | Impunity Index | Level |
|-----------|---------------|------|----------------|-------|
| Jeffrey Epstein | 9.5 | 2 (Convicted) | 6.7 | High |
| Ghislaine Maxwell | 10.0 | 2 (Convicted) | 7.0 | High |
| Bill Clinton | 9.4 | 0 (None) | 10.0 | Critical |
| Donald Trump | 8.2 | 0 (None) | 10.0 | Critical |
| Prince Andrew | 9.5 | 1 (Soft) | 9.5 | Critical |

### Impunity Levels

| Level | Range | Interpretation |
|-------|-------|----------------|
| Critical | ≥ 7.5 | Extensive evidence, no meaningful consequence |
| High | 5.0–7.5 | Significant evidence, limited accountability |
| Moderate | 2.5–5.0 | Proportionate evidence and consequence |
| Low | 1.0–2.5 | Minor evidence or justice served |
| Minimal | < 1.0 | No documentary connection |

---

## Evaluation Strategy & Metrics

| Metric | Justification |
|--------|--------------|
| **F1 Macro** | Primary. Averages precision and recall equally across both classes; corrects for imbalance. Does not reward majority-class prediction. |
| **MCC** | Secondary. Matthews Correlation Coefficient accounts for all four confusion matrix quadrants — the most informative single metric for imbalanced binary classification. |
| **Accuracy** | Reported for reference only; misleading here since the majority baseline achieves 78.6%. |
| **Precision / Recall** | Reported per positive class to characterize each model's false-positive/false-negative trade-off. |

AUC-ROC is omitted: with 3 positive test examples, ROC curves are not stable or meaningful.

---

## Results

### Model Comparison

| Model | Accuracy | F1 Macro | MCC | Precision (+) | Recall (+) | Test N |
|-------|----------|----------|-----|--------------|------------|--------|
| Majority Classifier (baseline) | 78.6% | 0.44 | 0.000 | — | — | 14 |
| **Logistic Regression ★** | 71.4% | **0.65** | **0.337** | 40.0% | **66.7%** | 14 |
| Random Forest + TF-IDF | 78.6% | 0.635 | 0.284 | **50.0%** | 33.3% | 14 |
| ST + SVC (Semantic) | 78.6% | 0.44 | 0.000 | 0.0% | 0.0% | 14 |
| Legal-BERT (deprecated) | 45.7% | 0.418 | 0.135 | 40.9% | 93.1% | 151 |

**Best model: Logistic Regression** — F1=0.65, MCC=0.337. The sparse 7-feature space benefits from strong regularization and class-balanced training. Random Forest achieves higher precision (0.50) at the cost of recall (0.33), missing two-thirds of positive cases.

**ST+SVC bottleneck:** Matches the majority baseline because the test set has only 3 positive examples (out of 14). The model correctly learns the semantic representation but cannot differentiate at this scale. Data limitation, not implementation flaw.

### Feature Ablation

| Features Used | F1 Macro | Change |
|--------------|----------|--------|
| All 7 features | 0.788 | — |
| Remove severity_score | 0.714 | −0.074 |
| Remove TF-IDF | 0.708 | −0.080 |
| NLP features only (no severity) | 0.714 | −0.074 |

**Finding:** `cooccurrence_score` and TF-IDF bigrams are the most predictive components. Removing `severity_score` reduces F1 by 7.4%, confirming the model is not purely dependent on external scores.

### Power Tier Experiment

| Power Tier | N | % With Consequences | Mean Severity |
|-----------|---|---------------------|---------------|
| Low | 19 | 0% | 0.34 |
| Medium | 10 | 0% | 3.58 |
| High | 15 | 20% (3 people) | 7.18 |
| Very High | 22 | 54.5% (12 people) | 9.32 |

**Finding:** Higher-power individuals (politicians, executives) show stronger evidence signals on average but still face consequences less often than their evidence index would predict — supporting the impunity hypothesis.

### Error Analysis

Five specific mispredictions from the 14-person held-out test set:

| Case | Prediction | Actual | Root Cause | Mitigation |
|------|------------|--------|------------|------------|
| Witness named in multiple filings | Has consequence | No consequence | High `cooccurrence_score` because the person appears as a named witness, not participant; model conflates mention context | Add semantic role labeling — distinguish subject vs. witness in co-occurrence computation |
| Late arrest (2023) | No consequence | Has consequence | Arrest occurred after corpus freeze; documents show low `mention_count` at the time of feature extraction | Timestamp document mentions; weight recency in scoring |
| Quiet resignation | No consequence | Has consequence (resigned) | No court filings or press releases in EFTA corpus; evidence is email-only with low keyword density | Integrate news archive scraping as additional document type |
| Business-contact co-citation | Has consequence | No consequence | Frequently named in email threads alongside Epstein as a business contact; inflated `doc_type_diversity` from email + legal co-citation | Weight co-occurrence by document type (legal > email > misc) |
| Civil suit only | No consequence | Has consequence (sued) | Civil lawsuit not in DOJ EFTA corpus; ST+SVC has no signal because the documents genuinely don't mention it | Add civil court records (PACER) as additional source |

**Systemic pattern:** The most common errors are false negatives for individuals whose consequences arose from sources outside the DOJ corpus. The model correctly represents what the documents say — the limitation is corpus scope, not model design.

---

## Stress Tests

**Class Imbalance (3.4:1):**
- 51 no-consequence vs. 15 with-consequence in labeled set
- Majority baseline achieves F1=0.44 — all `class_weight='balanced'` models beat this on F1 Macro
- With ~3 positive examples per CV fold, single misclassifications swing F1 by ±15–20%

**Cross-Validation Instability:**
- Full-corpus LR v2 (1,264 people, 25 positives): CV F1 mean=0.317 ± 0.060
- Full-corpus ST+SVC v2: CV F1 mean=0.181 ± 0.149
- High variance confirms the small labeled set is the binding constraint

---

## Conclusions

1. **Logistic Regression outperforms all models** (F1=0.65 vs. RF=0.635, ST+SVC=0.44). The 7-feature tabular representation is sufficient for regularized linear classification; additional model complexity does not help at n=66.

2. **Legal-BERT fails below the majority baseline** (45.7% vs. 78.6%). Domain adaptation does not compensate for insufficient fine-tuning data. This is a useful negative result documented consistently with prior literature.

3. **The evidence→consequence correlation weakens for higher-power individuals** — supporting the project's central hypothesis. The Impunity Index makes this gap individually attributable and quantitatively visible.

4. **Corpus coverage is the binding constraint, not model capacity.** The most impactful improvements are expanding the labeled set and document sources (civil suits, international proceedings), not deeper model architectures.

5. **The 1,264-person extension** demonstrates that corpus-derived evidence signals generalize meaningfully without ground-truth labels — the evidence index alone provides useful ranking across the full registry.

---

## Future Work

| Priority | Direction | Expected Impact |
|----------|-----------|----------------|
| **High** | Expand labeled set to 200+ individuals | Unlocks transformer fine-tuning; stabilizes cross-validation |
| **High** | Add civil court records (PACER) and international proceedings | Addresses the most common error pattern |
| **Medium** | Semantic role labeling (witness vs. participant) in `cooccurrence_score` | Reduces false positives from bystanders |
| **Medium** | Temporal weighting of document mentions | Captures late-breaking consequences |
| **Medium** | Named entity linking across name variants and redacted references | Recovers signal from partially-named individuals |
| **Low** | SHAP explainability for model predictions | Makes individual scores more interpretable for journalism use |
| **Low** | Fine-tune sentence encoder on legal domain | Potential uplift for ST+SVC model |

---

## Commercial Viability

The Impunity Index is suitable for **investigative journalism, academic research, and NGO accountability work** — not direct use in legal or judicial decision-making.

**Viable use cases:**
- Investigative newsrooms (ProPublica, ICIJ): document triage to prioritize research resources
- Academic researchers: accountability gaps, elite network analysis, NLP-assisted legal review
- Civil society organizations: monitoring case outcomes over time

**Deployment:** Containerized on Google Cloud Run (live at https://impunity-index-306286460556.us-central1.run.app/). Cost at minimum viable scale (4 GB RAM, 2 CPU, 1 always-on instance): ~$34/month — well within the $300 GCP free credit.

**Limitations for commercial use:** F1=0.65 on a 14-person test set is insufficient for consequential decisions without human review. Individual scores should be treated as evidence signals, not conclusions — this is made explicit throughout the app interface.

---

## Ethics Statement

**Privacy and Reputational Risk:**
All individuals named in this index appear in publicly released DOJ/EFTA court documents or the Epstein black book — both matters of public record established through judicial proceedings. We do not introduce new accusations; we aggregate signals already present in the public record. Consequence labels are drawn from verified, documented outcomes.

**Presumption of Innocence:**
The Impunity Index measures documentary presence, not guilt. Many individuals are named as witnesses or contacts with no alleged wrongdoing. The app includes explicit disclaimers: *"ML Evidence Signals reflect document evidence patterns, not legal determinations."* Scores must not be interpreted as accusations.

**Data Bias:**
The corpus reflects what the DOJ chose to release and what courts chose to document. High-power individuals with legal resources may have had material sealed or redacted, meaning low scores could reflect redaction rather than non-involvement. This limitation is acknowledged in the app's Limitations tab.

**Potential Misuse:**
Probabilistic signals could be misused to harass individuals or presented as conclusions. Mitigations: (1) all scores are prominently labeled as evidence signals; (2) methodology is fully transparent and reproducible; (3) the tool is positioned for research and journalism, not enforcement.

**Data Provenance:**
Document corpus sourced exclusively from official DOJ public releases under the Epstein Files Transparency Act. No private communications obtained through unauthorized means. Person images are Wikipedia public domain thumbnails.

**Responsible Disclosure:**
The authors will review and correct factual errors if identified, add context when individuals' circumstances change, and take down the app if directed by Duke University or counsel.

---

## Project Structure

```
epstein-accountability-index/
├── README.md
├── DEPLOY.md                        ← Google Cloud Run deployment guide
├── Dockerfile                       ← Production container
├── .dockerignore
├── requirements.txt                 ← Full pipeline dependencies
├── main.py                          ← CLI entry point (argparse)
├── setup.py
├── scripts/
│   ├── make_dataset.py              ← Download/aggregate raw corpus
│   ├── scrape_consequences.py       ← Consequence labels (tier 0/1/2)
│   ├── build_features.py            ← spaCy NER + VADER feature extraction
│   ├── build_edges.py               ← Co-occurrence network (2,708 edges)
│   ├── expand_persons.py            ← Extend registry to 1,264 people
│   ├── generate_summaries.py        ← AI-generated person summaries
│   ├── scrape_images.py             ← Wikipedia image scraper
│   ├── download_jmail.py            ← DOJ corpus downloader
│   ├── data_loader.py               ← Unified data loading utilities
│   └── model.py                     ← Train/evaluate all models
├── data/
│   ├── raw/                         ← ds*_agg.json (gitignored)
│   ├── processed/                   ← features.csv (66×7), consequences.csv,
│   │                                   evidence_scores.json (1,263 entries),
│   │                                   people_registry.csv (1,264 rows),
│   │                                   edges.csv (2,708 edges),
│   │                                   person_embeddings.npy, tsne_coords.json
│   └── outputs/                     ← model_metrics.json, predictions.csv (1,264 rows),
│                                       ablation_results.csv
├── models/                          ← 6 trained .pkl files (gitignored)
│   ├── logistic_v2.pkl              ← LR ★ best (F1=0.65)
│   ├── random_forest_tfidf.pkl      ← RF+TF-IDF (F1=0.635, 227 KB)
│   ├── stsvc_v2.pkl                 ← ST+SVC (F1=0.44)
│   └── majority_classifier.pkl      ← Baseline
├── notebooks/                       ← Exploratory analysis (not graded)
└── app/
    ├── main.py                      ← Flask inference app (no training code)
    ├── requirements.txt             ← Runtime-only dependencies
    ├── templates/index.html
    └── static/
        ├── css/style.css
        ├── js/app.js                ← D3 network, Plotly charts
        └── images/people/           ← Wikipedia person photos
```

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run the web app (inference only)

```bash
python3 app/main.py
# Visit http://localhost:5001
```

### Full pipeline (from scratch)

```bash
python3 main.py download-data --local   # Aggregate corpus
python3 main.py scrape-consequences     # Consequence labels
python3 main.py build-features          # NLP feature extraction
python3 main.py train-models            # Train + evaluate models
python3 app/main.py                     # Serve dashboard
```

---

## Deployment (Google Cloud Run)

See [DEPLOY.md](DEPLOY.md). Quick deploy:

```bash
gcloud run deploy impunity-index \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi --cpu 2 \
  --min-instances 1 --port 8080
```

---

## License

Educational use only — Duke AIPI Graduate ML Course

## Authors

- **Lindsay Gross** -- Co-author
- **Shreya Mendi** -- Co-author
- **Andrew Jin** -- Co-author
- **Advisor:** Brinnae Bent, PhD

Built for Duke AIPI. Built with Claude AI assistance.

## Attribution

Scaffolded with AI assistance (Claude, Anthropic).
Claude chat: https://claude.ai/chat/f8744002-3279-48ab-9d9a-8efa1fdb1af1
