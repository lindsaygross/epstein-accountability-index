# Project: The Impunity Index
# Authors: Lindsay Gross, Shreya Mendi, Andrew Jin
# Advisor: Brinnae Bent, PhD
# Claude chat: https://claude.ai/chat/f8744002-3279-48ab-9d9a-8efa1fdb1af1
# Built with Claude AI assistance

"""
Flask web application for The Impunity Index.

This app provides an interactive dashboard for exploring the relationship
between Epstein file evidence and real-world consequences,
using NLP-derived features and ML classification.
"""

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

DATA = {}
MODELS = {}




def compute_impunity_scores(features_df: pd.DataFrame, consequences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute impunity index from NLP features and consequence outcomes.

    The impunity index is our own metric derived from:
    1. Evidence Index (0-10): Weighted combination of NLP-extracted features
       using log-scaled, percentile-capped normalization (P90 = 1.0).
       This prevents extreme outliers from compressing everyone else.
    2. Consequence modifier: Adjusts based on whether justice has been served
       - No consequence: +30% (high impunity — gap exists)
       - Soft consequence: neutral
       - Hard consequence (convicted): -30% (low impunity — justice served)
    """
    merged = features_df.merge(
        consequences_df[['name', 'consequence_tier']],
        on='name', how='left'
    )
    merged['consequence_tier'] = merged['consequence_tier'].fillna(0).astype(int)

    # Log-scaled percentile-capped normalization
    # P90 of nonzero values = 1.0, preventing outliers from squishing everyone
    nlp_cols = {
        'mention_count': 0.30,
        'cooccurrence_score': 0.25,
        'total_mentions': 0.15,
        'doc_type_diversity': 0.10,
        'name_in_subject_line': 0.10,
    }

    for col in nlp_cols:
        logged = np.log1p(merged[col].astype(float))
        nonzero_vals = logged[logged > 0].sort_values()
        if len(nonzero_vals) > 0:
            cap_idx = min(len(nonzero_vals) - 1, int(len(nonzero_vals) * 0.9))
            cap_val = nonzero_vals.iloc[cap_idx]
            if cap_val > 0:
                merged[f'{col}_norm'] = (logged / cap_val).clip(upper=1.0)
            else:
                merged[f'{col}_norm'] = 0.0
        else:
            merged[f'{col}_norm'] = 0.0

    # Invert sentiment: more negative context = more concerning
    sent_col = 'mean_context_sentiment'
    s_max = merged[sent_col].max()
    s_min = merged[sent_col].min()
    if s_max != s_min:
        merged['sentiment_norm'] = (s_max - merged[sent_col]) / (s_max - s_min)
    else:
        merged['sentiment_norm'] = 0.0

    # Evidence index: weighted combination of normalized features
    merged['evidence_index'] = (
        nlp_cols['mention_count'] * merged['mention_count_norm'] +
        nlp_cols['cooccurrence_score'] * merged['cooccurrence_score_norm'] +
        nlp_cols['total_mentions'] * merged['total_mentions_norm'] +
        nlp_cols['doc_type_diversity'] * merged['doc_type_diversity_norm'] +
        nlp_cols['name_in_subject_line'] * merged['name_in_subject_line_norm'] +
        0.10 * merged['sentiment_norm']
    ) * 10

    # Apply consequence modifier
    def _score(row):
        ev = row['evidence_index']
        tier = row['consequence_tier']
        if tier == 0:
            return min(10.0, ev * 1.3)
        elif tier == 1:
            return ev * 1.0
        else:
            return ev * 0.7

    merged['impunity_index'] = merged.apply(_score, axis=1).round(1)
    merged['evidence_index'] = merged['evidence_index'].round(1)

    return merged[['name', 'impunity_index', 'evidence_index']]


def get_impunity_level(score: float) -> str:
    """Map impunity index to a named level."""
    if score >= 7.5:
        return 'Critical'
    elif score >= 5.0:
        return 'High'
    elif score >= 2.5:
        return 'Moderate'
    elif score >= 1.0:
        return 'Low'
    else:
        return 'Minimal'


def get_tier_badge(tier: int) -> Dict[str, str]:
    """Get badge color and label for consequence tier."""
    badges = {
        0: {'color': 'none', 'label': 'No Consequence'},
        1: {'color': 'soft', 'label': 'Soft Consequence'},
        2: {'color': 'hard', 'label': 'Hard Consequence'}
    }
    return badges.get(tier, {'color': 'none', 'label': 'Unknown'})


def load_data() -> None:
    """Load all required data files and compute derived metrics."""
    logger.info("Loading data files...")
    base_path = Path(__file__).parent.parent

    # Load full 1264-person registry (from expand_persons.py output)
    registry_path = base_path / "data" / "processed" / "people_registry.csv"
    if registry_path.exists():
        DATA['registry'] = pd.read_csv(registry_path)
        logger.info(f"Loaded people registry: {len(DATA['registry'])} people")

    # Load evidence scores derived from corpus (doj_pdf, court, wiki, doj_press)
    # These are computed purely from document evidence — no external severity scores
    evidence_scores_path = base_path / "data" / "processed" / "evidence_scores.json"
    if evidence_scores_path.exists():
        with open(evidence_scores_path, 'r') as f:
            DATA['evidence_scores'] = json.load(f)
        logger.info(f"Loaded evidence scores for {len(DATA['evidence_scores'])} people")

    # Load legacy scores JSON (83 entries from epsteinoverview) — kept only for
    # consequence descriptions, NOT used for scoring
    scores_path = base_path / "data" / "scraped" / "epsteinoverview_scores.json"
    if scores_path.exists():
        with open(scores_path, 'r') as f:
            DATA['scores'] = json.load(f)
        logger.info(f"Loaded {len(DATA['scores'])} legacy score entries (consequence descriptions only)")

    # Load features
    features_path = base_path / "data" / "processed" / "features.csv"
    if features_path.exists():
        DATA['features'] = pd.read_csv(features_path)

    # Load consequences
    consequences_path = base_path / "data" / "processed" / "consequences.csv"
    if consequences_path.exists():
        DATA['consequences'] = pd.read_csv(consequences_path)

    # Compute impunity scores — prefer evidence_scores.json (corpus-derived, no external sources)
    # Fall back to NLP features + consequences computation for the 66-person subset
    if 'evidence_scores' in DATA and 'consequences' in DATA and 'registry' in DATA:
        # Build impunity from evidence_scores for all registry people
        impunity_rows = []
        ev_scores = DATA['evidence_scores']
        for _, reg_row in DATA['registry'].iterrows():
            name = str(reg_row['name'])
            ev = ev_scores.get(name, {})
            evidence_idx = float(ev.get('evidence_index', 0.0))
            # Apply consequence modifier
            tier = int(reg_row.get('consequence_tier', 0) or 0)
            if tier == 0 and evidence_idx > 0:
                imp = min(10.0, evidence_idx * 1.3)
            elif tier == 2:
                imp = evidence_idx * 0.7
            else:
                imp = evidence_idx
            impunity_rows.append({'name': name, 'impunity_index': round(imp, 1), 'evidence_index': round(evidence_idx, 1)})
        DATA['impunity'] = pd.DataFrame(impunity_rows)
        logger.info(f"Computed evidence-based impunity scores for {len(DATA['impunity'])} people")
    elif 'features' in DATA and 'consequences' in DATA:
        DATA['impunity'] = compute_impunity_scores(
            DATA['features'], DATA['consequences']
        )
        logger.info(f"Computed NLP impunity scores for {len(DATA['impunity'])} people (fallback)")

    # Load edges (co-occurrence data for network graph)
    edges_path = base_path / "data" / "processed" / "edges.csv"
    if edges_path.exists():
        DATA['edges'] = pd.read_csv(edges_path)
        logger.info(f"Loaded {len(DATA['edges'])} edges")

    # Load predictions
    predictions_path = base_path / "data" / "outputs" / "predictions.csv"
    if predictions_path.exists():
        DATA['predictions'] = pd.read_csv(predictions_path)

    # Load experiment results
    experiment_path = base_path / "data" / "outputs" / "experiment_results.csv"
    if experiment_path.exists():
        DATA['experiment_results'] = pd.read_csv(experiment_path)

    # Load ablation results
    ablation_path = base_path / "data" / "outputs" / "ablation_results.csv"
    if ablation_path.exists():
        DATA['ablation_results'] = pd.read_csv(ablation_path)

    # Load image URL mapping (Wikipedia thumbnails, keyed by person name)
    image_urls_path = base_path / "data" / "processed" / "person_image_urls.json"
    if image_urls_path.exists():
        with open(image_urls_path, 'r') as f:
            DATA['image_urls'] = json.load(f)
        logger.info(f"Loaded image URLs for {len(DATA['image_urls'])} people")
    else:
        DATA['image_urls'] = {}

    # Load summaries
    summaries_path = base_path / "data" / "processed" / "summaries.json"
    if summaries_path.exists():
        with open(summaries_path, 'r') as f:
            DATA['summaries'] = json.load(f)

    # Load consequence source links (manually curated verification URLs)
    consequence_sources_path = base_path / "data" / "processed" / "consequence_sources.json"
    if consequence_sources_path.exists():
        with open(consequence_sources_path, 'r') as f:
            DATA['consequence_sources'] = json.load(f)
        logger.info(f"Loaded consequence sources for {len(DATA['consequence_sources'])} people")

    # Load document summaries (extractive, from pipeline_enhancements.py)
    doc_summaries_path = base_path / "data" / "processed" / "document_summaries.json"
    if doc_summaries_path.exists():
        with open(doc_summaries_path, 'r') as f:
            DATA['document_summaries'] = json.load(f)
        logger.info(f"Loaded {len(DATA['document_summaries'])} document summaries")

    # Load per-person topic distributions
    topic_dist_path = base_path / "data" / "processed" / "person_topic_distributions.json"
    if topic_dist_path.exists():
        with open(topic_dist_path, 'r') as f:
            DATA['person_topics'] = json.load(f)
        logger.info(f"Loaded topic distributions for {len(DATA['person_topics'])} people")

    # Merge features and consequences for person detail
    if 'features' in DATA and 'consequences' in DATA:
        DATA['merged'] = DATA['features'].merge(
            DATA['consequences'][['name', 'consequence_tier', 'consequence_description']],
            on='name', how='left'
        )
        logger.info(f"Loaded {len(DATA['merged'])} merged records")


def load_models() -> None:
    """Load trained ML model artifacts for live inference."""
    logger.info("Loading models...")
    base_path = Path(__file__).parent.parent
    models_path = base_path / "models"

    model_files = {
        "majority_classifier": "majority_classifier.pkl",
        # Prefer v2 models trained on full corpus; fall back to v1
        "logistic_regression": "logistic_v2.pkl",
        "random_forest_tfidf": "random_forest_tfidf.pkl",
        "sentence_transformer_svc": "stsvc_v2.pkl",
    }

    fallbacks = {
        "logistic_regression": "logistic_regression.pkl",
        "sentence_transformer_svc": "stsvc.pkl",
    }
    for name, filename in model_files.items():
        path = models_path / filename
        if path.exists():
            MODELS[name] = joblib.load(path)
            logger.info(f"Loaded {name} from {filename}")
        elif name in fallbacks:
            fb_path = models_path / fallbacks[name]
            if fb_path.exists():
                MODELS[name] = joblib.load(fb_path)
                logger.info(f"Loaded {name} from fallback {fallbacks[name]}")
        else:
            logger.warning(f"Model not found: {filename}")


def get_severity_level(score: float) -> str:
    """Map severity score to a named level."""
    if score >= 8.5:
        return 'Critical'
    elif score >= 7.0:
        return 'High'
    elif score >= 4.0:
        return 'Medium'
    elif score >= 1.0:
        return 'Low'
    else:
        return 'Minimal'


def clean_str(val: Any, default: str = '') -> str:
    """Return clean string, replacing NaN/None/empty with default."""
    if val is None:
        return default
    s = str(val).strip()
    if s in ('nan', 'None', 'NaN', 'null', ''):
        return default
    return s

# ── Page Routes ───────────────────────────────────────────────

@app.route('/')
def index() -> str:
    """Render main dashboard."""
    return render_template('index.html')


@app.route('/about')
def about() -> str:
    """Render about/disclaimer/ethics page."""
    return render_template('about.html')


# ── API: People & Network ────────────────────────────────────

@app.route('/api/people')
def get_all_people() -> Any:
    """Get all people with impunity scores for grid and network.

    Impunity score is derived purely from document evidence:
      - Keyword co-occurrence with incriminating terms in DOJ PDFs, court docs, Wikipedia
      - Mention frequency in evidence documents
      - Flights on Epstein aircraft (public flight logs)
      - Connection count and black book presence
    No external severity scores from third-party sites are used.
    """
    # Build fast lookup indexes once
    impunity_lookup = {}
    if 'impunity' in DATA:
        for _, r in DATA['impunity'].iterrows():
            impunity_lookup[r['name']] = {
                'impunity_index': float(r['impunity_index']),
                'evidence_index': float(r['evidence_index']),
            }

    feature_lookup = {}
    if 'features' in DATA:
        for _, r in DATA['features'].iterrows():
            feature_lookup[r['name']] = int(r.get('mention_count', 0))

    # Evidence scores from corpus (replaces external severity_score)
    evidence_scores = DATA.get('evidence_scores', {})

    people = []

    if 'registry' in DATA:
        for _, row in DATA['registry'].iterrows():
            name = str(row['name'])

            # Priority 1: NLP-computed impunity index from features.csv (66 people)
            imp_data = impunity_lookup.get(name, {})
            imp_score = imp_data.get('impunity_index', 0.0)
            evidence_idx = imp_data.get('evidence_index', 0.0)

            # Priority 2: Evidence-based score from corpus (all 1264 people)
            if imp_score == 0.0 and name in evidence_scores:
                ev = evidence_scores[name]
                imp_score = round(float(ev.get('evidence_index', 0.0)), 1)
                evidence_idx = imp_score
                # Apply consequence modifier: no consequence = higher impunity
                consequence_tier = int(row.get('consequence_tier', 0) or 0)
                if consequence_tier == 0 and imp_score > 0:
                    imp_score = round(min(10.0, imp_score * 1.3), 1)
                elif consequence_tier == 2:
                    imp_score = round(imp_score * 0.7, 1)

            consequence_tier = int(row.get('consequence_tier', 0) or 0)
            consequence_desc = clean_str(row.get('consequence_description', ''), '')

            image_url = DATA.get('image_urls', {}).get(name, '/static/images/people/placeholder.png')

            ev_data = evidence_scores.get(name, {})
            people.append({
                'name': name,
                'impunity_index': imp_score,
                'evidence_index': evidence_idx,
                'level': get_impunity_level(imp_score),
                'consequence_tier': consequence_tier,
                'consequence_description': consequence_desc,
                'badge': get_tier_badge(consequence_tier),
                'mention_count': feature_lookup.get(name, int(ev_data.get('doc_mentions', 0))),
                'sector': str(row.get('sector', '') or ''),
                'country': str(row.get('country', 'Unknown') or 'Unknown'),
                'nationality': str(row.get('nationality', '') or ''),
                'flights': int(row.get('flights', 0) or 0),
                'in_black_book': bool(row.get('in_black_book', False)),
                'keyword_cooccurrence': int(ev_data.get('keyword_cooccurrence', 0)),
                'doc_mentions': int(ev_data.get('doc_mentions', 0)),
                'image_url': image_url,
            })

    return jsonify(people)


@app.route('/api/edges')
def get_edges() -> Any:
    if 'edges' not in DATA or DATA['edges'].empty:
        return jsonify([])
    return jsonify(DATA['edges'].to_dict('records'))


def _build_score_reasoning(name: str, ev_data: dict, imp_score: float, evidence_idx: float, tier: int, flights: int, in_black_book: bool) -> str:
    """Build a human-readable explanation of the impunity score."""
    parts = []
    jmail = int(ev_data.get('jmail_doc_count', 0))
    mentions = int(ev_data.get('doc_mentions', 0))
    cooc = int(ev_data.get('keyword_cooccurrence', 0))
    conns = int(ev_data.get('connections', 0))

    if jmail > 0:
        parts.append(f"appears in {jmail:,} Epstein email/EFTA documents")
    if mentions > 0:
        parts.append(f"mentioned {mentions} times across the DOJ corpus")
    if cooc > 0:
        parts.append(f"co-occurs with incriminating keywords {cooc} times")
    if flights > 0:
        parts.append(f"logged {flights} Epstein flight legs")
    if conns > 0:
        parts.append(f"connected to {conns} other individuals")
    if in_black_book:
        parts.append("listed in Epstein's black book")

    if not parts:
        return "Limited direct corpus evidence found."

    reasoning = f"Evidence index {evidence_idx:.1f}/10 based on: {'; '.join(parts)}."
    if tier == 0 and evidence_idx > 0:
        reasoning += f" No legal consequences documented → impunity modifier ×1.3 → impunity index {imp_score:.1f}."
    elif tier == 2:
        reasoning += f" Hard legal consequence on record → modifier ×0.7 → impunity index {imp_score:.1f}."
    elif tier == 1:
        reasoning += f" Soft consequence on record → no modifier → impunity index {imp_score:.1f}."
    return reasoning


@app.route('/api/person/<name>')
def get_person(name: str) -> Any:
    """Get person profile with impunity index and evidence-based features.

    Falls back to registry + evidence_scores for people not in features.csv.
    """
    # Try NLP-features merged data first (66 people)
    person = None
    if 'merged' in DATA:
        person_data = DATA['merged'][DATA['merged']['name'] == name]
        if not person_data.empty:
            person = person_data.iloc[0].to_dict()

    # Fall back to registry for all 1264 people
    reg_row = None
    if 'registry' in DATA:
        reg_data = DATA['registry'][DATA['registry']['name'] == name]
        if not reg_data.empty:
            reg_row = reg_data.iloc[0]

    if person is None and reg_row is None:
        return jsonify({'error': 'Person not found'}), 404

    # Consequence tier and description
    tier = 0
    consequence_desc = ''
    source_url = ''
    if person is not None:
        tier = person.get('consequence_tier', 0)
        tier = int(tier) if pd.notna(tier) else 0
        consequence_desc = clean_str(person.get('consequence_description', ''), '')
    elif reg_row is not None:
        tier = int(reg_row.get('consequence_tier', 0) or 0)
        consequence_desc = clean_str(reg_row.get('consequence_description', ''), '')

    if 'consequences' in DATA:
        c_row = DATA['consequences'][DATA['consequences']['name'] == name]
        if not c_row.empty:
            source_url = str(c_row.iloc[0].get('source_url', '') or '')
            if not consequence_desc:
                consequence_desc = clean_str(c_row.iloc[0].get('consequence_description', ''), '')

    # Fall back to bio from registry if no consequence description
    if not consequence_desc and reg_row is not None:
        consequence_desc = clean_str(reg_row.get('bio', ''), '')

    badge = get_tier_badge(tier)

    # Get predictions
    predictions = {}
    if 'predictions' in DATA:
        pred_data = DATA['predictions'][DATA['predictions']['name'] == name]
        if not pred_data.empty:
            pred_row = pred_data.iloc[0]
            # Extract model predictions with probabilities
            model_keys = ['logistic_regression', 'sentence_transformer_svc', 'random_forest_tfidf', 'majority_classifier']
            for model_key in model_keys:
                pred_col = f'{model_key}_pred'
                prob_col = f'{model_key}_prob'
                if pred_col in pred_row.index and pd.notna(pred_row.get(pred_col)):
                    predictions[model_key] = {
                        'label': int(pred_row[pred_col]),
                        'probability': round(float(pred_row[prob_col]), 4) if prob_col in pred_row.index and pd.notna(pred_row.get(prob_col)) else None,
                    }
            # Consensus
            if 'consensus_prob' in pred_row.index and pd.notna(pred_row.get('consensus_prob')):
                predictions['consensus'] = {
                    'label': int(pred_row.get('consensus_label', 0)),
                    'probability': round(float(pred_row['consensus_prob']), 4),
                }

    # Connected people from edges
    connections = []
    if 'edges' in DATA and not DATA['edges'].empty:
        edges_df = DATA['edges']
        connected = edges_df[
            (edges_df['source'] == name) | (edges_df['target'] == name)
        ].copy()
        connected = connected.sort_values('weight', ascending=False).head(10)
        for _, row in connected.iterrows():
            other = row['target'] if row['source'] == name else row['source']
            connections.append({'name': other, 'weight': int(row['weight'])})

    image_url = DATA.get('image_urls', {}).get(name, '/static/images/people/placeholder.png')

    # Impunity + evidence index — NLP features take priority
    imp_score = 0.0
    evidence_idx = 0.0
    if 'impunity' in DATA:
        imp_row = DATA['impunity'][DATA['impunity']['name'] == name]
        if not imp_row.empty:
            imp_score = float(imp_row.iloc[0]['impunity_index'])
            evidence_idx = float(imp_row.iloc[0]['evidence_index'])

    # Fall back to corpus evidence scores for registry-only people
    ev_data = DATA.get('evidence_scores', {}).get(name, {})
    if imp_score == 0.0 and ev_data:
        evidence_idx = round(float(ev_data.get('evidence_index', 0.0)), 1)
        imp_score = evidence_idx
        if tier == 0 and imp_score > 0:
            imp_score = round(min(10.0, imp_score * 1.3), 1)
        elif tier == 2:
            imp_score = round(imp_score * 0.7, 1)

    # Features — from NLP pipeline if available, else from evidence scores
    if person is not None:
        features = {
            'mention_count': int(person.get('mention_count', 0)),
            'total_mentions': int(person.get('total_mentions', 0)),
            'mean_sentiment': float(person.get('mean_context_sentiment', 0)),
            'cooccurrence_score': int(person.get('cooccurrence_score', 0)),
            'doc_type_diversity': int(person.get('doc_type_diversity', 0)),
            'in_subject_line': bool(person.get('name_in_subject_line', False)),
        }
    else:
        features = {
            'mention_count': int(ev_data.get('doc_mentions', 0)),
            'total_mentions': int(ev_data.get('doc_mentions', 0)),
            'mean_sentiment': 0.0,
            'cooccurrence_score': int(ev_data.get('keyword_cooccurrence', 0)),
            'doc_type_diversity': 0,
            'in_subject_line': False,
        }

    # Registry metadata
    sector = ''
    country = 'Unknown'
    nationality = ''
    flights = 0
    in_black_book = False
    if reg_row is not None:
        sector = str(reg_row.get('sector', '') or '')
        country = str(reg_row.get('country', 'Unknown') or 'Unknown')
        nationality = str(reg_row.get('nationality', '') or '')
        flights = int(reg_row.get('flights', 0) or 0)
        in_black_book = bool(reg_row.get('in_black_book', False))

    return jsonify({
        'name': name,
        'impunity_index': imp_score,
        'evidence_index': evidence_idx,
        'level': get_impunity_level(imp_score),
        'consequence_tier': tier,
        'consequence_description': consequence_desc,
        'source_url': source_url,
        'badge': badge,
        'predictions': predictions,
        'connections': connections,
        'image_url': image_url,
        'has_summary': name in DATA.get('summaries', {}),
        'sector': sector,
        'country': country,
        'nationality': nationality,
        'flights': flights,
        'in_black_book': in_black_book,
        'keyword_cooccurrence': int(ev_data.get('keyword_cooccurrence', 0)),
        'doc_mentions': int(ev_data.get('doc_mentions', 0)),
        'jmail_doc_count': int(ev_data.get('jmail_doc_count', 0)),
        'features': features,
        'score_reasoning': _build_score_reasoning(name, ev_data, imp_score, evidence_idx, tier, flights, in_black_book),
        'topic_distribution': DATA.get('person_topics', {}).get(name, {}),
    })


@app.route('/api/person/<name>/summary')
def get_person_summary(name: str) -> Any:
    """Get structured summary with document citations for a person."""
    if 'summaries' not in DATA:
        return jsonify({'error': 'Summaries not available'}), 404
    summary = DATA['summaries'].get(name)
    if not summary:
        return jsonify({'error': f'Summary not found for {name}'}), 404
    return jsonify(summary)


@app.route('/api/search')
def search_person() -> Any:
    """Search for people by name."""
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    if 'merged' in DATA:
        matches = DATA['merged'][
            DATA['merged']['name'].str.lower().str.contains(query, na=False)
        ]['name'].tolist()
        return jsonify(matches[:10])
    return jsonify([])

# ── API: Analysis & Charts ────────────────────────────────────

@app.route('/api/chart-data')
def get_chart_data() -> Any:
    """Get data for impunity gap scatter plot — uses all 1264 registry people."""
    if 'registry' not in DATA:
        return jsonify([])

    reg = DATA['registry']
    ev_scores = DATA.get('evidence_scores', {})
    imp_df = DATA.get('impunity', pd.DataFrame())
    imp_lookup = {}
    if not imp_df.empty:
        for _, r in imp_df.iterrows():
            imp_lookup[r['name']] = {
                'impunity_index': float(r['impunity_index']),
                'evidence_index': float(r['evidence_index']),
            }

    chart_data = []
    for _, row in reg.iterrows():
        name = str(row['name'])
        tier = int(row.get('consequence_tier', 0) or 0)
        imp = imp_lookup.get(name, {})
        impunity_idx = imp.get('impunity_index', 0.0)
        evidence_idx = imp.get('evidence_index', 0.0)
        if impunity_idx == 0.0:
            evidence_idx = float(ev_scores.get(name, {}).get('evidence_index', 0.0))
            impunity_idx = min(10.0, evidence_idx * 1.3) if tier == 0 and evidence_idx > 0 else evidence_idx

        # Bin by impunity index
        if impunity_idx >= 7.5:
            power_tier = 'Critical'
        elif impunity_idx >= 5.0:
            power_tier = 'High'
        elif impunity_idx >= 2.5:
            power_tier = 'Moderate'
        else:
            power_tier = 'Low'

        chart_data.append({
            'name': name,
            'impunity_index': round(impunity_idx, 1),
            'evidence_index': round(evidence_idx, 1),
            'consequence_tier': tier,
            'power_tier': power_tier,
            'description': clean_str(row.get('consequence_description', ''), '')
        })

    return jsonify(chart_data)


@app.route('/api/geo-data')
def get_geo_data() -> Any:
    """Get per-country counts and impunity data for geographic map."""
    if 'registry' not in DATA:
        return jsonify([])
    reg = DATA['registry']
    ev = DATA.get('evidence_scores', {})
    imp_df = DATA.get('impunity', pd.DataFrame())

    country_map = {}
    for _, row in reg.iterrows():
        country = clean_str(row.get('country', ''), 'Unknown')
        if not country or country == 'Unknown':
            continue
        name = str(row['name'])
        imp_score = 0.0
        if not imp_df.empty:
            imp_row = imp_df[imp_df['name'] == name]
            if not imp_row.empty:
                imp_score = float(imp_row.iloc[0]['impunity_index'])
        if imp_score == 0.0:
            imp_score = float(ev.get(name, {}).get('evidence_index', 0.0))
        tier = int(row.get('consequence_tier', 0) or 0)
        if country not in country_map:
            country_map[country] = {'country': country, 'count': 0, 'avg_impunity': 0.0, 'no_consequence': 0, 'names': []}
        country_map[country]['count'] += 1
        country_map[country]['avg_impunity'] += imp_score
        if tier == 0:
            country_map[country]['no_consequence'] += 1
        if imp_score > 0:
            country_map[country]['names'].append({'name': name, 'impunity': round(imp_score, 1), 'tier': tier})

    result = []
    for c, d in country_map.items():
        if d['count'] > 0:
            d['avg_impunity'] = round(d['avg_impunity'] / d['count'], 2)
            d['names'].sort(key=lambda x: x['impunity'], reverse=True)
            d['names'] = d['names'][:5]  # top 5 per country
        result.append(d)
    result.sort(key=lambda x: x['count'], reverse=True)
    return jsonify(result)


@app.route('/api/semantic-space')
def get_semantic_space() -> Any:
    """Return t-SNE 2D projection of person embeddings for visualization."""
    import numpy as np
    base_path = Path(__file__).parent.parent
    emb_path = base_path / "data" / "processed" / "person_embeddings.npy"
    names_path = base_path / "data" / "processed" / "person_embedding_names.json"

    if not emb_path.exists():
        return jsonify({'error': 'Embeddings not built yet'}), 404

    embeddings = np.load(str(emb_path))
    with open(names_path) as f:
        names = json.load(f)

    # Filter to people with non-zero embeddings
    nonzero_mask = embeddings.sum(axis=1) != 0
    emb_filtered = embeddings[nonzero_mask]
    names_filtered = [n for n, m in zip(names, nonzero_mask) if m]

    # t-SNE (cache result to avoid recomputing on every request)
    tsne_cache = base_path / "data" / "processed" / "tsne_coords.json"
    if tsne_cache.exists():
        with open(tsne_cache) as f:
            cached = json.load(f)
        if len(cached.get('names', [])) == len(names_filtered):
            pass  # use cache below
        else:
            cached = None
    else:
        cached = None

    if cached is None:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=min(30, len(emb_filtered) // 4),
                    random_state=42, max_iter=1000, learning_rate='auto', init='pca')
        coords = tsne.fit_transform(emb_filtered).tolist()
        cached = {'names': names_filtered, 'coords': coords}
        with open(tsne_cache, 'w') as f:
            json.dump(cached, f)

    # Attach metadata
    ev = DATA.get('evidence_scores', {})
    imp_df = DATA.get('impunity', pd.DataFrame())
    reg = DATA.get('registry', pd.DataFrame())

    points = []
    for name, (x, y) in zip(cached['names'], cached['coords']):
        imp_score = 0.0
        if not imp_df.empty:
            imp_row = imp_df[imp_df['name'] == name]
            if not imp_row.empty:
                imp_score = float(imp_row.iloc[0]['impunity_index'])
        if imp_score == 0.0:
            imp_score = float(ev.get(name, {}).get('evidence_index', 0.0))
        tier = 0
        if not reg.empty:
            reg_row = reg[reg['name'] == name]
            if not reg_row.empty:
                tier = int(reg_row.iloc[0].get('consequence_tier', 0) or 0)
        points.append({
            'name': name, 'x': round(x, 3), 'y': round(y, 3),
            'impunity': round(imp_score, 1), 'tier': tier,
            'level': get_impunity_level(imp_score),
        })

    return jsonify(points)


@app.route('/api/model-results')
def get_model_results() -> Any:
    base_path = Path(__file__).parent.parent
    metrics_path = base_path / "data" / "outputs" / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Model metrics not available'}), 404


@app.route('/api/experiment-results')
def get_experiment_results() -> Any:
    if 'experiment_results' not in DATA:
        return jsonify({'error': 'Experiment results not available'}), 404
    return jsonify(DATA['experiment_results'].to_dict('records'))


@app.route('/api/ablation-results')
def get_ablation_results() -> Any:
    """Get feature ablation study results."""
    if 'ablation_results' not in DATA:
        return jsonify({'error': 'Ablation results not available'}), 404
    return jsonify(DATA['ablation_results'].to_dict('records'))


# ── API: Live Inference ───────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def predict_consequence() -> Any:
    """
    Live consequence prediction endpoint.

    Accepts a JSON body with person NLP features and returns
    probability predictions from all trained models.

    Request body (all fields optional, defaults to 0):
        {
            "name": "Person Name",        -- used to look up known features
            "mention_count": 0,
            "total_mentions": 0,
            "mean_context_sentiment": 0.0,
            "cooccurrence_score": 0,
            "doc_type_diversity": 0,
            "name_in_subject_line": 0,
            "severity_score": 0.0
        }

    Response:
        {
            "name": "...",
            "predictions": {
                "logistic_regression": {"label": 0, "probability": 0.12},
                "random_forest_tfidf": {"label": 0, "probability": 0.08},
                ...
            },
            "consensus_label": 0,
            "consensus_probability": 0.10,
            "impunity_index": 2.4
        }
    """
    data = request.get_json(force=True, silent=True) or {}

    name = data.get("name", "Unknown")

    # Build feature vector — prefer known features for existing people
    feature_cols = [
        'mention_count', 'total_mentions', 'mean_context_sentiment',
        'cooccurrence_score', 'doc_type_diversity',
        'name_in_subject_line', 'severity_score'
    ]

    features = {}
    # Start with zeros
    for col in feature_cols:
        features[col] = 0.0

    # Override with any known data from features CSV
    if 'features' in DATA:
        row = DATA['features'][DATA['features']['name'] == name]
        if not row.empty:
            for col in feature_cols:
                if col in row.columns:
                    features[col] = float(row.iloc[0].get(col, 0) or 0)

    # Let request body override
    for col in feature_cols:
        if col in data:
            features[col] = float(data[col])

    X = np.array([[features[c] for c in feature_cols]])

    predictions = {}

    # Logistic Regression
    if 'logistic_regression' in MODELS:
        artifacts = MODELS['logistic_regression']
        model = artifacts['model']
        scaler = artifacts['scaler']
        X_scaled = scaler.transform(X)
        prob = float(model.predict_proba(X_scaled)[0, 1])
        label = int(model.predict(X_scaled)[0])
        predictions['logistic_regression'] = {'label': label, 'probability': round(prob, 4)}

    # Random Forest + TF-IDF (tabular-only path when no text provided)
    if 'random_forest_tfidf' in MODELS:
        artifacts = MODELS['random_forest_tfidf']
        model = artifacts['model']
        scaler = artifacts['scaler']
        tfidf = artifacts.get('tfidf')
        X_tab = scaler.transform(X)
        if tfidf is not None:
            from scipy import sparse
            text = data.get('text', '')
            X_tfidf = tfidf.transform([text])
            X_in = sparse.hstack([sparse.csr_matrix(X_tab), X_tfidf])
        else:
            X_in = X_tab
        prob = float(model.predict_proba(X_in)[0, 1])
        label = int(model.predict(X_in)[0])
        predictions['random_forest_tfidf'] = {'label': label, 'probability': round(prob, 4)}

    # SentenceTransformer + SVC
    if 'sentence_transformer_svc' in MODELS:
        artifacts = MODELS['sentence_transformer_svc']
        model = artifacts['model']
        scaler = artifacts['scaler']
        encoder_name = artifacts.get('encoder_name', 'all-MiniLM-L6-v2')
        name_to_text = artifacts.get('name_to_text', {})
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(encoder_name)
            text = name_to_text.get(name, data.get('text', ''))
            if text:
                windows = [text[i:i+512] for i in range(0, min(len(text), 10000), 384) if text[i:i+512].strip()]
                if windows:
                    embs = encoder.encode(windows, show_progress_bar=False, convert_to_numpy=True)
                    embedding = embs.mean(axis=0)
                else:
                    embedding = np.zeros(384)
            else:
                embedding = np.zeros(384)
            X_tab_scaled = scaler.transform(X)
            X_combined = np.hstack([embedding.reshape(1, -1), X_tab_scaled])
            prob = float(model.predict_proba(X_combined)[0, 1])
            label = int(model.predict(X_combined)[0])
            predictions['sentence_transformer_svc'] = {'label': label, 'probability': round(prob, 4)}
        except Exception as e:
            logger.warning(f"ST+SVC inference failed: {e}")

    if not predictions:
        return jsonify({'error': 'No models loaded. Run scripts/model.py first.'}), 503

    # Consensus: mean probability across all models
    probs = [v['probability'] for v in predictions.values()]
    consensus_prob = round(sum(probs) / len(probs), 4)
    consensus_label = int(consensus_prob >= 0.5)

    # Quick impunity index from features
    mention_norm = min(1.0, features['mention_count'] / 50)
    cooc_norm = min(1.0, features['cooccurrence_score'] / 20)
    evidence_idx = round((mention_norm * 0.4 + cooc_norm * 0.3 + min(1.0, features['total_mentions'] / 200) * 0.15 + features['doc_type_diversity'] / 5 * 0.15) * 10, 1)

    return jsonify({
        'name': name,
        'input_features': features,
        'predictions': predictions,
        'consensus_label': consensus_label,
        'consensus_probability': consensus_prob,
        'impunity_index': evidence_idx,
        'models_used': list(predictions.keys()),
    })




@app.route('/api/person/<path:name>/citations')
def get_person_citations(name: str) -> Any:
    """
    Get document citations for a person from local ChromaDB.

    Queries the locally-built ChromaDB vector store containing chunks
    from DOJ PDFs, court docs, and other Epstein case files.

    Response:
        {
            "name": "...",
            "citations": [
                {
                    "index": 1,
                    "source": "doj_pdf",
                    "url": "...",
                    "date": "...",
                    "efta_id": "EFTA01234567",
                    "quote": "...",
                    "score": 0.87
                }, ...
            ]
        }
    """
    base_path = Path(__file__).resolve().parent.parent
    chroma_dir = base_path / "chroma_db"

    if not chroma_dir.exists():
        # Fall through to summaries-based citations below
        citations = _get_summary_citations(name)
        return jsonify({
            'name': name,
            'citations': citations,
            'total_found': len(citations),
        })

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_or_create_collection(
            "epstein_docs",
            metadata={"hnsw:space": "cosine"}
        )

        if collection.count() == 0:
            return jsonify({'name': name, 'citations': [], 'error': 'ChromaDB collection is empty'}), 404

        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        query = f"{name} Epstein connection evidence documents"
        q_emb = encoder.encode([query]).tolist()

        # First pass: query with name-specific query, require name match
        results = collection.query(
            query_embeddings=q_emb,
            n_results=min(30, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        citations = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        name_lower = name.lower()
        # Use meaningful name parts (>3 chars) for matching
        name_parts = [p for p in name_lower.split() if len(p) > 3]

        for doc, meta, dist in zip(docs, metas, distances):
            score = round(1 - dist, 4)
            doc_lower = doc.lower()
            name_found = name_parts and any(part in doc_lower for part in name_parts)

            # Build citation URL — link directly to DOJ PDF
            raw_url = meta.get("url", "")
            efta_id_raw = meta.get("efta_id", "") or meta.get("doc_id", "")
            efta_clean = str(efta_id_raw).replace(".pdf", "").strip() if efta_id_raw else ""
            dataset = meta.get("dataset", "")

            # Map dataset code to DOJ dataset number
            ds_map = {"ds8": "8", "ds9": "9", "ds10": "10", "ds11": "11", "ds12": "12"}
            ds_num = ds_map.get(dataset, "")
            if not ds_num and dataset.isdigit():
                ds_num = dataset

            if efta_clean and ds_num:
                doc_url = f"https://www.justice.gov/epstein/files/DataSet%20{ds_num}/{efta_clean}.pdf"
            elif raw_url and "justice.gov" in raw_url:
                # Already a DOJ URL — use as-is
                doc_url = raw_url.replace(".pdf.pdf", ".pdf")  # guard double extension
            elif efta_clean:
                # Fallback: try to infer dataset from EFTA ID range
                # Dataset 8 IDs: EFTA00000001–EFTA00099999 (approx)
                # Dataset 10+: EFTA01000000+ (approx)
                try:
                    efta_num = int(efta_clean.replace("EFTA", ""))
                    if efta_num < 100000:
                        doc_url = f"https://www.justice.gov/epstein/files/DataSet%208/{efta_clean}.pdf"
                    elif efta_num < 1500000:
                        doc_url = f"https://www.justice.gov/epstein/files/DataSet%2010/{efta_clean}.pdf"
                    else:
                        doc_url = f"https://www.justice.gov/epstein/files/DataSet%2011/{efta_clean}.pdf"
                except ValueError:
                    doc_url = raw_url
            else:
                doc_url = raw_url

            citation: Dict[str, Any] = {
                "source": meta.get("source", ""),
                "url": doc_url,
                "date": meta.get("date", ""),
                "score": score,
                "quote": doc[:250].replace("\n", " ").strip(),
                "name_mentioned": name_found,
            }
            if efta_clean:
                citation["efta_id"] = efta_clean
                citation["dataset"] = meta.get("dataset", "")

            citations.append(citation)

        # Prefer chunks that mention the name; fall back to top semantic matches
        named = [c for c in citations if c.get("name_mentioned")]
        candidates = sorted(named, key=lambda c: c["score"], reverse=True) if named else \
                     sorted(citations, key=lambda c: c["score"], reverse=True)

        # Deduplicate by efta_id then by URL, keeping highest-scoring per doc
        seen_efta: set = set()
        seen_url: set = set()
        seen_quote_starts: set = set()
        final = []
        for c in candidates:
            efta = c.get("efta_id", "")
            url = c.get("url", "")
            quote_start = c.get("quote", "")[:80].lower().strip()
            if efta and efta in seen_efta:
                continue
            if url and url in seen_url:
                continue
            if quote_start and quote_start in seen_quote_starts:
                continue
            if efta:
                seen_efta.add(efta)
            if url:
                seen_url.add(url)
            if quote_start:
                seen_quote_starts.add(quote_start)
            final.append(c)
            if len(final) >= 6:
                break

        for i, c in enumerate(final):
            c["index"] = i + 1
            c.pop("name_mentioned", None)

        citations = final

        return jsonify({
            'name': name,
            'citations': citations,
            'total_found': len(citations),
            'collection_size': collection.count(),
        })

    except ImportError:
        pass  # Fall through to summaries fallback
    except Exception as e:
        logger.warning(f"ChromaDB citations lookup failed for {name}: {e}")
        # Fall through to summaries fallback

    # Fallback: extract citations from summaries.json (always available)
    citations = _get_summary_citations(name)
    return jsonify({
        'name': name,
        'citations': citations,
        'total_found': len(citations),
    })



def _get_summary_citations(name: str) -> list:
    """Extract document citations from summaries.json for a person."""
    summary = DATA.get('summaries', {}).get(name, {})
    raw_citations = summary.get('citations', [])
    if not raw_citations:
        return []

    seen_efta = set()
    citations = []
    for c in raw_citations:
        doc_id = c.get('doc_id', '')
        doc_type = c.get('doc_type', 'document')
        snippet = c.get('snippet', '')
        # Extract EFTA ID from doc_id (format: ds10_agg_EFTA01298161)
        import re as _re
        m = _re.search(r'(EFTA\d+)', doc_id)
        efta_id = m.group(1) if m else ''

        if efta_id and efta_id in seen_efta:
            continue
        if efta_id:
            seen_efta.add(efta_id)

        # Build document URL — link directly to DOJ PDF
        # Infer dataset number from EFTA ID range (Dataset 8: <100k, 10: 1M-1.5M, 11: 1.5M+)
        if efta_id:
            try:
                efta_num = int(efta_id.replace("EFTA", ""))
                if efta_num < 100000:
                    url = f"https://www.justice.gov/epstein/files/DataSet%208/{efta_id}.pdf"
                elif efta_num < 1500000:
                    url = f"https://www.justice.gov/epstein/files/DataSet%2010/{efta_id}.pdf"
                else:
                    url = f"https://www.justice.gov/epstein/files/DataSet%2011/{efta_id}.pdf"
            except ValueError:
                url = ''
        else:
            url = ''

        # Determine source label
        source_label = {
            'email': 'Epstein Email/EFTA',
            'document': 'DOJ/EFTA Document',
            'legal_filing': 'Court Filing',
            'deposition': 'Deposition',
        }.get(doc_type, 'DOJ Document')

        # Get document summary and topic from pipeline data if available
        doc_data = DATA.get('document_summaries', {}).get(efta_id, {})
        doc_summary = doc_data.get('summary', '')
        doc_topic = doc_data.get('topic', source_label)

        citations.append({
            'index': len(citations) + 1,
            'source': source_label,
            'efta_id': efta_id,
            'url': url,
            'quote': snippet[:300].replace('\n', ' ').strip(),
            'doc_summary': doc_summary,
            'topic': doc_topic,
            'score': 0,
        })

        if len(citations) >= 8:
            break

    return citations


@app.route('/api/person/<path:name>/consequence-sources')
def get_consequence_sources(name: str) -> Any:
    """Get source links verifying an individual's consequence status."""
    sources = DATA.get('consequence_sources', {}).get(name, [])
    return jsonify({'name': name, 'sources': sources})


@app.route('/api/registry')
def get_registry() -> Any:
    """
    Get the full 1264-person registry with geographic and sector data.

    Query params:
        country=USA        -- filter by country
        sector=finance     -- filter by sector
        tier=0             -- filter by consequence tier (0/1/2)
        limit=100          -- max records (default 100)
        offset=0           -- pagination offset
    """
    if 'registry' not in DATA:
        return jsonify({'error': 'Registry not loaded'}), 503

    df = DATA['registry'].copy()

    country = request.args.get('country')
    sector = request.args.get('sector')
    tier = request.args.get('tier')
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))

    if country:
        df = df[df['country'].str.lower() == country.lower()]
    if sector:
        df = df[df['sector'].str.lower() == sector.lower()]
    if tier is not None:
        df = df[df['consequence_tier'] == int(tier)]

    total = len(df)
    page = df.iloc[offset:offset + limit]

    records = []
    for _, row in page.iterrows():
        records.append({
            'name': row.get('name', ''),
            'sector': row.get('sector', ''),
            'country': row.get('country', 'Unknown'),
            'jurisdiction': row.get('jurisdiction', 'unknown'),
            'nationality': row.get('nationality', ''),
            'consequence_tier': int(row.get('consequence_tier', 0)),
            'consequence_source': row.get('consequence_source', 'kaggle_only'),
            'flights': int(row.get('flights', 0)),
            'in_black_book': bool(row.get('in_black_book', False)),
            'severity_score': float(row.get('severity_score', 0)),
        })

    return jsonify({
        'total': total,
        'offset': offset,
        'limit': limit,
        'results': records,
    })


# ── Error Handlers ────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error) -> tuple:
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error) -> tuple:
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


def main() -> None:
    logger.info("Starting Flask application...")
    load_data()
    load_models()
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    app.run(host='0.0.0.0', port=port, debug=debug)


# Initialize data when module is imported (supports gunicorn and direct execution)
load_data()
load_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    app.run(host='0.0.0.0', port=port, debug=debug)
