# Attribution: Scaffolded with AI assistance (Claude, Anthropic)

"""
Flask web application for The Impunity Index.

This app provides an interactive dashboard for exploring the relationship
between Epstein file evidence and real-world consequences,
using NLP-derived features and ML classification.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, Any

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


def compute_impunity_scores(features_df: pd.DataFrame, consequences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute impunity index from NLP features and consequence outcomes.

    The impunity index is our own metric derived from:
    1. Evidence Index (0-10): Weighted combination of NLP-extracted features
       - Document mention frequency, co-occurrence with incriminating terms,
         context sentiment, document type diversity, subject line presence
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

    # Normalize NLP features to 0-1 using min-max scaling
    nlp_cols = {
        'mention_count': 0.30,
        'cooccurrence_score': 0.20,
        'total_mentions': 0.15,
        'doc_type_diversity': 0.15,
        'name_in_subject_line': 0.10,
    }

    for col in nlp_cols:
        col_max = merged[col].max()
        if col_max > 0:
            merged[f'{col}_norm'] = merged[col] / col_max
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

    # Load severity scores JSON (people list source)
    scores_path = base_path / "data" / "scraped" / "epsteinoverview_scores.json"
    if scores_path.exists():
        with open(scores_path, 'r') as f:
            DATA['scores'] = json.load(f)
        logger.info(f"Loaded {len(DATA['scores'])} people entries")

    # Load features
    features_path = base_path / "data" / "processed" / "features.csv"
    if features_path.exists():
        DATA['features'] = pd.read_csv(features_path)

    # Load consequences
    consequences_path = base_path / "data" / "processed" / "consequences.csv"
    if consequences_path.exists():
        DATA['consequences'] = pd.read_csv(consequences_path)

    # Compute impunity scores from NLP features + consequences
    if 'features' in DATA and 'consequences' in DATA:
        DATA['impunity'] = compute_impunity_scores(
            DATA['features'], DATA['consequences']
        )
        logger.info(f"Computed impunity scores for {len(DATA['impunity'])} people")

    # Load edges
    edges_path = base_path / "data" / "processed" / "edges.csv"
    if edges_path.exists():
        DATA['edges'] = pd.read_csv(edges_path)

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

    # Load images manifest
    images_manifest_path = base_path / "app" / "static" / "images" / "people" / "images_manifest.json"
    if images_manifest_path.exists():
        with open(images_manifest_path, 'r') as f:
            DATA['images'] = json.load(f)

    # Load summaries
    summaries_path = base_path / "data" / "processed" / "summaries.json"
    if summaries_path.exists():
        with open(summaries_path, 'r') as f:
            DATA['summaries'] = json.load(f)

    # Merge features and consequences for person detail
    if 'features' in DATA and 'consequences' in DATA:
        DATA['merged'] = DATA['features'].merge(
            DATA['consequences'][['name', 'consequence_tier', 'consequence_description']],
            on='name', how='left'
        )
        # Also merge impunity scores
        if 'impunity' in DATA:
            DATA['merged'] = DATA['merged'].merge(
                DATA['impunity'], on='name', how='left'
            )


# ── Page Routes ───────────────────────────────────────────────

@app.route('/')
def index() -> str:
    return render_template('index.html')


# ── API: People & Network ────────────────────────────────────

@app.route('/api/people')
def get_all_people() -> Any:
    """Get all people with impunity scores for grid and network."""
    non_person_topics = {
        "dentist", "gynecologist", "pregnant", "whoops",
        "beef jerky", "pizza", "cream cheese",
        "drugs", "bitcoin", "9/11", "zorro ranch",
        "baal and occult references", "israel and mossad",
        "dangene and jennie enterprise", "epstein suicide",
        "qatar", "lifetouch",
    }

    people = []

    if 'scores' in DATA:
        for entry in DATA['scores']:
            name = entry['name']
            if name.lower().strip() in non_person_topics:
                continue

            # Get impunity index (our derived metric)
            imp_score = 0.0
            evidence_idx = 0.0
            if 'impunity' in DATA:
                a_row = DATA['impunity'][DATA['impunity']['name'] == name]
                if not a_row.empty:
                    imp_score = float(a_row.iloc[0]['impunity_index'])
                    evidence_idx = float(a_row.iloc[0]['evidence_index'])

            level = get_impunity_level(imp_score)

            # Get consequence info
            consequence_tier = 0
            consequence_desc = ''
            if 'consequences' in DATA:
                c_row = DATA['consequences'][DATA['consequences']['name'] == name]
                if not c_row.empty:
                    consequence_tier = int(c_row.iloc[0]['consequence_tier'])
                    consequence_desc = str(c_row.iloc[0].get('consequence_description', ''))

            # Get mention count
            mention_count = 0
            if 'features' in DATA:
                f_row = DATA['features'][DATA['features']['name'] == name]
                if not f_row.empty:
                    mention_count = int(f_row.iloc[0].get('mention_count', 0))

            # Image URL
            image_file = DATA.get('images', {}).get(name, 'placeholder.png')
            image_url = f"/static/images/people/{image_file}"

            people.append({
                'name': name,
                'impunity_index': imp_score,
                'evidence_index': evidence_idx,
                'level': level,
                'consequence_tier': consequence_tier,
                'consequence_description': consequence_desc,
                'badge': get_tier_badge(consequence_tier),
                'mention_count': mention_count,
                'image_url': image_url
            })

    return jsonify(people)


@app.route('/api/edges')
def get_edges() -> Any:
    if 'edges' not in DATA or DATA['edges'].empty:
        return jsonify([])
    return jsonify(DATA['edges'].to_dict('records'))


@app.route('/api/person/<name>')
def get_person(name: str) -> Any:
    """Get person profile with impunity index and NLP features."""
    if 'merged' not in DATA:
        return jsonify({'error': 'Data not loaded'}), 500

    person_data = DATA['merged'][DATA['merged']['name'] == name]
    if person_data.empty:
        return jsonify({'error': 'Person not found'}), 404

    person = person_data.iloc[0].to_dict()

    tier = person.get('consequence_tier', 0)
    tier = int(tier) if pd.notna(tier) else 0
    badge = get_tier_badge(tier)

    imp_score = float(person.get('impunity_index', 0))
    evidence_idx = float(person.get('evidence_index', 0))

    # Get predictions
    predictions = {}
    if 'predictions' in DATA:
        pred_data = DATA['predictions'][DATA['predictions']['name'] == name]
        if not pred_data.empty:
            pred_row = pred_data.iloc[0]
            for col in pred_row.index:
                if '_pred' in col:
                    predictions[col.replace('_pred', '')] = (
                        int(pred_row[col]) if pd.notna(pred_row[col]) else None
                    )

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

    image_file = DATA.get('images', {}).get(name, 'placeholder.png')
    image_url = f"/static/images/people/{image_file}"

    return jsonify({
        'name': person['name'],
        'impunity_index': imp_score,
        'evidence_index': evidence_idx,
        'level': get_impunity_level(imp_score),
        'consequence_tier': tier,
        'consequence_description': person.get('consequence_description', 'No information available'),
        'badge': badge,
        'predictions': predictions,
        'connections': connections,
        'image_url': image_url,
        'has_summary': name in DATA.get('summaries', {}),
        'features': {
            'mention_count': int(person.get('mention_count', 0)),
            'total_mentions': int(person.get('total_mentions', 0)),
            'mean_sentiment': float(person.get('mean_context_sentiment', 0)),
            'cooccurrence_score': int(person.get('cooccurrence_score', 0)),
            'doc_type_diversity': int(person.get('doc_type_diversity', 0)),
            'in_subject_line': bool(person.get('name_in_subject_line', False))
        }
    })


@app.route('/api/person/<name>/summary')
def get_person_summary(name: str) -> Any:
    if 'summaries' not in DATA:
        return jsonify({'error': 'Summaries not available'}), 404
    summary = DATA['summaries'].get(name)
    if not summary:
        return jsonify({'error': f'Summary not found for {name}'}), 404
    return jsonify(summary)


@app.route('/api/search')
def search_person() -> Any:
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
    """Get data for impunity gap scatter plot."""
    if 'merged' not in DATA:
        return jsonify([])

    df = DATA['merged'].dropna(subset=['consequence_tier']).copy()

    if 'impunity_index' in df.columns:
        score_col = 'impunity_index'
    elif 'accountability_score' in df.columns:
        score_col = 'accountability_score'
    else:
        score_col = 'severity_score'

    bins = [0, 2.5, 5.0, 7.5, 10.01]
    df['power_tier'] = pd.cut(
        df[score_col], bins=bins,
        labels=['Low', 'Moderate', 'High', 'Critical'],
        include_lowest=True
    )

    chart_data = []
    for _, row in df.iterrows():
        chart_data.append({
            'name': row['name'],
            'impunity_index': float(row.get('impunity_index', 0)),
            'evidence_index': float(row.get('evidence_index', 0)),
            'consequence_tier': int(row['consequence_tier']),
            'power_tier': str(row['power_tier']) if pd.notna(row['power_tier']) else 'Unknown',
            'description': row.get('consequence_description', '')
        })

    return jsonify(chart_data)


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
    if 'ablation_results' not in DATA:
        return jsonify({'error': 'Ablation results not available'}), 404
    return jsonify(DATA['ablation_results'].to_dict('records'))


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
    app.run(host='0.0.0.0', port=5001, debug=True)


if __name__ == "__main__":
    main()
