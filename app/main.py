# Attribution: Scaffolded with AI assistance (Claude, Anthropic)

"""
Flask web application for The Accountability Gap.

This app provides an interactive dashboard for exploring the relationship
between Epstein file mention severity and real-world consequences.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for loaded data and models
DATA = {}
MODELS = {}


def load_data() -> None:
    """Load all required data files."""
    logger.info("Loading data files...")

    base_path = Path(__file__).parent.parent

    # Load severity scores JSON (primary source of truth for people list)
    scores_path = base_path / "data" / "scraped" / "epsteinoverview_scores.json"
    if scores_path.exists():
        with open(scores_path, 'r') as f:
            DATA['scores'] = json.load(f)
        logger.info(f"Loaded {len(DATA['scores'])} severity scores")

    # Load features
    features_path = base_path / "data" / "processed" / "features.csv"
    if features_path.exists():
        DATA['features'] = pd.read_csv(features_path)

    # Load consequences
    consequences_path = base_path / "data" / "processed" / "consequences.csv"
    if consequences_path.exists():
        DATA['consequences'] = pd.read_csv(consequences_path)

    # Load edges (co-occurrence data for network graph)
    edges_path = base_path / "data" / "processed" / "edges.csv"
    if edges_path.exists():
        DATA['edges'] = pd.read_csv(edges_path)
        logger.info(f"Loaded {len(DATA['edges'])} edges")
    else:
        logger.warning("Edges file not found - network graph will have no connections")

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

    # Merge features and consequences
    if 'features' in DATA and 'consequences' in DATA:
        DATA['merged'] = DATA['features'].merge(
            DATA['consequences'][['name', 'consequence_tier', 'consequence_description']],
            on='name',
            how='left'
        )
        logger.info(f"Loaded {len(DATA['merged'])} merged records")


def load_models() -> None:
    """Load trained models."""
    logger.info("Loading models...")

    base_path = Path(__file__).parent.parent
    models_path = base_path / "models"

    # Load Gradient Boosting model
    gb_path = models_path / "gradient_boosting_model.pkl"
    if gb_path.exists():
        MODELS['gradient_boosting'] = joblib.load(gb_path)
        logger.info("Loaded Gradient Boosting model")
    else:
        logger.warning("Gradient Boosting model not found")


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


def get_tier_badge(tier: int) -> Dict[str, str]:
    """Get badge color and label for consequence tier."""
    badges = {
        0: {'color': 'none', 'label': 'No Consequence'},
        1: {'color': 'soft', 'label': 'Soft Consequence'},
        2: {'color': 'hard', 'label': 'Hard Consequence'}
    }
    return badges.get(tier, {'color': 'none', 'label': 'Unknown'})


# ── Page Routes ───────────────────────────────────────────────

@app.route('/')
def index() -> str:
    """Render main dashboard."""
    return render_template('index.html')


# ── API: People & Network ────────────────────────────────────

@app.route('/api/people')
def get_all_people() -> Any:
    """
    Get all 66 people with summary data for the grid and network graph.

    Returns:
        JSON array of person objects with severity, consequence, and features
    """
    # Non-person topics to filter out
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

            score = float(entry.get('severity_score', 0))
            level = get_severity_level(score)

            # Get consequence info if available
            consequence_tier = 0
            consequence_desc = ''
            if 'consequences' in DATA:
                c_row = DATA['consequences'][DATA['consequences']['name'] == name]
                if not c_row.empty:
                    consequence_tier = int(c_row.iloc[0]['consequence_tier'])
                    consequence_desc = str(c_row.iloc[0].get('consequence_description', ''))

            # Get feature data if available
            mention_count = 0
            if 'features' in DATA:
                f_row = DATA['features'][DATA['features']['name'] == name]
                if not f_row.empty:
                    mention_count = int(f_row.iloc[0].get('mention_count', 0))

            people.append({
                'name': name,
                'severity_score': score,
                'level': level,
                'consequence_tier': consequence_tier,
                'consequence_description': consequence_desc,
                'badge': get_tier_badge(consequence_tier),
                'mention_count': mention_count
            })

    return jsonify(people)


@app.route('/api/edges')
def get_edges() -> Any:
    """
    Get co-occurrence edges for the network graph.

    Returns:
        JSON array of edge objects with source, target, weight
    """
    if 'edges' not in DATA or DATA['edges'].empty:
        return jsonify([])

    edges = DATA['edges'].to_dict('records')
    return jsonify(edges)


@app.route('/api/person/<name>')
def get_person(name: str) -> Any:
    """Get person profile data."""
    if 'merged' not in DATA:
        return jsonify({'error': 'Data not loaded'}), 500

    person_data = DATA['merged'][DATA['merged']['name'] == name]

    if person_data.empty:
        return jsonify({'error': 'Person not found'}), 404

    person = person_data.iloc[0].to_dict()

    tier = person.get('consequence_tier', 0)
    tier = int(tier) if pd.notna(tier) else 0
    badge = get_tier_badge(tier)
    score = float(person.get('severity_score', 0))

    # Get predictions if available
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

    # Get connected people from edges
    connections = []
    if 'edges' in DATA and not DATA['edges'].empty:
        edges_df = DATA['edges']
        connected = edges_df[
            (edges_df['source'] == name) | (edges_df['target'] == name)
        ].copy()
        connected = connected.sort_values('weight', ascending=False).head(10)
        for _, row in connected.iterrows():
            other = row['target'] if row['source'] == name else row['source']
            connections.append({
                'name': other,
                'weight': int(row['weight'])
            })

    response = {
        'name': person['name'],
        'severity_score': score,
        'level': get_severity_level(score),
        'consequence_tier': tier,
        'consequence_description': person.get(
            'consequence_description', 'No information available'
        ),
        'badge': badge,
        'predictions': predictions,
        'connections': connections,
        'features': {
            'mention_count': int(person.get('mention_count', 0)),
            'total_mentions': int(person.get('total_mentions', 0)),
            'mean_sentiment': float(person.get('mean_context_sentiment', 0)),
            'cooccurrence_score': int(person.get('cooccurrence_score', 0)),
            'doc_type_diversity': int(person.get('doc_type_diversity', 0)),
            'in_subject_line': bool(person.get('name_in_subject_line', False))
        }
    }

    return jsonify(response)


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
    """Get data for accountability gap scatter plot."""
    if 'merged' not in DATA:
        return jsonify([])

    df = DATA['merged'].dropna(subset=['severity_score', 'consequence_tier']).copy()

    bins = [0, 2, 5, 8, 10.01]
    df['power_tier'] = pd.cut(
        df['severity_score'],
        bins=bins,
        labels=['Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )

    chart_data = []
    for _, row in df.iterrows():
        chart_data.append({
            'name': row['name'],
            'severity_score': float(row['severity_score']),
            'consequence_tier': int(row['consequence_tier']),
            'power_tier': str(row['power_tier']) if pd.notna(row['power_tier']) else 'Unknown',
            'description': row.get('consequence_description', '')
        })

    return jsonify(chart_data)


@app.route('/api/model-results')
def get_model_results() -> Any:
    """Get model performance metrics."""
    base_path = Path(__file__).parent.parent
    metrics_path = base_path / "data" / "outputs" / "model_metrics.json"

    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            results = json.load(f)
        return jsonify(results)

    return jsonify({'error': 'Model metrics not available'}), 404


@app.route('/api/experiment-results')
def get_experiment_results() -> Any:
    """Get power tier experiment results."""
    if 'experiment_results' not in DATA:
        return jsonify({'error': 'Experiment results not available'}), 404

    results = DATA['experiment_results'].to_dict('records')
    return jsonify(results)


@app.route('/api/ablation-results')
def get_ablation_results() -> Any:
    """Get feature ablation study results."""
    if 'ablation_results' not in DATA:
        return jsonify({'error': 'Ablation results not available'}), 404

    results = DATA['ablation_results'].to_dict('records')
    return jsonify(results)


# ── Error Handlers ────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error) -> tuple:
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error) -> tuple:
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


def main() -> None:
    """Main entry point for the app."""
    logger.info("Starting Flask application...")

    load_data()
    load_models()

    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )


if __name__ == "__main__":
    main()
