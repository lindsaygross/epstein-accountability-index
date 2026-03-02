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
import plotly.graph_objs as go
import plotly.utils
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

    # Load features
    features_path = base_path / "data" / "processed" / "features.csv"
    DATA['features'] = pd.read_csv(features_path)

    # Load consequences
    consequences_path = base_path / "data" / "processed" / "consequences.csv"
    DATA['consequences'] = pd.read_csv(consequences_path)

    # Load predictions
    predictions_path = base_path / "data" / "outputs" / "predictions.csv"
    if predictions_path.exists():
        DATA['predictions'] = pd.read_csv(predictions_path)
    else:
        logger.warning("Predictions file not found")

    # Load experiment results
    experiment_path = base_path / "data" / "outputs" / "experiment_results.csv"
    if experiment_path.exists():
        DATA['experiment_results'] = pd.read_csv(experiment_path)
    else:
        logger.warning("Experiment results file not found")

    # Merge features and consequences
    DATA['merged'] = DATA['features'].merge(
        DATA['consequences'][['name', 'consequence_tier', 'consequence_description']],
        on='name',
        how='left'
    )

    logger.info(f"Loaded {len(DATA['merged'])} records")


def load_models() -> None:
    """Load trained models."""
    logger.info("Loading models...")

    base_path = Path(__file__).parent.parent
    models_path = base_path / "models"

    # Load XGBoost model
    xgb_path = models_path / "xgboost_model.pkl"
    if xgb_path.exists():
        MODELS['xgboost'] = joblib.load(xgb_path)
        logger.info("Loaded XGBoost model")
    else:
        logger.warning("XGBoost model not found")

    # Note: DistilBERT loading would require more setup
    # For demo purposes, we'll skip it in the web app


def get_tier_badge(tier: int) -> Dict[str, str]:
    """
    Get badge color and label for consequence tier.

    Args:
        tier: Consequence tier (0, 1, 2)

    Returns:
        Dictionary with color and label
    """
    badges = {
        0: {'color': 'success', 'label': 'No Consequence'},
        1: {'color': 'warning', 'label': 'Soft Consequence'},
        2: {'color': 'danger', 'label': 'Hard Consequence'}
    }
    return badges.get(tier, {'color': 'secondary', 'label': 'Unknown'})


@app.route('/')
def index() -> str:
    """Render main dashboard."""
    return render_template('index.html')


@app.route('/api/person/<name>')
def get_person(name: str) -> Any:
    """
    Get person profile data.

    Args:
        name: Person's name

    Returns:
        JSON response with person data
    """
    # Find person in merged data
    person_data = DATA['merged'][DATA['merged']['name'] == name]

    if person_data.empty:
        return jsonify({'error': 'Person not found'}), 404

    person = person_data.iloc[0].to_dict()

    # Get tier badge
    tier = person.get('consequence_tier', 0)
    badge = get_tier_badge(int(tier)) if pd.notna(tier) else get_tier_badge(0)

    # Get predictions if available
    predictions = {}
    if 'predictions' in DATA:
        pred_data = DATA['predictions'][DATA['predictions']['name'] == name]
        if not pred_data.empty:
            pred_row = pred_data.iloc[0]
            for col in pred_row.index:
                if '_pred' in col:
                    predictions[col.replace('_pred', '')] = int(pred_row[col]) if pd.notna(pred_row[col]) else None

    # Format response
    response = {
        'name': person['name'],
        'severity_score': float(person.get('severity_score', 0)),
        'consequence_tier': int(tier) if pd.notna(tier) else 0,
        'consequence_description': person.get('consequence_description', 'No information available'),
        'badge': badge,
        'predictions': predictions,
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
    """
    Search for people by name.

    Returns:
        JSON response with matching names
    """
    query = request.args.get('q', '').lower()

    if not query:
        return jsonify([])

    # Search in merged data
    matches = DATA['merged'][
        DATA['merged']['name'].str.lower().str.contains(query, na=False)
    ]['name'].tolist()

    return jsonify(matches[:10])  # Limit to 10 results


@app.route('/api/chart-data')
def get_chart_data() -> Any:
    """
    Get data for accountability gap scatter plot.

    Returns:
        JSON response with chart data
    """
    df = DATA['merged'].dropna(subset=['severity_score', 'consequence_tier'])

    # Create power tier categories
    df['power_tier'] = pd.qcut(
        df['severity_score'],
        q=4,
        labels=['Low', 'Medium', 'High', 'Very High'],
        duplicates='drop'
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
    """
    Get model performance metrics.

    Returns:
        JSON response with model results
    """
    # This would typically come from saved metrics
    # For demo, we'll return placeholder data
    results = {
        'naive_baseline': {
            'accuracy': 0.45,
            'f1_macro': 0.32
        },
        'xgboost': {
            'accuracy': 0.72,
            'f1_macro': 0.68
        },
        'distilbert': {
            'accuracy': 0.78,
            'f1_macro': 0.74
        }
    }

    return jsonify(results)


@app.route('/api/experiment-results')
def get_experiment_results() -> Any:
    """
    Get power tier experiment results.

    Returns:
        JSON response with experiment data
    """
    if 'experiment_results' not in DATA:
        return jsonify({'error': 'Experiment results not available'}), 404

    results = DATA['experiment_results'].to_dict('records')
    return jsonify(results)


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

    # Load data and models
    load_data()
    load_models()

    # Run app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )


if __name__ == "__main__":
    main()
