# Attribution: Scaffolded with AI assistance (Claude, Anthropic)

"""
Train and evaluate classification models for consequence prediction.

This script trains three models to predict whether an individual faced
consequences (binary: 0 = none, 1 = any consequence) based on NLP features
extracted from Epstein case documents.

Models:
    1. Naive Baseline - Most frequent class (DummyClassifier)
    2. XGBoost - Gradient boosting with GridSearchCV
    3. DistilBERT - Fine-tuned transformer on feature-derived text

Note: With only 66 samples (51 no-consequence, 15 with consequences),
we use binary classification and careful cross-validation. The 3-class
tier analysis is performed separately in the experiment.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent.parent / p


class ModelTrainer:
    """Unified trainer for all models."""

    def __init__(
        self,
        features_path: str = "data/processed/features.csv",
        consequences_path: str = "data/processed/consequences.csv"
    ):
        """
        Initialize the model trainer.

        Args:
            features_path: Path to features CSV
            consequences_path: Path to consequences CSV
        """
        self.features_path = str(_resolve_path(features_path))
        self.consequences_path = str(_resolve_path(consequences_path))
        self.results = {}

        logger.info("Loading data...")
        self.load_data()

    def load_data(self) -> None:
        """Load and merge features and consequences data."""
        features_df = pd.read_csv(self.features_path)
        consequences_df = pd.read_csv(self.consequences_path)

        # Merge on name
        self.df = features_df.merge(
            consequences_df[['name', 'consequence_tier']],
            on='name',
            how='inner'
        )

        logger.info(f"Loaded {len(self.df)} records")
        logger.info(
            f"3-class distribution:\n"
            f"{self.df['consequence_tier'].value_counts().sort_index()}"
        )

        # Create binary label: 0 = no consequence, 1 = any consequence
        self.df['has_consequence'] = (self.df['consequence_tier'] > 0).astype(int)
        logger.info(
            f"Binary distribution:\n"
            f"{self.df['has_consequence'].value_counts().sort_index()}"
        )

        # Prepare feature columns
        self.feature_cols = [
            'mention_count', 'total_mentions', 'mean_context_sentiment',
            'cooccurrence_score', 'doc_type_diversity',
            'name_in_subject_line', 'severity_score'
        ]

        # Feature matrix and binary target
        self.X = self.df[self.feature_cols].fillna(0)
        self.y = self.df['has_consequence']

        # Split with stratification (binary is safe: 51 vs 15)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=0.2,
            stratify=self.y,
            random_state=42
        )

        logger.info(
            f"Train: {len(self.X_train)} (pos={self.y_train.sum()}), "
            f"Test: {len(self.X_test)} (pos={self.y_test.sum()})"
        )

    def train_naive_baseline(self) -> Dict[str, Any]:
        """
        Train naive baseline model (most frequent class).

        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training naive baseline...")

        model = DummyClassifier(strategy="most_frequent", random_state=42)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        logger.info(f"Naive Baseline - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        return {
            'model': model,
            'accuracy': accuracy,
            'f1_macro': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred
        }

    def train_gradient_boosting(self) -> Dict[str, Any]:
        """
        Train Gradient Boosting model with hyperparameter tuning.

        Uses sklearn's GradientBoostingClassifier (XGBoost-equivalent
        that works without OpenMP/libomp).

        Returns:
            Dictionary with model and metrics
        """
        from sklearn.ensemble import GradientBoostingClassifier

        logger.info("Training Gradient Boosting with GridSearchCV...")

        # Reduced parameter grid for small dataset
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [2, 3, 5],
            'learning_rate': [0.05, 0.1, 0.3],
            'subsample': [0.8, 1.0],
        }

        base_model = GradientBoostingClassifier(random_state=42)

        # 3-fold CV (safe for 12 positive training samples)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(self.X_train, self.y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")

        # Get best model
        model = grid_search.best_estimator_

        # Evaluate on test set
        y_pred = model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        logger.info(f"Gradient Boosting - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        logger.info(
            f"Classification Report:\n"
            f"{classification_report(self.y_test, y_pred, target_names=['None', 'Consequence'])}"
        )

        # Feature importances
        feature_importances = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"Feature Importances:\n{feature_importances}")

        # Save model
        model_path = _resolve_path("models/gradient_boosting_model.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Saved Gradient Boosting model to {model_path}")

        return {
            'model': model,
            'accuracy': accuracy,
            'f1_macro': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred,
            'feature_importances': feature_importances,
            'best_params': grid_search.best_params_
        }

    def train_distilbert(self) -> Optional[Dict[str, Any]]:
        """
        Train DistilBERT model on feature-derived text.

        For 66 samples, this is a proof-of-concept demonstrating that
        transformer models can be applied to this task. Performance
        likely won't exceed XGBoost on tabular features alone.

        Returns:
            Dictionary with model and metrics, or None if deps missing
        """
        try:
            import torch
            from torch.utils.data import Dataset
            from transformers import (
                DistilBertTokenizer, DistilBertForSequenceClassification,
                Trainer, TrainingArguments
            )
        except ImportError as e:
            logger.warning(f"DistilBERT dependencies not available: {e}")
            logger.warning("Skipping DistilBERT training")
            return None

        logger.info("Training DistilBERT...")

        # Create text representations from features
        def create_text(row: pd.Series) -> str:
            parts = [
                f"This person was mentioned in {int(row['mention_count'])} documents",
                f"with {int(row['total_mentions'])} total mentions.",
                f"The average sentiment around their name was {row['mean_context_sentiment']:.2f}.",
                f"They co-occurred with incriminating keywords {int(row['cooccurrence_score'])} times.",
                f"They appeared across {int(row['doc_type_diversity'])} document types.",
                f"Their severity score is {row['severity_score']:.1f} out of 10.",
            ]
            if row['name_in_subject_line']:
                parts.append("Their name appeared in email subject lines.")
            return " ".join(parts)

        train_df = pd.concat([self.X_train, self.y_train], axis=1)
        test_df = pd.concat([self.X_test, self.y_test], axis=1)

        train_texts = [create_text(row) for _, row in train_df.iterrows()]
        test_texts = [create_text(row) for _, row in test_df.iterrows()]
        train_labels = self.y_train.tolist()
        test_labels = self.y_test.tolist()

        # Tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        )

        # Simple PyTorch dataset
        class TextDataset(Dataset):
            def __init__(self, texts, labels, tok, max_len=128):
                self.texts = texts
                self.labels = labels
                self.tok = tok
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                enc = self.tok(
                    self.texts[idx],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_len,
                    return_tensors='pt'
                )
                return {
                    'input_ids': enc['input_ids'].flatten(),
                    'attention_mask': enc['attention_mask'].flatten(),
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                }

        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        test_dataset = TextDataset(test_texts, test_labels, tokenizer)

        # Training arguments (conservative for small dataset)
        output_dir = str(_resolve_path("models/distilbert"))
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=5,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            seed=42,
            report_to='none',  # Disable wandb etc.
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            preds = np.argmax(predictions, axis=1)
            acc = accuracy_score(labels, preds)
            f1_val = f1_score(labels, preds, average='macro')
            return {'accuracy': acc, 'f1': f1_val}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        # Suppress excessive logging
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.train()

        # Evaluate
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)

        accuracy = accuracy_score(test_labels, y_pred)
        f1_val = f1_score(test_labels, y_pred, average='macro')
        conf_matrix = confusion_matrix(test_labels, y_pred)

        logger.info(f"DistilBERT - Accuracy: {accuracy:.4f}, F1: {f1_val:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        # Save model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved DistilBERT model to {output_dir}")

        return {
            'model': model,
            'tokenizer': tokenizer,
            'accuracy': accuracy,
            'f1_macro': f1_val,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred
        }

    def run_experiment(self) -> pd.DataFrame:
        """
        Run stratified experiment: Does power protect?

        Bins people by severity score into power tiers and checks whether
        higher severity correlates with (or protects from) consequences.

        Returns:
            DataFrame with experiment results
        """
        logger.info("Running power tier experiment...")

        # Use the full dataset (not train/test split) for the experiment
        exp_df = self.df.copy()

        # Create power tiers using severity score thresholds
        # (qcut can fail with duplicates, so use manual bins)
        bins = [0, 2, 5, 8, 10.01]
        labels = ['low', 'medium', 'high', 'very_high']
        exp_df['power_tier'] = pd.cut(
            exp_df['severity_score'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        results = []

        for tier in labels:
            tier_df = exp_df[exp_df['power_tier'] == tier]

            if len(tier_df) < 3:
                logger.warning(f"Not enough data for tier {tier} (n={len(tier_df)})")
                continue

            # Check if there's variance in both columns
            if tier_df['severity_score'].std() == 0 or tier_df['consequence_tier'].std() == 0:
                corr, p_value = 0.0, 1.0
            else:
                corr, p_value = pearsonr(
                    tier_df['severity_score'],
                    tier_df['consequence_tier']
                )

            results.append({
                'power_tier': tier,
                'n_people': len(tier_df),
                'correlation': corr,
                'p_value': p_value,
                'mean_severity': tier_df['severity_score'].mean(),
                'mean_consequence': tier_df['consequence_tier'].mean(),
                'pct_with_consequence': (tier_df['consequence_tier'] > 0).mean()
            })

            logger.info(
                f"Tier {tier}: n={len(tier_df)}, "
                f"corr={corr:.3f}, p={p_value:.3f}, "
                f"pct_consequence={results[-1]['pct_with_consequence']:.1%}"
            )

        results_df = pd.DataFrame(results)

        # Save results
        output_path = _resolve_path("data/outputs/experiment_results.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved experiment results to {output_path}")

        # Also compute overall correlation
        overall_corr, overall_p = pearsonr(
            exp_df['severity_score'],
            exp_df['consequence_tier']
        )
        logger.info(
            f"\nOverall severity-consequence correlation: "
            f"{overall_corr:.3f} (p={overall_p:.4f})"
        )

        return results_df

    def train_all_models(self) -> Dict[str, Dict]:
        """
        Train all models and return results.

        Returns:
            Dictionary with results for all models
        """
        logger.info("Training all models...")

        results = {}
        results['naive_baseline'] = self.train_naive_baseline()
        results['gradient_boosting'] = self.train_gradient_boosting()

        distilbert_result = self.train_distilbert()
        if distilbert_result is not None:
            results['distilbert'] = distilbert_result

        self.results = results
        return results

    def evaluate_all_models(self) -> None:
        """Print comparison table and save predictions."""
        if not self.results:
            logger.error("No results available. Run train_all_models() first.")
            return

        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON (Binary: Consequence vs None)")
        logger.info("=" * 60)

        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'F1 (Macro)': f"{result['f1_macro']:.4f}"
            })

        comparison_df = pd.DataFrame(comparison_data)
        logger.info(f"\n{comparison_df.to_string(index=False)}")

        # Save model metrics JSON
        metrics = {}
        for model_name, result in self.results.items():
            metrics[model_name] = {
                'accuracy': result['accuracy'],
                'f1_macro': result['f1_macro'],
                'confusion_matrix': result['confusion_matrix']
            }

        metrics_path = _resolve_path("data/outputs/model_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved model metrics to {metrics_path}")

        # Save predictions
        predictions_df = self.df[['name', 'consequence_tier', 'has_consequence']].copy()

        for model_name, result in self.results.items():
            pred_col = f'{model_name}_pred'
            predictions_df[pred_col] = None
            predictions_df.loc[self.X_test.index, pred_col] = result['predictions']

        output_path = _resolve_path("data/outputs/predictions.csv")
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")


def main() -> None:
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train consequence prediction models"
    )
    parser.add_argument(
        "--features-path", default="data/processed/features.csv",
        help="Path to features CSV"
    )
    parser.add_argument(
        "--consequences-path", default="data/processed/consequences.csv",
        help="Path to consequences CSV"
    )
    parser.add_argument(
        "--run-experiment", action="store_true",
        help="Run power tier experiment after training"
    )
    parser.add_argument(
        "--skip-distilbert", action="store_true",
        help="Skip DistilBERT training (faster)"
    )
    args = parser.parse_args()

    trainer = ModelTrainer(
        features_path=args.features_path,
        consequences_path=args.consequences_path
    )

    if args.skip_distilbert:
        # Train only baseline and Gradient Boosting
        trainer.results['naive_baseline'] = trainer.train_naive_baseline()
        trainer.results['gradient_boosting'] = trainer.train_gradient_boosting()
    else:
        trainer.train_all_models()

    trainer.evaluate_all_models()

    if args.run_experiment:
        trainer.run_experiment()


if __name__ == "__main__":
    main()
