# Attribution: Scaffolded with AI assistance (Claude, Anthropic)

"""
Train and evaluate all classification models.

This script trains three models: a naive baseline, XGBoost, and DistilBERT
to predict consequence tiers from NLP features.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    Trainer, TrainingArguments
)
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConsequenceDataset(Dataset):
    """PyTorch Dataset for consequence classification."""

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer: DistilBertTokenizer,
        max_length: int = 512
    ):
        """
        Initialize the dataset.

        Args:
            texts: List of text inputs
            labels: List of labels
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item.

        Args:
            idx: Item index

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


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
        self.features_path = features_path
        self.consequences_path = consequences_path
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
        logger.info(f"Class distribution:\n{self.df['consequence_tier'].value_counts()}")

        # Prepare feature columns
        self.feature_cols = [
            'mention_count', 'total_mentions', 'mean_context_sentiment',
            'cooccurrence_score', 'doc_type_diversity',
            'name_in_subject_line', 'severity_score'
        ]

        # Split data
        self.X = self.df[self.feature_cols].fillna(0)
        self.y = self.df['consequence_tier']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=0.2,
            stratify=self.y,
            random_state=42
        )

        logger.info(f"Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")

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
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }

    def train_xgboost(self) -> Dict[str, Any]:
        """
        Train XGBoost model with hyperparameter tuning.

        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training XGBoost with GridSearchCV...")

        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        base_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
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

        logger.info(f"XGBoost - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        # Feature importances
        feature_importances = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info(f"Feature Importances:\n{feature_importances}")

        # Save model
        model_path = Path("models/xgboost_model.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Saved XGBoost model to {model_path}")

        return {
            'model': model,
            'accuracy': accuracy,
            'f1_macro': f1,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'feature_importances': feature_importances,
            'best_params': grid_search.best_params_
        }

    def prepare_text_data(self) -> Tuple[list, list, list, list]:
        """
        Prepare text data for DistilBERT.

        Returns:
            Tuple of (train_texts, test_texts, train_labels, test_labels)
        """
        # For each person, create a text representation from their features
        # In a real scenario, you'd use actual document excerpts
        # Here we'll create synthetic text from features

        def create_text_representation(row: pd.Series) -> str:
            """Create text from features for demo purposes."""
            parts = [
                f"Person mentioned in {row['mention_count']} documents",
                f"with {row['total_mentions']} total mentions.",
                f"Sentiment score: {row['mean_context_sentiment']:.2f}.",
                f"Co-occurrence score: {row['cooccurrence_score']}.",
                f"Document diversity: {row['doc_type_diversity']}.",
                f"Severity score: {row['severity_score']:.2f}."
            ]
            return " ".join(parts)

        train_df = pd.concat([self.X_train, self.y_train], axis=1)
        test_df = pd.concat([self.X_test, self.y_test], axis=1)

        train_texts = [create_text_representation(row) for _, row in train_df.iterrows()]
        test_texts = [create_text_representation(row) for _, row in test_df.iterrows()]

        train_labels = self.y_train.tolist()
        test_labels = self.y_test.tolist()

        return train_texts, test_texts, train_labels, test_labels

    def train_distilbert(self) -> Dict[str, Any]:
        """
        Train DistilBERT model.

        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training DistilBERT...")

        # Prepare text data
        train_texts, test_texts, train_labels, test_labels = self.prepare_text_data()

        # Initialize tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=3
        )

        # Create datasets
        train_dataset = ConsequenceDataset(train_texts, train_labels, tokenizer)
        test_dataset = ConsequenceDataset(test_texts, test_labels, tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir='models/distilbert',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_dir='models/distilbert/logs',
            logging_steps=10,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            seed=42
        )

        # Define compute_metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            preds = np.argmax(predictions, axis=1)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='macro')
            return {'accuracy': acc, 'f1': f1}

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        # Train
        trainer.train()

        # Evaluate
        eval_results = trainer.evaluate()
        logger.info(f"DistilBERT evaluation results: {eval_results}")

        # Get predictions
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)

        accuracy = accuracy_score(test_labels, y_pred)
        f1 = f1_score(test_labels, y_pred, average='macro')
        conf_matrix = confusion_matrix(test_labels, y_pred)

        logger.info(f"DistilBERT - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        # Save model
        model_path = Path("models/distilbert")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        logger.info(f"Saved DistilBERT model to {model_path}")

        return {
            'model': model,
            'tokenizer': tokenizer,
            'accuracy': accuracy,
            'f1_macro': f1,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }

    def run_experiment(self) -> pd.DataFrame:
        """
        Run stratified experiment: Does power protect?

        Returns:
            DataFrame with experiment results
        """
        logger.info("Running power tier experiment...")

        # Define public figure tiers based on severity score quartiles
        # This is a simplified approach; ideally you'd have explicit categories
        self.df['power_tier'] = pd.qcut(
            self.df['severity_score'],
            q=4,
            labels=['low', 'medium', 'high', 'very_high']
        )

        results = []

        for tier in ['low', 'medium', 'high', 'very_high']:
            tier_df = self.df[self.df['power_tier'] == tier]

            if len(tier_df) < 3:
                logger.warning(f"Not enough data for tier {tier}")
                continue

            # Compute correlation
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
                'mean_consequence': tier_df['consequence_tier'].mean()
            })

            logger.info(
                f"Tier {tier}: n={len(tier_df)}, "
                f"correlation={corr:.3f}, p={p_value:.3f}"
            )

        results_df = pd.DataFrame(results)

        # Save results
        output_path = Path("data/outputs/experiment_results.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)

        logger.info(f"Saved experiment results to {output_path}")

        return results_df

    def train_all_models(self) -> Dict[str, Dict]:
        """
        Train all models and return results.

        Returns:
            Dictionary with results for all models
        """
        logger.info("Training all models...")

        results = {
            'naive_baseline': self.train_naive_baseline(),
            'xgboost': self.train_xgboost(),
            'distilbert': self.train_distilbert()
        }

        self.results = results
        return results

    def evaluate_all_models(self) -> None:
        """Print comparison table of all models."""
        if not self.results:
            logger.error("No results available. Run train_all_models() first.")
            return

        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON")
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

        # Save predictions
        predictions_df = self.df[['name', 'consequence_tier']].copy()
        predictions_df['actual'] = self.y_test

        for model_name, result in self.results.items():
            # Align predictions with test set
            pred_col = f'{model_name}_pred'
            predictions_df[pred_col] = None
            predictions_df.loc[self.X_test.index, pred_col] = result['predictions']

        output_path = Path("data/outputs/predictions.csv")
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"\nSaved predictions to {output_path}")


def main() -> None:
    """Main entry point for the script."""
    trainer = ModelTrainer()

    # Train all models
    trainer.train_all_models()

    # Evaluate and compare
    trainer.evaluate_all_models()

    # Run experiment
    trainer.run_experiment()


if __name__ == "__main__":
    main()
