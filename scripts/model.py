# Project: The Impunity Index
# Authors: Lindsay Gross, Shreya Mendi, Andrew Jin
# Advisor: Brinnae Bent, PhD
# Claude chat: https://claude.ai/chat/f8744002-3279-48ab-9d9a-8efa1fdb1af1
# Built with Claude AI assistance

"""
Train and evaluate classification models for consequence prediction.

This script trains four models to predict whether an individual faced
consequences (binary: 0 = none, 1 = any consequence) based on NLP features
extracted from Epstein case documents.

Models:
    0. MajorityClassifier  — naive baseline (always predicts majority class)
    1. Logistic Regression  — balanced baseline on 7 tabular features
    2. Random Forest + TF-IDF — combines document text with tabular features
    3. SentenceTransformer + LinearSVC — deep learning: all-MiniLM-L6-v2
       embeddings (fixed encoder, no fine-tuning) + tabular features + SVM
    (Legacy) Legal-BERT — fine-tuned nlpaueb/legal-bert-base-uncased (deprecated)

Note: With only 66 people (51 no-consequence, 15 with consequences),
we use binary classification and careful cross-validation. The 3-class
tier analysis is performed separately in the experiment.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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
        consequences_path: str = "data/processed/consequences.csv",
        corpora_path: str = "data/processed/person_corpora.json",
    ):
        """
        Initialize the model trainer.

        Args:
            features_path: Path to features CSV
            consequences_path: Path to consequences CSV
            corpora_path: Path to per-person document corpora JSON
        """
        self.features_path = str(_resolve_path(features_path))
        self.consequences_path = str(_resolve_path(consequences_path))
        self.corpora_path = str(_resolve_path(corpora_path))
        self.results = {}

        logger.info("Loading data...")
        self.load_data()

    def load_data(self) -> None:
        """Load and merge features, consequences, and optionally text corpora."""
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

        # Prepare feature columns (7 tabular features)
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

        # Load text corpora (for RF + TF-IDF model)
        self.person_corpora = {}
        corpora_file = Path(self.corpora_path)
        if corpora_file.exists():
            with open(corpora_file) as f:
                self.person_corpora = json.load(f)
            non_empty = sum(1 for v in self.person_corpora.values() if v)
            logger.info(f"Loaded text corpora: {non_empty}/{len(self.person_corpora)} people have documents")
        else:
            logger.warning(f"No text corpora found at {corpora_file}. "
                          f"RF+TF-IDF will use tabular features only.")

    # ------------------------------------------------------------------
    # MODEL 0: MajorityClassifier (naive baseline — rubric requirement)
    # ------------------------------------------------------------------
    def train_majority_classifier(self) -> Dict[str, Any]:
        """
        Train a naive majority-class baseline.

        Always predicts the most frequent class (no consequence). This is
        the rubric-required naive baseline — any useful model must beat it.

        Returns:
            Dictionary with model and metrics
        """
        logger.info("=" * 60)
        logger.info("MODEL 0: MajorityClassifier (Naive Baseline)")
        logger.info("=" * 60)

        model = DummyClassifier(strategy="most_frequent", random_state=42)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        # DummyClassifier supports predict_proba for most_frequent
        y_prob = model.predict_proba(self.X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        majority_class = self.y_train.mode()[0]
        class_dist = self.y_train.value_counts(normalize=True).to_dict()

        logger.info(
            f"Majority class: {majority_class} "
            f"(training distribution: {class_dist})"
        )
        logger.info(f"MajorityClassifier - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(
            "NOTE: F1 is low because majority classifier never predicts "
            "the minority class (consequence), so macro avg is penalized."
        )
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        logger.info(
            f"Classification Report:\n"
            f"{classification_report(self.y_test, y_pred, target_names=['None', 'Consequence'], zero_division=0)}"
        )

        # Save model
        model_dir = _resolve_path("models")
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': model}, model_dir / "majority_classifier.pkl")
        logger.info(f"Saved MajorityClassifier to {model_dir / 'majority_classifier.pkl'}")

        return {
            'model': model,
            'accuracy': accuracy,
            'f1_macro': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred,
            'probabilities': y_prob,
            'majority_class': int(majority_class),
            'class_distribution': {str(k): float(v) for k, v in class_dist.items()},
        }

    # ------------------------------------------------------------------
    # MODEL 1: Logistic Regression (replaces DummyClassifier)
    # ------------------------------------------------------------------
    def train_logistic_regression(self) -> Dict[str, Any]:
        """
        Train Logistic Regression baseline on 7 tabular features.

        Uses balanced class weights to handle the 51:15 imbalance,
        StandardScaler for feature normalization, and GridSearchCV
        over regularization strength.

        Returns:
            Dictionary with model and metrics
        """
        logger.info("=" * 60)
        logger.info("MODEL 1: Logistic Regression (Baseline)")
        logger.info("=" * 60)

        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        # GridSearchCV over regularization strength
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
        }

        base_model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train_scaled, self.y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")

        model = grid_search.best_estimator_

        # Evaluate on test set
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        logger.info(f"Logistic Regression - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        logger.info(
            f"Classification Report:\n"
            f"{classification_report(self.y_test, y_pred, target_names=['None', 'Consequence'])}"
        )

        # Feature coefficients
        coef_df = pd.DataFrame({
            'feature': self.feature_cols,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', ascending=False)
        logger.info(f"Feature Coefficients:\n{coef_df}")

        # Save model + scaler
        model_dir = _resolve_path("models")
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': model, 'scaler': scaler}, model_dir / "logistic_regression.pkl")
        logger.info(f"Saved Logistic Regression model to {model_dir / 'logistic_regression.pkl'}")

        return {
            'model': model,
            'scaler': scaler,
            'accuracy': accuracy,
            'f1_macro': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred,
            'probabilities': y_prob,
            'best_params': grid_search.best_params_,
            'feature_coefficients': coef_df,
        }

    # ------------------------------------------------------------------
    # MODEL 2: Random Forest + TF-IDF (replaces Gradient Boosting)
    # ------------------------------------------------------------------
    def train_random_forest_tfidf(self) -> Dict[str, Any]:
        """
        Train Random Forest combining TF-IDF text features with tabular features.

        For each person, their document corpus is vectorized with TF-IDF,
        then horizontally stacked with the 7 tabular features. Uses
        GridSearchCV for hyperparameter tuning.

        Falls back to tabular-only Random Forest if no corpora available.

        Returns:
            Dictionary with model and metrics
        """
        logger.info("=" * 60)
        logger.info("MODEL 2: Random Forest + TF-IDF")
        logger.info("=" * 60)

        # Scale tabular features
        scaler = StandardScaler()
        X_train_tab = scaler.fit_transform(self.X_train)
        X_test_tab = scaler.transform(self.X_test)

        # Get names for train/test
        train_names = self.df.loc[self.X_train.index, 'name'].tolist()
        test_names = self.df.loc[self.X_test.index, 'name'].tolist()

        # Build text corpora for train and test
        has_corpora = bool(self.person_corpora)
        tfidf = None

        if has_corpora:
            train_texts = [self.person_corpora.get(n, "") for n in train_names]
            test_texts = [self.person_corpora.get(n, "") for n in test_names]

            # Check if we actually have text
            non_empty_train = sum(1 for t in train_texts if t.strip())
            logger.info(f"  Train texts available: {non_empty_train}/{len(train_texts)}")

            if non_empty_train >= 5:
                # TF-IDF vectorization
                tfidf = TfidfVectorizer(
                    max_features=500,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                    max_df=0.95,
                    stop_words='english',
                )

                X_train_tfidf = tfidf.fit_transform(train_texts)
                X_test_tfidf = tfidf.transform(test_texts)

                logger.info(f"  TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")
                logger.info(f"  TF-IDF matrix shape: {X_train_tfidf.shape}")

                # Combine TF-IDF with tabular features
                X_train_combined = sparse.hstack([
                    sparse.csr_matrix(X_train_tab),
                    X_train_tfidf
                ])
                X_test_combined = sparse.hstack([
                    sparse.csr_matrix(X_test_tab),
                    X_test_tfidf
                ])

                logger.info(f"  Combined feature matrix: {X_train_combined.shape}")
            else:
                logger.warning("  Not enough text data, falling back to tabular-only RF")
                has_corpora = False

        if not has_corpora or tfidf is None:
            logger.info("  Using tabular features only (no text corpora)")
            X_train_combined = X_train_tab
            X_test_combined = X_test_tab

        # GridSearchCV for Random Forest
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 10, None],
            'class_weight': ['balanced', 'balanced_subsample'],
            'min_samples_leaf': [1, 2, 3],
        }

        base_model = RandomForestClassifier(random_state=42)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train_combined, self.y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")

        model = grid_search.best_estimator_

        # Evaluate on test set
        y_pred = model.predict(X_test_combined)
        y_prob = model.predict_proba(X_test_combined)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        logger.info(f"Random Forest+TF-IDF - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        logger.info(
            f"Classification Report:\n"
            f"{classification_report(self.y_test, y_pred, target_names=['None', 'Consequence'])}"
        )

        # Feature importances (top 20)
        n_tabular = len(self.feature_cols)
        importances = model.feature_importances_

        if tfidf is not None:
            vocab = tfidf.get_feature_names_out()
            all_feature_names = list(self.feature_cols) + list(vocab)
        else:
            all_feature_names = list(self.feature_cols)

        # Only show top features if we have many
        n_show = min(20, len(all_feature_names))
        top_idx = np.argsort(importances)[::-1][:n_show]

        feature_importances = pd.DataFrame({
            'feature': [all_feature_names[i] for i in top_idx],
            'importance': importances[top_idx],
            'type': ['tabular' if i < n_tabular else 'tfidf' for i in top_idx]
        })

        logger.info(f"Top {n_show} Feature Importances:\n{feature_importances}")

        # Save model + artifacts
        model_dir = _resolve_path("models")
        model_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {
            'model': model,
            'scaler': scaler,
            'tfidf': tfidf,
            'feature_cols': self.feature_cols,
        }
        joblib.dump(artifacts, model_dir / "random_forest_tfidf.pkl")
        logger.info(f"Saved RF+TF-IDF model to {model_dir / 'random_forest_tfidf.pkl'}")

        return {
            'model': model,
            'scaler': scaler,
            'tfidf': tfidf,
            'accuracy': accuracy,
            'f1_macro': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred,
            'probabilities': y_prob,
            'feature_importances': feature_importances,
            'best_params': grid_search.best_params_,
            'has_tfidf': tfidf is not None,
        }

    # ------------------------------------------------------------------
    # MODEL 3: Legal-BERT (replaces DistilBERT)
    # ------------------------------------------------------------------
    def train_legal_bert(
        self,
        doc_dataset_path: str = "data/processed/doc_level_dataset.csv",
        max_length: int = 512,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
    ) -> Optional[Dict[str, Any]]:
        """
        Fine-tune Legal-BERT on document-level classification task.

        Uses nlpaueb/legal-bert-base-uncased (110M params, trained on 12GB
        of legal text including US court opinions, EU legislation, etc.)

        The model is trained on document-level samples (2000+ when jmail
        data is available) rather than person-level samples (66). At
        inference, document predictions are aggregated per-person.

        Falls back to training on person-level synthetic text if no
        document-level dataset is available.

        Args:
            doc_dataset_path: Path to document-level dataset CSV
            max_length: Maximum token length
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate

        Returns:
            Dictionary with model and metrics, or None if deps missing
        """
        try:
            import torch
            from torch.utils.data import Dataset, DataLoader
            from transformers import (
                AutoTokenizer, AutoModelForSequenceClassification,
                Trainer, TrainingArguments
            )
        except ImportError as e:
            logger.warning(f"Legal-BERT dependencies not available: {e}")
            logger.warning("Skipping Legal-BERT training")
            return None

        logger.info("=" * 60)
        logger.info("MODEL 3: Legal-BERT (nlpaueb/legal-bert-base-uncased)")
        logger.info("=" * 60)

        model_name = "nlpaueb/legal-bert-base-uncased"

        # Try to load document-level dataset
        doc_path = _resolve_path(doc_dataset_path)
        use_doc_level = doc_path.exists()

        if use_doc_level:
            doc_df = pd.read_csv(doc_path)
            logger.info(f"Loaded document-level dataset: {len(doc_df)} samples")
            logger.info(f"  Label distribution: {doc_df['label'].value_counts().to_dict()}")
            logger.info(f"  Unique people: {doc_df['person_name'].nunique()}")

            if len(doc_df) < 20:
                logger.warning("Too few document-level samples, falling back to person-level")
                use_doc_level = False

        if use_doc_level:
            # Split by person to prevent data leakage
            unique_people = doc_df['person_name'].unique()
            np.random.seed(42)
            np.random.shuffle(unique_people)
            split_idx = int(len(unique_people) * 0.8)
            train_people = set(unique_people[:split_idx])
            test_people = set(unique_people[split_idx:])

            train_doc_df = doc_df[doc_df['person_name'].isin(train_people)]
            test_doc_df = doc_df[doc_df['person_name'].isin(test_people)]

            # Balance training set if very imbalanced
            pos_count = train_doc_df['label'].sum()
            neg_count = len(train_doc_df) - pos_count

            if neg_count > pos_count * 3:
                # Downsample negatives
                neg_df = train_doc_df[train_doc_df['label'] == 0].sample(
                    n=min(pos_count * 3, neg_count), random_state=42
                )
                pos_df = train_doc_df[train_doc_df['label'] == 1]
                train_doc_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42)

            train_texts = train_doc_df['text_window'].tolist()
            train_labels = train_doc_df['label'].tolist()
            test_texts = test_doc_df['text_window'].tolist()
            test_labels = test_doc_df['label'].tolist()

            logger.info(f"  Doc-level train: {len(train_texts)} (pos={sum(train_labels)})")
            logger.info(f"  Doc-level test: {len(test_texts)} (pos={sum(test_labels)})")

        else:
            # Fall back to person-level with text from corpora
            logger.info("Using person-level text (from corpora or feature-derived)")

            train_idx = self.X_train.index
            test_idx = self.X_test.index
            train_names = self.df.loc[train_idx, 'name'].tolist()
            test_names = self.df.loc[test_idx, 'name'].tolist()

            def get_person_text(name: str, row: pd.Series) -> str:
                """Get text for a person from corpora or synthesize from features."""
                if self.person_corpora.get(name, ""):
                    # Use first 2048 chars of their document corpus
                    return self.person_corpora[name][:2048]
                # Fallback: synthesize from features
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

            train_texts = [
                get_person_text(name, self.df.loc[idx])
                for name, idx in zip(train_names, train_idx)
            ]
            test_texts = [
                get_person_text(name, self.df.loc[idx])
                for name, idx in zip(test_names, test_idx)
            ]
            train_labels = self.y_train.tolist()
            test_labels = self.y_test.tolist()

        # Load tokenizer and model
        logger.info(f"Loading {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2
            )
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            logger.info("Falling back to distilbert-base-uncased")
            model_name = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2
            )

        logger.info(f"Model loaded: {model_name}")
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"  Parameters: {param_count:,}")

        # Simple PyTorch dataset
        class TextDataset(Dataset):
            def __init__(self, texts, labels, tok, max_len):
                self.texts = texts
                self.labels = labels
                self.tok = tok
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = str(self.texts[idx])[:5000]  # Cap input length
                enc = self.tok(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_len,
                    return_tensors='pt'
                )
                import torch as _torch
                return {
                    'input_ids': enc['input_ids'].flatten(),
                    'attention_mask': enc['attention_mask'].flatten(),
                    'labels': _torch.tensor(self.labels[idx], dtype=_torch.long)
                }

        train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
        test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length)

        # Training arguments
        output_dir = str(_resolve_path("models/legal_bert"))

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=10,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            seed=42,
            report_to='none',
            fp16=False,  # MPS may not support fp16 well
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

        # Train
        logger.info("Starting Legal-BERT training...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.train()

        # Evaluate
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_prob = predictions.predictions[:, 1]  # raw logits, not probs
        # Convert logits to probabilities
        from scipy.special import softmax
        probs = softmax(predictions.predictions, axis=1)
        y_prob = probs[:, 1]

        accuracy = accuracy_score(test_labels, y_pred)
        f1_val = f1_score(test_labels, y_pred, average='macro')
        conf_matrix = confusion_matrix(test_labels, y_pred)

        logger.info(f"Legal-BERT - Accuracy: {accuracy:.4f}, F1: {f1_val:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        logger.info(
            f"Classification Report:\n"
            f"{classification_report(test_labels, y_pred, target_names=['None', 'Consequence'])}"
        )

        # If we used doc-level training, aggregate to person-level predictions
        person_predictions = {}
        if use_doc_level:
            logger.info("\nAggregating document predictions to person level...")
            test_doc_df = test_doc_df.copy()
            test_doc_df['pred_prob'] = y_prob
            test_doc_df['pred_label'] = y_pred

            for person, group in test_doc_df.groupby('person_name'):
                mean_prob = group['pred_prob'].mean()
                person_predictions[person] = {
                    'probability': float(mean_prob),
                    'prediction': int(mean_prob >= 0.5),
                    'n_documents': len(group),
                }

            logger.info(f"  Person-level predictions: {len(person_predictions)}")

        # Save model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved Legal-BERT model to {output_dir}")

        return {
            'model': model,
            'tokenizer': tokenizer,
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_macro': f1_val,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred,
            'probabilities': y_prob,
            'person_predictions': person_predictions,
            'use_doc_level': use_doc_level,
            'n_train_samples': len(train_texts),
            'n_test_samples': len(test_texts),
        }

    # ------------------------------------------------------------------
    # MODEL 3b: SentenceTransformer + LinearSVC (deep learning model)
    # ------------------------------------------------------------------
    def train_sentence_transformer_svc(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        corpus_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Train LinearSVC on top of SentenceTransformer embeddings.

        Architecture:
            1. Embed each person's document corpus with all-MiniLM-L6-v2
               (384-dim), using mean-pooling over overlapping windows.
            2. Concatenate 384-dim semantic vector with 7 tabular features
               → 391-dim combined representation.
            3. Train LinearSVC (C grid-search) with class_weight='balanced'.
            4. Wrap with CalibratedClassifierCV to get calibrated probabilities.

        The encoder is used as a FIXED FEATURE EXTRACTOR — no fine-tuning.
        The model builds its own semantic space from the Epstein corpus.

        Falls back gracefully if sentence-transformers is not installed.

        Args:
            model_name: sentence-transformers model identifier.
            corpus_path: Override path to corpus JSONL.  If None, uses
                         the paper-trail sibling repo corpus.

        Returns:
            Dictionary with model and metrics, or None if deps missing.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install it in the paper-trail venv: "
                "pip install sentence-transformers"
            )
            return None

        logger.info("=" * 60)
        logger.info("MODEL 3b: SentenceTransformer + LinearSVC (deep learning)")
        logger.info("=" * 60)
        logger.info(f"Encoder: {model_name}  (fixed — no fine-tuning)")

        # ── Resolve corpus path ──────────────────────────────────────────
        if corpus_path is None:
            # Try sibling paper-trail repo first (2,492 docs)
            sibling = _resolve_path("..") / "epstein-paper-trail" / "data" / "raw" / "raw_corpus.jsonl"
            # Fall back to local person_corpora.json
            local_corpora = _resolve_path("data/processed/person_corpora.json")
            use_jsonl = sibling.exists()
        else:
            sibling = Path(corpus_path)
            use_jsonl = sibling.exists()
            local_corpora = _resolve_path("data/processed/person_corpora.json")

        # ── Build per-person text corpus ─────────────────────────────────
        name_to_text: Dict[str, str] = {}

        if use_jsonl:
            logger.info(f"Loading corpus from {sibling} ...")
            import json as _json
            per_person: Dict[str, List[str]] = {}
            with open(sibling, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _json.loads(line)
                    except Exception:
                        continue
                    text = (rec.get("text") or "").strip()
                    if not text:
                        continue
                    # Attribute this chunk to any person name it mentions
                    text_lower = text.lower()
                    for name in self.df["name"].tolist():
                        if name.lower() in text_lower:
                            per_person.setdefault(name, []).append(text)

            for name, chunks in per_person.items():
                name_to_text[name] = " ".join(chunks)[:50_000]  # cap at 50k chars

            logger.info(f"  Found text for {len(name_to_text)}/{len(self.df)} people")

        elif local_corpora.exists():
            import json as _json
            with open(local_corpora) as fh:
                raw = _json.load(fh)
            name_to_text = {k: v[:50_000] for k, v in raw.items() if v}
            logger.info(f"  Loaded local corpora: {len(name_to_text)} people")

        # ── Load encoder ─────────────────────────────────────────────────
        logger.info(f"Loading SentenceTransformer: {model_name} ...")
        encoder = SentenceTransformer(model_name)

        # ── Build embeddings per person ──────────────────────────────────
        def _embed_person(name: str) -> np.ndarray:
            """Mean-pool embeddings of 512-char windows over corpus text."""
            text = name_to_text.get(name, "")
            if not text:
                # Zero vector if no corpus — model sees only tabular features
                return np.zeros(384, dtype=np.float32)

            # Overlapping windows (512 chars, 128 overlap)
            windows = []
            step = 384
            for i in range(0, len(text), step):
                window = text[i: i + 512]
                if window.strip():
                    windows.append(window)

            if not windows:
                return np.zeros(384, dtype=np.float32)

            embs = encoder.encode(windows, batch_size=32, show_progress_bar=False,
                                  convert_to_numpy=True)
            return embs.mean(axis=0)

        logger.info("Embedding all people ...")
        all_names = self.df["name"].tolist()
        all_embeddings = np.vstack([_embed_person(n) for n in all_names])
        logger.info(f"  Embedding matrix: {all_embeddings.shape}")

        # ── Combine embeddings with tabular features ─────────────────────
        scaler = StandardScaler()
        X_tab = scaler.fit_transform(self.X)       # All 66 people, 7 features
        X_combined = np.hstack([all_embeddings, X_tab])  # (66, 391)

        # Re-split using same indices as the rest of training
        X_train_c = X_combined[self.X_train.index]
        X_test_c = X_combined[self.X_test.index]

        logger.info(f"  Combined feature dim: {X_combined.shape[1]}")
        logger.info(f"  Train: {X_train_c.shape}, Test: {X_test_c.shape}")

        # ── Train LinearSVC with calibration (for probabilities) ─────────
        param_grid = {"estimator__C": [0.01, 0.1, 1.0, 10.0]}

        base_svc = LinearSVC(
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )
        # CalibratedClassifierCV wraps SVC to give predict_proba
        calibrated = CalibratedClassifierCV(base_svc, cv=3, method="sigmoid")

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            calibrated,
            param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X_train_c, self.y_train)
        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV F1:  {grid_search.best_score_:.4f}")

        model = grid_search.best_estimator_

        # ── Evaluate ─────────────────────────────────────────────────────
        y_pred = model.predict(X_test_c)
        y_prob = model.predict_proba(X_test_c)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average="macro", zero_division=0)
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        logger.info(
            f"SentenceTransformer+SVC - Accuracy: {accuracy:.4f}, F1: {f1:.4f}"
        )
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        logger.info(
            f"Classification Report:\n"
            f"{classification_report(self.y_test, y_pred, target_names=['None', 'Consequence'], zero_division=0)}"
        )

        # ── Save model artifacts ─────────────────────────────────────────
        model_dir = _resolve_path("models")
        model_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {
            "model": model,
            "scaler": scaler,
            "encoder_name": model_name,
            "feature_cols": self.feature_cols,
            "name_to_text": name_to_text,  # for inference on new names
        }
        joblib.dump(artifacts, model_dir / "stsvc.pkl")
        logger.info(f"Saved ST+SVC model to {model_dir / 'stsvc.pkl'}")

        # Store per-person probabilities for use in impunity scoring
        person_probs: Dict[str, float] = {}
        for idx, name in zip(self.X_test.index, self.df.loc[self.X_test.index, "name"]):
            row_pos = list(self.X_test.index).index(idx)
            person_probs[name] = float(y_prob[row_pos])

        return {
            "model": model,
            "scaler": scaler,
            "encoder_name": model_name,
            "accuracy": accuracy,
            "f1_macro": f1,
            "confusion_matrix": conf_matrix.tolist(),
            "predictions": y_pred,
            "probabilities": y_prob,
            "person_probabilities": person_probs,
            "embedding_dim": 384,
            "combined_dim": X_combined.shape[1],
            "corpus_coverage": len(name_to_text),
        }

    # ------------------------------------------------------------------
    # EXPERIMENT: Power Tier Analysis
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # ABLATION STUDY
    # ------------------------------------------------------------------
    def run_ablation_study(self) -> pd.DataFrame:
        """
        Run feature ablation study using Random Forest.

        Systematically removes each feature (and feature groups) to measure
        how much each contributes to prediction performance. Includes
        text-vs-tabular comparison when TF-IDF is available.

        Returns:
            DataFrame with ablation results
        """
        logger.info("\n" + "=" * 60)
        logger.info("ABLATION STUDY: Feature Importance by Removal")
        logger.info("=" * 60)

        nlp_features = [
            'mention_count', 'total_mentions', 'mean_context_sentiment',
            'cooccurrence_score', 'doc_type_diversity', 'name_in_subject_line'
        ]

        ablation_runs = [
            ('All Features', self.feature_cols),
            ('Severity Score Only', ['severity_score']),
            ('NLP Features Only', nlp_features),
        ]

        # Add "drop one feature at a time" runs
        for feat in self.feature_cols:
            remaining = [f for f in self.feature_cols if f != feat]
            ablation_runs.append((f'Without {feat}', remaining))

        results = []

        for run_name, features in ablation_runs:
            X_train_sub = self.X_train[features]
            X_test_sub = self.X_test[features]

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                class_weight='balanced',
                random_state=42
            )

            model.fit(X_train_sub, self.y_train)
            y_pred = model.predict(X_test_sub)

            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='macro')

            results.append({
                'run': run_name,
                'n_features': len(features),
                'features': ', '.join(features),
                'accuracy': accuracy,
                'f1_macro': f1
            })

            logger.info(f"  {run_name:30s} | Acc: {accuracy:.4f} | F1: {f1:.4f}")

        # If we have TF-IDF, add text-only and combined runs
        if self.person_corpora:
            train_names = self.df.loc[self.X_train.index, 'name'].tolist()
            test_names = self.df.loc[self.X_test.index, 'name'].tolist()
            train_texts = [self.person_corpora.get(n, "") for n in train_names]
            test_texts = [self.person_corpora.get(n, "") for n in test_names]

            non_empty = sum(1 for t in train_texts if t.strip())
            if non_empty >= 5:
                tfidf = TfidfVectorizer(
                    max_features=500,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    stop_words='english',
                )
                X_train_tfidf = tfidf.fit_transform(train_texts)
                X_test_tfidf = tfidf.transform(test_texts)

                # TF-IDF Only
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=5,
                    class_weight='balanced', random_state=42
                )
                model.fit(X_train_tfidf, self.y_train)
                y_pred = model.predict(X_test_tfidf)
                results.append({
                    'run': 'TF-IDF Only (no tabular)',
                    'n_features': X_train_tfidf.shape[1],
                    'features': 'tfidf_500',
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'f1_macro': f1_score(self.y_test, y_pred, average='macro'),
                })
                logger.info(
                    f"  {'TF-IDF Only (no tabular)':30s} | "
                    f"Acc: {results[-1]['accuracy']:.4f} | F1: {results[-1]['f1_macro']:.4f}"
                )

                # Combined: Tabular + TF-IDF
                scaler = StandardScaler()
                X_train_tab_scaled = scaler.fit_transform(self.X_train)
                X_test_tab_scaled = scaler.transform(self.X_test)

                X_train_comb = sparse.hstack([
                    sparse.csr_matrix(X_train_tab_scaled), X_train_tfidf
                ])
                X_test_comb = sparse.hstack([
                    sparse.csr_matrix(X_test_tab_scaled), X_test_tfidf
                ])

                model = RandomForestClassifier(
                    n_estimators=100, max_depth=5,
                    class_weight='balanced', random_state=42
                )
                model.fit(X_train_comb, self.y_train)
                y_pred = model.predict(X_test_comb)
                results.append({
                    'run': 'Tabular + TF-IDF Combined',
                    'n_features': X_train_comb.shape[1],
                    'features': 'tabular + tfidf_500',
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'f1_macro': f1_score(self.y_test, y_pred, average='macro'),
                })
                logger.info(
                    f"  {'Tabular + TF-IDF Combined':30s} | "
                    f"Acc: {results[-1]['accuracy']:.4f} | F1: {results[-1]['f1_macro']:.4f}"
                )

        results_df = pd.DataFrame(results)

        # Compute delta from "All Features" baseline
        baseline_f1 = results_df.loc[
            results_df['run'] == 'All Features', 'f1_macro'
        ].values[0]
        results_df['f1_delta'] = results_df['f1_macro'] - baseline_f1

        # Log key findings
        logger.info("\n--- Key Findings ---")
        severity_only = results_df[results_df['run'] == 'Severity Score Only']
        nlp_only = results_df[results_df['run'] == 'NLP Features Only']
        logger.info(
            f"Severity alone: F1={severity_only['f1_macro'].values[0]:.4f} "
            f"(delta={severity_only['f1_delta'].values[0]:+.4f})"
        )
        logger.info(
            f"NLP features alone: F1={nlp_only['f1_macro'].values[0]:.4f} "
            f"(delta={nlp_only['f1_delta'].values[0]:+.4f})"
        )

        drop_runs = results_df[results_df['run'].str.startswith('Without')]
        if not drop_runs.empty:
            worst_drop = drop_runs.loc[drop_runs['f1_macro'].idxmin()]
            logger.info(
                f"Most important feature: {worst_drop['run']} → "
                f"F1={worst_drop['f1_macro']:.4f} "
                f"(delta={worst_drop['f1_delta']:+.4f})"
            )

        # Save results
        output_path = _resolve_path("data/outputs/ablation_results.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"\nSaved ablation results to {output_path}")

        return results_df

    # ------------------------------------------------------------------
    # ORCHESTRATION
    # ------------------------------------------------------------------
    def train_all_models(
        self,
        skip_bert: bool = True,
        skip_st: bool = False,
    ) -> Dict[str, Dict]:
        """
        Train all models and return results.

        Args:
            skip_bert: Skip Legal-BERT (deprecated, defaults to True).
            skip_st:   Skip SentenceTransformer+SVC model.

        Returns:
            Dictionary with results for all models
        """
        logger.info("Training all models...")

        results = {}

        # Model 0: MajorityClassifier (naive baseline)
        results['majority_classifier'] = self.train_majority_classifier()

        # Model 1: Logistic Regression
        results['logistic_baseline'] = self.train_logistic_regression()

        # Model 2: Random Forest + TF-IDF
        results['random_forest_tfidf'] = self.train_random_forest_tfidf()

        # Model 3b: SentenceTransformer + LinearSVC (deep learning)
        if not skip_st:
            st_result = self.train_sentence_transformer_svc()
            if st_result is not None:
                results['sentence_transformer_svc'] = st_result

        # Model 3 (Legacy): Legal-BERT — skip by default (deprecated)
        if not skip_bert:
            bert_result = self.train_legal_bert()
            if bert_result is not None:
                results['legal_bert'] = bert_result

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

        # Save predictions per person
        predictions_df = self.df[['name', 'consequence_tier', 'has_consequence']].copy()

        for model_name, result in self.results.items():
            pred_col = f'{model_name}_pred'
            prob_col = f'{model_name}_prob'
            predictions_df[pred_col] = None
            predictions_df[prob_col] = None

            if 'person_predictions' in result and result['person_predictions']:
                # Legal-BERT: use person-level aggregated predictions
                for person, ppred in result['person_predictions'].items():
                    mask = predictions_df['name'] == person
                    predictions_df.loc[mask, pred_col] = ppred['prediction']
                    predictions_df.loc[mask, prob_col] = ppred['probability']
            else:
                # Logistic Regression / RF: test set predictions
                predictions_df.loc[self.X_test.index, pred_col] = result['predictions']
                if 'probabilities' in result:
                    predictions_df.loc[self.X_test.index, prob_col] = result['probabilities']

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
        "--corpora-path", default="data/processed/person_corpora.json",
        help="Path to per-person document corpora JSON"
    )
    parser.add_argument(
        "--run-experiment", action="store_true",
        help="Run power tier experiment after training"
    )
    parser.add_argument(
        "--skip-bert", action="store_true", default=True,
        help="Skip Legal-BERT training (deprecated; defaults to True)"
    )
    parser.add_argument(
        "--run-bert", action="store_true",
        help="Enable Legal-BERT training (slow, deprecated)"
    )
    parser.add_argument(
        "--skip-st", action="store_true",
        help="Skip SentenceTransformer+SVC model"
    )
    parser.add_argument(
        "--run-ablation", action="store_true",
        help="Run feature ablation study"
    )
    args = parser.parse_args()

    trainer = ModelTrainer(
        features_path=args.features_path,
        consequences_path=args.consequences_path,
        corpora_path=args.corpora_path,
    )

    skip_bert = not args.run_bert  # default: skip Legal-BERT
    trainer.train_all_models(skip_bert=skip_bert, skip_st=args.skip_st)
    trainer.evaluate_all_models()

    if args.run_experiment:
        trainer.run_experiment()

    if args.run_ablation:
        trainer.run_ablation_study()


if __name__ == "__main__":
    main()
