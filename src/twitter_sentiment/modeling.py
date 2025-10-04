"""Model training utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .config import Config
from .logging_utils import get_logger
from .preprocessing import preprocess_dataframe

LOGGER = get_logger(__name__)

MetricsDict = Dict[str, float]


def build_pipeline(config: Config) -> Pipeline:
    model_settings = config.model

    vectorizer = TfidfVectorizer(
        max_features=model_settings.get("max_features", 5000),
        ngram_range=tuple(model_settings.get("ngram_range", (1, 1))),
        stop_words="english",
    )

    estimator = LogisticRegression(
        class_weight=model_settings.get("class_weight"),
        max_iter=1000,
        multi_class="auto",
        solver="lbfgs",
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("clf", estimator),
        ]
    )

    return pipeline


def _resolve_test_size(config: Config, target) -> float | int:
    test_size = config.model.get("test_size", 0.2)
    n_samples = len(target)
    n_classes = int(getattr(target, "nunique", lambda: len(set(target)))())

    if isinstance(test_size, float):
        computed = max(1, int(round(test_size * n_samples)))
        if computed < n_classes and n_samples > 0:
            adjusted = max(n_classes / n_samples, test_size)
            test_size = min(adjusted, 0.5)
            LOGGER.warning(
                "Adjusted test_size to %.2f to ensure each class appears in the test split",
                test_size,
            )
    else:
        if test_size < n_classes:
            test_size = n_classes
            LOGGER.warning(
                "Adjusted test_size to %d to ensure each class appears in the test split",
                test_size,
            )

    return test_size


def train_and_evaluate(config: Config) -> Tuple[Pipeline, MetricsDict]:
    df = preprocess_dataframe(load_dataset(config), config)

    data_settings = config.data
    text_column = data_settings.get("text_column", "text")
    target_column = data_settings.get("target_column", "sentiment")

    y_full = df[target_column].astype(str)
    test_size = _resolve_test_size(config, y_full)

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_column],
        y_full,
        test_size=test_size,
        random_state=config.model.get("random_state", 42),
        stratify=y_full,
    )

    pipeline = build_pipeline(config)
    LOGGER.info("Starting model training on %d samples", X_train.shape[0])
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    classification_report = metrics.classification_report(y_test, predictions, output_dict=True)

    metrics_summary = {
        "accuracy": metrics.accuracy_score(y_test, predictions),
        "f1_macro": metrics.f1_score(y_test, predictions, average="macro"),
        "precision_macro": metrics.precision_score(y_test, predictions, average="macro"),
        "recall_macro": metrics.recall_score(y_test, predictions, average="macro"),
    }
    LOGGER.info("Evaluation metrics: %s", metrics_summary)

    cv_folds = int(config.training.get("cv_folds", 0))
    if cv_folds > 1:
        value_counts = getattr(y_full, 'value_counts', lambda: [])()
        min_class_count = int(value_counts.min()) if len(value_counts) else 0
        if min_class_count < cv_folds:
            cv_folds = max(min_class_count, 1)
            LOGGER.warning("Reduced cv_folds to %d based on smallest class size", cv_folds)
        if cv_folds > 1:
            cv = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=config.model.get("random_state", 42),
            )
            cv_scores = cross_val_score(
                pipeline,
                df[text_column],
                y_full,
                cv=cv,
                scoring=config.training.get("scoring", "f1_macro"),
            )
            metrics_summary["cv_mean"] = float(np.mean(cv_scores))
            metrics_summary["cv_std"] = float(np.std(cv_scores))
            LOGGER.info(
                "Cross-validation %s: mean=%.4f std=%.4f",
                config.training.get("scoring", "f1_macro"),
                metrics_summary["cv_mean"],
                metrics_summary["cv_std"],
            )

    class_order = config.data.get("class_order", [])
    for label in class_order:
        if label in classification_report:
            metrics_summary[f"{label}_f1"] = classification_report[label]["f1-score"]

    return pipeline, metrics_summary


def persist_artifacts(pipeline: Pipeline, config: Config, metrics_summary: MetricsDict) -> Path:
    artifact_dir = Path(config.model.get("artifact_dir", "artifacts"))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = artifact_dir / config.model.get("pipeline_filename", "sentiment_pipeline.joblib")
    joblib.dump({"pipeline": pipeline, "metrics": metrics_summary}, artifact_path)

    LOGGER.info("Saved pipeline and metrics to %s", artifact_path)
    return artifact_path


# Avoid circular import by late import
from .data_loader import load_dataset  # noqa: E402  pylint: disable=wrong-import-position


__all__ = ["build_pipeline", "train_and_evaluate", "persist_artifacts"]
