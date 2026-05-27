"""Definicije, obuka i evaluacija svih klasifikacionih modela
zahtjevanih u zadatku."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score)


def build_classifiers(random_state: int = 14) -> Dict[str, "object"]:
    """Vraća dict {ime: model} svih klasifikacionih modela koji se traže.
    Svi modeli izlažu predict_proba ili decision_function za PR/ROC krive.
    """
    models = {
        "SGD klasifikator":
            CalibratedClassifierCV(
                SGDClassifier(loss="hinge", max_iter=1500, tol=1e-3,
                              random_state=random_state),
                method="sigmoid", cv=3),

        "Logistička regresija":
            LogisticRegression(max_iter=2000, multi_class="auto",
                               random_state=random_state),

        "SVC (linearno jezgro)":
            CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=5000,
                          random_state=random_state),
                method="sigmoid", cv=3),

        "SVC (polinom. jezgro st. 2)":
            SVC(kernel="poly", degree=2, C=1.0, gamma="scale",
                probability=True, random_state=random_state),

        "SVC (RBF jezgro)":
            SVC(kernel="rbf", C=1.0, gamma="scale",
                probability=True, random_state=random_state),

        "Stablo odlučivanja":
            DecisionTreeClassifier(max_depth=10, random_state=random_state),

        "Random Forest (bagging)":
            RandomForestClassifier(n_estimators=120, max_depth=14,
                                   bootstrap=True, n_jobs=-1,
                                   random_state=random_state),

        "Random Forest (pasting)":
            BaggingClassifier(estimator=DecisionTreeClassifier(
                                  max_depth=14, random_state=random_state),
                              n_estimators=120,
                              max_samples=0.7, bootstrap=False,
                              n_jobs=-1, random_state=random_state),
    }
    return models


def evaluate_classifier(y_true, y_pred) -> Dict[str, float]:
    """Tačnost, makro-prosjek preciznosti, opoziva i F1."""
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro",
                                            zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro",
                                         zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="macro",
                                     zero_division=0)),
    }


def train_and_evaluate_all(models: Dict[str, "object"],
                           X_train, y_train, X_test, y_test
                           ) -> Tuple[Dict[str, Dict[str, float]],
                                      Dict[str, np.ndarray],
                                      Dict[str, np.ndarray],
                                      Dict]:
    """Trenira sve modele i vraća (metrike, predikcije, proba, fitovani modeli)."""
    metrics, predictions, probabilities, fitted = {}, {}, {}, {}
    for name, model in models.items():
        print(f"  • Obučavanje: {name} ...", end=" ", flush=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics[name] = evaluate_classifier(y_test, y_pred)
        predictions[name] = y_pred
        try:
            probabilities[name] = model.predict_proba(X_test)
        except (AttributeError, ValueError):
            probabilities[name] = None
        fitted[name] = model
        print(f"acc={metrics[name]['accuracy']:.3f}  "
              f"f1={metrics[name]['f1']:.3f}")
    return metrics, predictions, probabilities, fitted


def metrics_to_dataframe(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(metrics).T.sort_values("f1", ascending=False)
    df.index.name = "Model"
    return df.round(4)
