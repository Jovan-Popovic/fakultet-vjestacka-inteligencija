"""Definicije, obuka i evaluacija svih regresionih modela
zahtjevanih u zadatku."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import (LinearRegression, SGDRegressor,
                                   Ridge, Lasso, ElasticNet)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def build_regressors(random_state: int = 14) -> Dict[str, "object"]:
    """Vraća dict {ime: model} svih regresionih modela koji se traže."""

    models = {
        "Linearna regresija":
            LinearRegression(),

        "Polinomijalna regresija (st. 2)":
            Pipeline([("poly", PolynomialFeatures(degree=2,
                                                  interaction_only=False,
                                                  include_bias=False)),
                      ("lin",  LinearRegression())]),

        "SGD regresor":
            SGDRegressor(max_iter=1500, tol=1e-3, random_state=random_state),

        "Ridge regresija":
            Ridge(alpha=1.0, random_state=random_state),

        "Lasso regresija":
            Lasso(alpha=0.1, max_iter=5000, random_state=random_state),

        "Elastična mreža":
            ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000,
                       random_state=random_state),

        "SVR (linearno jezgro)":
            LinearSVR(C=1.0, epsilon=10.0, max_iter=5000,
                      random_state=random_state),

        "SVR (polinom. jezgro st. 2)":
            SVR(kernel="poly", degree=2, C=1.0, epsilon=10.0,
                gamma="scale"),

        "SVR (RBF jezgro)":
            SVR(kernel="rbf", C=1.0, epsilon=10.0, gamma="scale"),

        "Stablo odlučivanja":
            DecisionTreeRegressor(max_depth=8, random_state=random_state),

        "Random Forest (bagging)":
            RandomForestRegressor(n_estimators=120, max_depth=12,
                                  bootstrap=True, n_jobs=-1,
                                  random_state=random_state),

        "Random Forest (pasting)":
            BaggingRegressor(estimator=DecisionTreeRegressor(
                                 max_depth=12, random_state=random_state),
                             n_estimators=120,
                             max_samples=0.7, bootstrap=False,
                             n_jobs=-1, random_state=random_state),
    }
    return models


def evaluate_regressor(y_true, y_pred) -> Dict[str, float]:
    """Računa RMSE i MAE."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae}


def train_and_evaluate_all(models: Dict[str, "object"],
                           X_train, y_train, X_test, y_test
                           ) -> Tuple[Dict[str, Dict[str, float]], Dict]:
    """Trenira sve modele i vraća (metrike, fitovani modeli)."""
    metrics = {}
    fitted = {}
    for name, model in models.items():
        print(f"  • Obučavanje: {name} ...", end=" ", flush=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics[name] = evaluate_regressor(y_test, y_pred)
        fitted[name] = model
        print(f"RMSE={metrics[name]['RMSE']:.2f}  MAE={metrics[name]['MAE']:.2f}")
    return metrics, fitted


def metrics_to_dataframe(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(metrics).T.sort_values("RMSE")
    df.index.name = "Model"
    return df.round(2)
