"""Funkcije za vizuelizaciju retail sales dataset-a i performansi modela."""

from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              precision_recall_curve, roc_curve, auc)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize

sns.set_style("whitegrid")


# ---------------- EKSPLORATORNE VIZUELIZACIJE ---------------- #
def plot_histograms(df: pd.DataFrame, numeric_cols: List[str],
                    save_path: Optional[str] = None,
                    bins: int = 30) -> None:
    """Crta histograme za sve numeričke kolone u jednoj figuri."""
    n = len(numeric_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=bins, color="steelblue",
                     edgecolor="black", alpha=0.85)
        axes[i].set_title(f"Histogram: {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frekvencija")
        axes[i].grid(True, alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, numeric_cols: List[str],
                            save_path: Optional[str] = None) -> None:
    """Crta heatmap korelacione matrice numeričkih atributa."""
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(0.8 * len(numeric_cols) + 3,
                                    0.8 * len(numeric_cols) + 2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Korelaciona matrica", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


def plot_combined_attributes(df: pd.DataFrame, x: str, y: str, hue: str,
                             save_path: Optional[str] = None) -> None:
    """Smislena kombinacija: scatter plot dva numerička atributa sa
    bojom po trećem (kategoričkom) atributu.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.55,
                    edgecolor="black", linewidth=0.3, ax=ax,
                    palette="tab10", s=35)
    ax.set_title(f"Kombinacija: {x} vs {y} po {hue}", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


def plot_category_target_boxplot(df: pd.DataFrame, category: str,
                                 target: str,
                                 save_path: Optional[str] = None) -> None:
    """Box plot raspodjele target-a po kategoriji."""
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=df, x=category, y=target, ax=ax, palette="Set2")
    ax.set_title(f"Raspodjela {target} po {category}", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


# ---------------- PERFORMANSE REGRESIONIH MODELA ---------------- #
def plot_learning_curve(estimator, X, y, title: str, scoring: str = "neg_root_mean_squared_error",
                        cv: int = 3, n_jobs: int = -1,
                        train_sizes=np.linspace(0.1, 1.0, 6),
                        save_path: Optional[str] = None) -> None:
    """Crta krivu učenja za jedan model (trening + validacijska greška)."""
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
        train_sizes=train_sizes, shuffle=True, random_state=14)

    # neg_RMSE -> RMSE
    if "neg_" in scoring:
        train_scores = -train_scores
        val_scores = -val_scores

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes_abs, train_mean, "o-", color="steelblue", label="trening")
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    color="steelblue", alpha=0.15)
    ax.plot(train_sizes_abs, val_mean, "s-", color="crimson", label="validacija (CV)")
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                    color="crimson", alpha=0.15)
    ax.set_title(f"Kriva učenja: {title}", fontweight="bold")
    ax.set_xlabel("Broj instanci za obuku")
    ax.set_ylabel("RMSE" if "rmse" in scoring or "root" in scoring else scoring)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


def plot_regression_metrics_bar(metrics: Dict[str, Dict[str, float]],
                                save_path: Optional[str] = None) -> None:
    """Crta uporednu bar tabelu RMSE i MAE-a za sve modele."""
    names = list(metrics.keys())
    rmse_vals = [metrics[n]["RMSE"] for n in names]
    mae_vals = [metrics[n]["MAE"] for n in names]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(names)), 4))
    ax.bar(x - width/2, rmse_vals, width, label="RMSE", color="steelblue")
    ax.bar(x + width/2, mae_vals,  width, label="MAE",  color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Greška (€)")
    ax.set_title("Performanse regresionih modela (manje je bolje)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for i, (r, m) in enumerate(zip(rmse_vals, mae_vals)):
        ax.text(i - width/2, r + max(rmse_vals)*0.01, f"{r:.0f}",
                ha="center", fontsize=8)
        ax.text(i + width/2, m + max(rmse_vals)*0.01, f"{m:.0f}",
                ha="center", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


# ---------------- PERFORMANSE KLASIFIKACIONIH MODELA ---------------- #
def plot_confusion_matrices(y_true, predictions: Dict[str, np.ndarray],
                            class_names: List[str],
                            save_path: Optional[str] = None) -> None:
    """Crta matricu konfuzije za svaki model na zajedničkoj figuri."""
    n = len(predictions)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(ax=axes[i], cmap="Blues", colorbar=False,
                  values_format="d")
        axes[i].set_title(name, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Predviđeno")
        axes[i].set_ylabel("Stvarno")
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


def plot_classification_metrics_bar(metrics: Dict[str, Dict[str, float]],
                                    save_path: Optional[str] = None) -> None:
    """Uporedna bar tabela ACC / PREC / REC / F1."""
    names = list(metrics.keys())
    keys = ["accuracy", "precision", "recall", "f1"]
    values = {k: [metrics[n][k] for n in names] for k in keys}

    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(names)), 4))
    for i, k in enumerate(keys):
        ax.bar(x + (i - 1.5) * width, values[k], width, label=k.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Vrijednost")
    ax.set_ylim(0, 1.05)
    ax.set_title("Performanse klasifikacionih modela (više je bolje)",
                 fontweight="bold")
    ax.legend(ncol=4)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


def plot_pr_curves_multiclass(y_true, probabilities: Dict[str, np.ndarray],
                              n_classes: int,
                              save_path: Optional[str] = None) -> None:
    """Crta Precision-Recall krive (mikro-prosjek) za sve klasifikatore."""
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, y_score in probabilities.items():
        if y_score is None:
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin.ravel(),
                                                      y_score.ravel())
        ax.plot(recall, precision, label=name, alpha=0.85)
    ax.set_xlabel("Opoziv (recall)")
    ax.set_ylabel("Preciznost (precision)")
    ax.set_title("PR kriva (mikro-prosjek) – klasifikacija",
                 fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


def plot_roc_curves_multiclass(y_true, probabilities: Dict[str, np.ndarray],
                               n_classes: int,
                               save_path: Optional[str] = None) -> None:
    """Crta ROC krive (mikro-prosjek) za sve klasifikatore."""
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, y_score in probabilities.items():
        if y_score is None:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})", alpha=0.85)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC kriva (mikro-prosjek) – klasifikacija",
                 fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()
