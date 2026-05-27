"""Učitavanje, čišćenje i priprema retail sales dataset-a za ML pipeline."""

from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ----------------- Konstante o skupu podataka ------------------ #
TARGET_COL = "sales_amount"

# Identifikacijski stupci - bez prediktivne vrijednosti
ID_COLS = ["transaction_id", "customer_id", "product_id",
           "product_name", "transaction_date"]

# Kolone koje direktno određuju target (formula: sales_amount = qty * unit_price * (1 - disc/100))
# unit_price izbacujemo da regresija ne bude trivijalna.
LEAKING_COLS = ["unit_price"]

# Kategoričke karakteristike koje ulaze u model
CATEGORICAL_COLS = [
    "customer_gender", "customer_age_group", "customer_segment",
    "category", "brand", "payment_method", "sales_channel", "region",
]

# Numeričke karakteristike koje ulaze u model
NUMERICAL_COLS = ["quantity", "discount_pct"]


def load_dataset(csv_path: str, sample_size: int = 10_000,
                 random_state: int = 14) -> pd.DataFrame:
    """Učitava CSV i opciono stratifikovano uzorkuje sample_size redova
    po quartil-binovima cilja kako bi se ubrzala obuka modela.
    """
    df = pd.read_csv(csv_path)

    if sample_size is not None and sample_size < len(df):
        # Stratifikovani sub-sample po kvartilima target-a
        bins = pd.qcut(df[TARGET_COL], q=4, labels=False)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size,
                                     random_state=random_state)
        idx, _ = next(sss.split(df, bins))
        df = df.iloc[idx].reset_index(drop=True)

    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Izbacuje identifikatore i leaking kolone (unit_price)."""
    to_drop = [c for c in ID_COLS + LEAKING_COLS if c in df.columns]
    return df.drop(columns=to_drop)


def add_synthetic_missing(df: pd.DataFrame, columns: List[str],
                          fraction: float = 0.03,
                          random_state: int = 14) -> pd.DataFrame:
    """Sintetski ubacuje NaN u dio vrijednosti odabranih kolona kako
    bismo demonstrirali rad sa nedostajućim vrijednostima (originalni
    dataset nema NaN). Vraća novi DataFrame.
    """
    rng = np.random.default_rng(random_state)
    df_out = df.copy()
    for col in columns:
        if col not in df_out.columns:
            continue
        mask = rng.random(len(df_out)) < fraction
        df_out.loc[mask, col] = np.nan
    return df_out


def stratified_split(df: pd.DataFrame, test_size: float = 0.2,
                     random_state: int = 14) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratifikovana podjela skupa na trening i test, sa kvartilima
    sales_amount-a kao slojem - bilanse target distribuciju u oba seta.
    """
    bins = pd.qcut(df[TARGET_COL], q=4, labels=False)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=random_state)
    train_idx, test_idx = next(sss.split(df, bins))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def split_features_target(df: pd.DataFrame
                          ) -> Tuple[pd.DataFrame, pd.Series]:
    """Razdvaja prediktore (X) i ciljanu vrijednost (y)."""
    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL])
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Pravi sklearn ColumnTransformer koji:
       - rješava nedostajuće vrijednosti (median za numeričke,
         najčešća vrijednost za kategoričke),
       - skalira numeričke karakteristike (StandardScaler),
       - one-hot kodira kategoričke karakteristike.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERICAL_COLS),
        ("cat", categorical_pipe, CATEGORICAL_COLS),
    ])


def make_classification_target(y_continuous: pd.Series,
                               n_classes: int = 4) -> Tuple[pd.Series, list]:
    """Generiše višeklasnu kategoričku etiketu iz numeričkog
    sales_amount-a koristeći kvantilske granice.
    Vraća (Series klasa, lista naziva klasa) - klase su 0..n-1.
    """
    labels = ["Niska", "Niža-srednja", "Viša-srednja", "Visoka"][:n_classes]
    y_class, edges = pd.qcut(y_continuous, q=n_classes,
                             labels=range(n_classes), retbins=True)
    return y_class.astype(int), labels, edges
