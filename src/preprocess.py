"""
preprocess.py
Nettoyage et transformation du dataset Telco Churn.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Supprimer l'ID client (non prédictif)
    df.drop(columns=["customerID"], inplace=True)

    # TotalCharges peut contenir des espaces → convertir en float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Imputer les NaN de TotalCharges par tenure * MonthlyCharges
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = (
        df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]
    )

    # Encoder la cible
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def encode_features(df: pd.DataFrame):
    """
    Encode les variables catégorielles avec LabelEncoder.
    Retourne le DataFrame encodé + la liste des colonnes catégorielles.
    """
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    return df, cat_cols


def split_and_scale(df: pd.DataFrame, target: str = "Churn", test_size: float = 0.2):
    """
    Split train/test + normalisation StandardScaler.
    Retourne X_train, X_test, y_train, y_test, scaler, feature_names.
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


def get_preprocessed_data(filepath: str):
    """Pipeline complet : load → clean → encode → split → scale."""
    df = load_and_clean(filepath)
    df, _ = encode_features(df)
    return split_and_scale(df)
