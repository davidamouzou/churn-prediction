"""
train.py
Entraînement de 3 modèles : LogisticRegression, RandomForest, XGBoost.
Comparaison des métriques + sauvegarde du meilleur modèle.
"""

import json
import pickle
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)
from imblearn.over_sampling import SMOTE

import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import get_preprocessed_data


DATA_PATH = Path(__file__).parent.parent / "data" / "telco_churn.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def get_models():
    from xgboost import XGBClassifier

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42,
            class_weight="balanced", n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=3,  # compense le déséquilibre
            random_state=42, eval_metric="logloss",
            verbosity=0,
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def get_feature_importance(model, feature_names: list, model_name: str) -> dict:
    """Extrait l'importance des features selon le type de modèle."""
    if model_name == "Logistic Regression":
        importances = np.abs(model.coef_[0])
    elif model_name == "Random Forest":
        importances = model.feature_importances_
    else:  # XGBoost
        importances = model.feature_importances_

    importance_dict = dict(zip(feature_names, importances.tolist()))
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def train_all(data_path: str = None) -> dict:
    """
    Entraîne les 3 modèles, compare les métriques,
    sauvegarde le meilleur modèle (XGBoost).
    Retourne un dict avec toutes les métriques.
    """
    path = data_path or DATA_PATH
    print(f"Chargement des données : {path}")

    X_train, X_test, y_train, y_test, scaler, feature_names = get_preprocessed_data(str(path))

    # SMOTE sur le train uniquement
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Après SMOTE — train : {X_train_res.shape[0]} samples")

    models = get_models()
    results = {}

    print("\n=== Entraînement des modèles ===")
    for name, model in models.items():
        print(f"  → {name}...", end=" ")
        model.fit(X_train_res, y_train_res)
        metrics = evaluate(model, X_test, y_test)
        metrics["feature_importance"] = get_feature_importance(model, feature_names, name)
        results[name] = metrics
        print(f"F1={metrics['f1']}  AUC={metrics['roc_auc']}")

    # Sauvegarder le meilleur modèle (XGBoost) + scaler
    best_model = models["XGBoost"]
    with open(MODEL_DIR / "xgboost_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nModèle sauvegardé dans {MODEL_DIR}/")
    return results, feature_names


if __name__ == "__main__":
    results, features = train_all()
    print("\n=== Résultats finaux ===")
    for name, m in results.items():
        print(f"{name:25s} | Accuracy={m['accuracy']} | F1={m['f1']} | AUC={m['roc_auc']}")
