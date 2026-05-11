# 📉 Churn Prediction — Telecom

Pipeline ML complet pour prédire le churn client dans le secteur des télécommunications.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-ff4b4b?logo=streamlit)

## Aperçu

| Métrique | Logistic Regression | Random Forest | **XGBoost** |
|----------|--------------------:|--------------:|------------:|
| Accuracy | 0.60 | 0.70 | **0.65** |
| F1 Score | 0.46 | 0.42 | **0.46** |
| ROC AUC  | 0.72 | 0.73 | **0.73** |

## Fonctionnalités

- **EDA interactive** : répartition du churn, analyse par contrat, ancienneté, charges
- **Comparaison de 3 modèles** : LogisticRegression, RandomForest, XGBoost
- **Gestion du déséquilibre** : SMOTE (oversampling de la classe minoritaire)
- **Prédiction live** : formulaire client → probabilité de churn + jauge
- **Feature Importance** : top 10 variables les plus prédictives

## Stack technique

```
Python · Pandas · scikit-learn · XGBoost · imbalanced-learn · Streamlit · Plotly
```

## Lancer le projet

```bash
git clone https://github.com/davidamz/churn-prediction
cd churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Structure

```
churn-prediction/
├── data/
│   └── telco_churn.csv       # Dataset Telco (7 043 clients)
├── models/
│   ├── xgboost_model.pkl     # Modèle sauvegardé
│   ├── scaler.pkl            # StandardScaler
│   └── metrics.json          # Métriques comparées
├── src/
│   ├── preprocess.py         # Nettoyage + encodage + split
│   └── train.py              # Entraînement 3 modèles
├── app.py                    # Application Streamlit
└── requirements.txt
```

## Auteur

**David Amouzou**  
[linkedin.com/in/davidamouzou](https://linkedin.com/in/davidamouzou) · [davidamz.com](https://davidamz.com)
