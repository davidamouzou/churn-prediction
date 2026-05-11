"""
app.py — Churn Prediction Dashboard
Streamlit app : EDA + comparaison modèles + prédiction live
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent / "src"))
from preprocess import load_and_clean, encode_features

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "telco_churn.csv"
MODEL_DIR = Path(__file__).parent / "models"

COLORS = {
    "primary": "#3941E3",
    "yes": "#ef4444",
    "no": "#22c55e",
    "bg": "#0f0f1a",
    "card": "#1a1a2a",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0a0f; }
    .stApp { background-color: #0a0a0f; color: #e8e8f0; }
    .metric-card {
        background: #1a1a2a;
        border: 1px solid #2a2a3a;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: 800; color: #3941E3; }
    .metric-lbl { font-size: 0.75rem; color: #666; letter-spacing: 2px; text-transform: uppercase; }
    h1, h2, h3 { color: #e8e8f0 !important; }
    .stTabs [data-baseweb="tab"] { color: #888; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #3941E3; border-bottom: 2px solid #3941E3; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_or_train():
    model_path = MODEL_DIR / "xgboost_model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(MODEL_DIR / "feature_names.json") as f:
            features = json.load(f)
        with open(MODEL_DIR / "metrics.json") as f:
            metrics = json.load(f)
        return model, scaler, features, metrics
    else:
        from train import train_all

        with st.spinner("Entraînement des modèles en cours…"):
            metrics, features = train_all()
        return load_or_train()


def plotly_dark():
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e8e8f0",
        xaxis=dict(gridcolor="#1e1e2e", color="#888"),
        yaxis=dict(gridcolor="#1e1e2e", color="#888"),
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📉 Churn Predictor")
    st.markdown("**Dataset** : Telco Customer Churn")
    st.markdown("**Modèles** : LogReg · RandomForest · XGBoost")
    st.markdown("---")
    uploaded = st.file_uploader("Uploader un CSV custom", type="csv")
    st.markdown("---")
    st.markdown(
        "**David Amouzou** · [GitHub](https://github.com) · [LinkedIn](https://linkedin.com/in/davidamouzou)"
    )

# ── Load ──────────────────────────────────────────────────────────────────────
df_raw = load_data() if not uploaded else pd.read_csv(uploaded)
model, scaler, feature_names, all_metrics = load_or_train()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Churn Prediction — Telecom")
st.markdown(
    "Pipeline ML complet · EDA · 3 modèles comparés · Prédiction en temps réel"
)
st.markdown("---")

# ── KPIs ──────────────────────────────────────────────────────────────────────
churn_rate = (df_raw["Churn"] == "Yes").mean()
col1, col2, col3, col4 = st.columns(4)
kpis = [
    (f"{len(df_raw):,}", "Clients"),
    (f"{churn_rate:.1%}", "Taux de churn"),
    (f"{df_raw['MonthlyCharges'].mean():.0f} €", "Charge mensuelle moy."),
    (f"{df_raw['tenure'].mean():.0f} mois", "Ancienneté moyenne"),
]
for col, (val, lbl) in zip([col1, col2, col3, col4], kpis):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{val}</div>'
            f'<div class="metric-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Analyse Exploratoire", "📊 Comparaison Modèles", "🎯 Prédiction Live"])

# ────────────────────────────────────────────────────
# TAB 1 — EDA
# ────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Répartition du Churn")
        counts = df_raw["Churn"].value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index,
            color_discrete_map={"Yes": COLORS["yes"], "No": COLORS["no"]},
            hole=0.55,
        )
        fig.update_layout(**plotly_dark(), showlegend=True, height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Churn par type de contrat")
        ct = df_raw.groupby(["Contract", "Churn"]).size().reset_index(name="count")
        fig = px.bar(
            ct, x="Contract", y="count", color="Churn",
            color_discrete_map={"Yes": COLORS["yes"], "No": COLORS["no"]},
            barmode="group",
        )
        fig.update_layout(**plotly_dark(), height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Ancienneté vs Churn")
        fig = px.histogram(
            df_raw, x="tenure", color="Churn",
            color_discrete_map={"Yes": COLORS["yes"], "No": COLORS["no"]},
            nbins=30, barmode="overlay", opacity=0.7,
        )
        fig.update_layout(**plotly_dark(), height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Charges mensuelles vs Churn")
        fig = px.box(
            df_raw, x="Churn", y="MonthlyCharges", color="Churn",
            color_discrete_map={"Yes": COLORS["yes"], "No": COLORS["no"]},
        )
        fig.update_layout(**plotly_dark(), height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn par service Internet")
    ci = df_raw.groupby(["InternetService", "Churn"]).size().reset_index(name="count")
    fig = px.bar(
        ci, x="InternetService", y="count", color="Churn",
        color_discrete_map={"Yes": COLORS["yes"], "No": COLORS["no"]},
        barmode="group",
    )
    fig.update_layout(**plotly_dark(), height=280, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────
# TAB 2 — Comparaison modèles
# ────────────────────────────────────────────────────
with tab2:
    st.subheader("Métriques comparées — 3 modèles")

    model_names = list(all_metrics.keys())
    metrics_keys = ["accuracy", "f1", "roc_auc", "precision", "recall"]
    metrics_labels = ["Accuracy", "F1 Score", "ROC AUC", "Precision", "Recall"]

    # Tableau
    rows = []
    for name in model_names:
        row = {"Modèle": name}
        for k, l in zip(metrics_keys, metrics_labels):
            row[l] = all_metrics[name][k]
        rows.append(row)
    df_metrics = pd.DataFrame(rows).set_index("Modèle")
    st.dataframe(
        df_metrics.style.highlight_max(color="#3941E320").format("{:.4f}"),
        use_container_width=True,
    )

    # Bar chart comparatif
    fig = go.Figure()
    colors_bar = [COLORS["primary"], "#f59e0b", "#22c55e"]
    for i, name in enumerate(model_names):
        fig.add_trace(go.Bar(
            name=name,
            x=metrics_labels,
            y=[all_metrics[name][k] for k in metrics_keys],
            marker_color=colors_bar[i],
            opacity=0.85,
        ))
    layout = plotly_dark()
    layout["yaxis"]["range"] = [0, 1]
    fig.update_layout(
        **layout,
        barmode="group",
        height=350,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance XGBoost
    st.subheader("Feature Importance — XGBoost (top 10)")
    fi = all_metrics["XGBoost"]["feature_importance"]
    fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).head(10)
    fig = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=["#1a1a2a", COLORS["primary"]],
    )
    fig.update_layout(**plotly_dark(), height=350, margin=dict(t=20, b=20))
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Matrice de confusion XGBoost
    st.subheader("Matrice de confusion — XGBoost")
    cm = np.array(all_metrics["XGBoost"]["confusion_matrix"])
    fig = px.imshow(
        cm,
        labels=dict(x="Prédit", y="Réel", color="Count"),
        x=["No Churn", "Churn"],
        y=["No Churn", "Churn"],
        color_continuous_scale=["#0a0a0f", COLORS["primary"]],
        text_auto=True,
    )
    fig.update_layout(**plotly_dark(), height=320, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────
# TAB 3 — Prédiction Live
# ────────────────────────────────────────────────────
with tab3:
    st.subheader("Prédire le risque de churn d'un client")
    st.markdown("Renseignez les informations du client :")

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Genre", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partenaire", ["Yes", "No"])
        dependents = st.selectbox("Dépendants", ["Yes", "No"])
        tenure = st.slider("Ancienneté (mois)", 0, 72, 12)

    with c2:
        phone_service = st.selectbox("Service téléphonique", ["Yes", "No"])
        multiple_lines = st.selectbox("Lignes multiples", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Service Internet", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Sécurité en ligne", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Support technique", ["Yes", "No", "No internet service"])

    with c3:
        contract = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Facturation dématérialisée", ["Yes", "No"])
        payment = st.selectbox("Méthode de paiement", [
            "Electronic check", "Mailed check", "Bank transfer", "Credit card"
        ])
        monthly = st.slider("Charges mensuelles (€)", 18.0, 119.0, 65.0)
        total = monthly * tenure

    if st.button("🔮 Prédire le churn", use_container_width=True):
        from sklearn.preprocessing import LabelEncoder

        # Construire le dataframe client
        client = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet,
            "OnlineSecurity": online_security, "TechSupport": tech_support,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
        }])

        # Encoder les colonnes catégorielles
        cat_cols = client.select_dtypes(include="object").columns
        le = LabelEncoder()
        # Fit sur les valeurs possibles connues pour encoder proprement
        for col in cat_cols:
            df_enc = load_and_clean(str(DATA_PATH))
            df_enc_copy = df_enc.copy()
            le.fit(df_enc_copy[col].astype(str) if col in df_enc_copy.columns else [client[col].iloc[0]])
            try:
                client[col] = le.transform(client[col].astype(str))
            except ValueError:
                client[col] = 0

        # Réordonner selon feature_names
        client = client.reindex(columns=feature_names, fill_value=0)
        client_scaled = scaler.transform(client)

        proba = model.predict_proba(client_scaled)[0][1]
        pred = "Churn" if proba > 0.5 else "Pas de churn"

        # Résultat
        st.markdown("---")
        risk_color = COLORS["yes"] if proba > 0.5 else COLORS["no"]
        st.markdown(
            f'<div class="metric-card" style="border-color: {risk_color};">'
            f'<div class="metric-val" style="color: {risk_color};">{proba:.1%}</div>'
            f'<div class="metric-lbl">Probabilité de churn</div>'
            f'<div style="margin-top:8px; font-size:1.1rem; font-weight:700; color:{risk_color};">{pred}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Jauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"color": risk_color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#444"},
                "bar": {"color": risk_color},
                "bgcolor": "#1a1a2a",
                "steps": [
                    {"range": [0, 30], "color": "#0d2010"},
                    {"range": [30, 60], "color": "#1a1a10"},
                    {"range": [60, 100], "color": "#200d0d"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": 50},
            },
        ))
        fig.update_layout(
            **plotly_dark(),
            height=280,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Facteurs de risque
        if proba > 0.5:
            st.warning(
                "**Facteurs de risque détectés :**\n"
                + ("- Contrat mensuel (risque élevé)\n" if contract == "Month-to-month" else "")
                + ("- Fibre optique souvent associée au churn\n" if internet == "Fiber optic" else "")
                + ("- Faible ancienneté\n" if tenure < 12 else "")
                + ("- Charges élevées\n" if monthly > 80 else "")
            )
        else:
            st.success("Client à faible risque de churn. Fidélisation en bonne voie.")
