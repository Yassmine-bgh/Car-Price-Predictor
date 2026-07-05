


import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prédiction Prix Voiture", page_icon="🚗", layout="centered")

# ---------------------------------------------------------------
# Chargement du modèle et des artefacts (mis en cache pour la perf)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("label_encoder.pkl")
        with open("metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return model, scaler, le, metadata
    except FileNotFoundError:
        return None, None, None, None

model, scaler, le, metadata = load_artifacts()

st.title("🚗 Estimateur de prix de voiture d'occasion")
st.caption("Modèle : RandomForestRegressor entraîné sur des données Carvana")

if model is None:
    st.error(
        "Fichiers du modèle introuvables (model.pkl, scaler.pkl, "
        "label_encoder.pkl, metadata.json).\n\n"
        "Lance d'abord `python train_model.py chemin/vers/carvana.csv` "
        "dans ce même dossier, puis relance l'application."
    )
    st.stop()

# ---------------------------------------------------------------
# Barre latérale : infos sur le modèle
# ---------------------------------------------------------------
with st.sidebar:
    st.header("À propos du modèle")
    st.metric("Erreur absolue moyenne (MAE)", f"${metadata['mae']:.2f}")
    st.metric("Score R²", f"{metadata['r2']:.3f}")
    st.write(f"Années couvertes : {metadata['year_min']} – {metadata['year_max']}")
    st.write(f"Kilométrage couvert : {metadata['miles_min']:,} – {metadata['miles_max']:,} miles")
    st.write(f"Nombre de modèles de voiture connus : {len(metadata['car_names'])}")

# ---------------------------------------------------------------
# Formulaire de saisie
# ---------------------------------------------------------------
st.subheader("Renseigne les informations de la voiture")

col1, col2 = st.columns(2)

with col1:
    name = st.selectbox(
        "Modèle de voiture",
        options=metadata["car_names"],
        help="Liste des modèles vus pendant l'entraînement.",
    )
    year = st.number_input(
        "Année",
        min_value=metadata["year_min"],
        max_value=metadata["year_max"] + 1,
        value=min(2020, metadata["year_max"]),
        step=1,
    )

with col2:
    miles = st.number_input(
        "Kilométrage (miles)",
        min_value=0,
        max_value=metadata["miles_max"] + 50000,
        value=min(50000, metadata["miles_max"]),
        step=1000,
    )

predict_btn = st.button("💰 Estimer le prix", type="primary", use_container_width=True)

# ---------------------------------------------------------------
# Prédiction
# ---------------------------------------------------------------
if predict_btn:
    try:
        name_encoded = le.transform([name])[0]
    except ValueError:
        st.error("Ce modèle de voiture n'a pas été vu pendant l'entraînement.")
        st.stop()

    miles_scaled = scaler.transform([[miles]])[0][0]

    X_new = pd.DataFrame(
        [[year, miles_scaled, name_encoded]],
        columns=["Year", "Miles_scaled", "Name_Encoded"],
    )

    predicted_price = model.predict(X_new)[0]

    st.success(f"### Prix estimé : ${predicted_price:,.0f}")

    # Petite marge d'erreur indicative basée sur le MAE du modèle
    low = predicted_price - metadata["mae"]
    high = predicted_price + metadata["mae"]
    st.caption(f"Fourchette approximative (± MAE) : ${low:,.0f} – ${high:,.0f}")