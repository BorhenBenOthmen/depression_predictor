# ===========================
# app.py
# Interface utilisateur pour le modèle de prédiction de dépression
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# 1️⃣ Charger le modèle, le scaler et les noms des features
# ---------------------------
st.title("🧠 Prédiction du risque de dépression")
st.write("Entrez vos habitudes de vie pour obtenir une prédiction.")

model = joblib.load("models/depression_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# ---------------------------
# 2️⃣ Créer les inputs utilisateur
# ---------------------------
st.sidebar.header("🛠 Paramètres utilisateur")

def user_input_features():
    Sleep_Duration = st.sidebar.slider("Heures de sommeil par jour", 0, 12, 7)
    Stress_Level = st.sidebar.slider("Niveau de stress (1-10)", 1, 10, 5)
    Physical_Activity = st.sidebar.slider("Heures d'activité physique par semaine", 0, 20, 3)
    Social_Media_Usage = st.sidebar.slider("Temps sur les réseaux sociaux par jour (heures)", 0, 12, 2)
    Diet_Quality = st.sidebar.slider("Qualité de l'alimentation (1=faible, 5=excellente)", 1, 5, 3)

    # Créer un dataframe
    data = {
        'Sleep Duration': Sleep_Duration,
        'Stress Level': Stress_Level,
        'Physical Activity': Physical_Activity,
        'Social Media Usage': Social_Media_Usage,
        'Diet Quality': Diet_Quality
    }

    df = pd.DataFrame(data, index=[0])
    return df

input_df = user_input_features()

# ---------------------------
# 3️⃣ Ajouter les colonnes manquantes pour matcher les features du modèle
# ---------------------------
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0  # Valeur par défaut pour les colonnes encodées

input_df = input_df[feature_names]  # Réordonner les colonnes

# ---------------------------
# 4️⃣ Normaliser les inputs
# ---------------------------
input_scaled = scaler.transform(input_df)

# ---------------------------
# 5️⃣ Prédiction
# ---------------------------
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)[0][1]  # Probabilité de dépression = classe 1

# ---------------------------
# 6️⃣ Affichage du résultat
# ---------------------------
st.subheader("🔮 Résultat de la prédiction")
if prediction == 1:
    st.error(f"⚠️ Le modèle prédit un risque de dépression.\nProbabilité estimée : {prediction_proba:.2f}")
else:
    st.success(f"✅ Le modèle ne détecte pas de risque de dépression.\nProbabilité estimée : {prediction_proba:.2f}")

st.write("\n💡 Remarque : Ce modèle est à titre informatif uniquement et ne remplace pas un avis médical.")
