# ===========================
# model_training.py
# Script pour entraîner le modèle de prédiction de dépression
# ===========================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# ---------------------------
# 1️⃣ Charger les données
# ---------------------------
print("📂 Chargement des données...")
data = pd.read_csv("data/mental_health_lifestyle.csv")
print(f"✅ Dataset chargé : {data.shape[0]} lignes, {data.shape[1]} colonnes\n")

# ---------------------------
# 2️⃣ Exploration rapide
# ---------------------------
print("📊 Aperçu des données :")
print(data.head())
print("\n📈 Informations sur le dataset :")
print(data.info())
print("\n🔍 Valeurs manquantes :")
print(data.isnull().sum())

# Supprimer les valeurs manquantes
data = data.dropna()
print(f"\n✅ Après nettoyage : {data.shape[0]} lignes\n")

# ---------------------------
# 3️⃣ Préparation des données
# ---------------------------
print("⚙️ Préparation des features et target...")

# Identifier la colonne cible (ajuster selon votre dataset)
target_col = 'Depression'  # ou 'depression', 'depressed', etc.

# Séparer features et target
X = data.drop(columns=[target_col])
y = data[target_col]

# Encoder les variables catégorielles si nécessaire
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"🔤 Encodage des variables catégorielles : {list(categorical_cols)}")
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"✅ Features : {X.shape[1]} colonnes")
print(f"✅ Target distribution :\n{y.value_counts()}\n")

# ---------------------------
# 4️⃣ Division train/test
# ---------------------------
print("🔀 Division train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train set : {X_train.shape[0]} échantillons")
print(f"✅ Test set : {X_test.shape[0]} échantillons\n")

# ---------------------------
# 5️⃣ Normalisation
# ---------------------------
print("📏 Normalisation des données...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Normalisation terminée\n")

# ---------------------------
# 6️⃣ Entraînement du modèle
# ---------------------------
print("🤖 Entraînement du modèle Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
print("✅ Modèle entraîné\n")

# ---------------------------
# 7️⃣ Validation croisée
# ---------------------------
print("🔄 Validation croisée (5-fold)...")
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"✅ CV Scores : {cv_scores}")
print(f"✅ Moyenne : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")

# ---------------------------
# 8️⃣ Évaluation sur le test set
# ---------------------------
print("📊 Évaluation sur le test set...")
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Accuracy : {accuracy:.4f}")
print("\n📋 Rapport de classification :")
print(classification_report(y_test, y_pred))
print("\n🔢 Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# 9️⃣ Importance des features
# ---------------------------
print("\n📌 Top 10 features les plus importantes :")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10))

# ---------------------------
# 🔟 Optimisation des hyperparamètres (optionnel)
# ---------------------------
print("\n🔧 Optimisation des hyperparamètres (GridSearch)...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)
print(f"\n✅ Meilleurs paramètres : {grid_search.best_params_}")
print(f"✅ Meilleur score CV : {grid_search.best_score_:.4f}")

# Utiliser le meilleur modèle
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"✅ Accuracy avec meilleurs paramètres : {accuracy_best:.4f}\n")

# ---------------------------
# 1️⃣1️⃣ Sauvegarder le modèle et le scaler
# ---------------------------
print("💾 Sauvegarde du modèle et du scaler...")
os.makedirs('models', exist_ok=True)

joblib.dump(best_model, 'models/depression_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Sauvegarder aussi les noms des features pour l'application
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

print("✅ Modèle sauvegardé : models/depression_model.pkl")
print("✅ Scaler sauvegardé : models/scaler.pkl")
print("✅ Feature names sauvegardés : models/feature_names.pkl")

print("\n🎉 Entraînement terminé avec succès !")