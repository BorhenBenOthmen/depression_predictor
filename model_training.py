"""
Mental Health Classification - Script d'entraînement du modèle (VERSION BINAIRE)
Prédiction BINAIRE du risque de suicide: 0=Pas de risque, 1=Risque
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')


def load_data(filepath):
    """Charger les données depuis le fichier CSV"""
    print("=" * 60)
    print("1. CHARGEMENT DES DONNÉES")
    print("=" * 60)

    df = pd.read_csv(filepath)
    print(f"✓ Dataset chargé avec succès!")
    print(f"  - Dimensions: {df.shape}")
    print(f"  - Nombre de lignes: {df.shape[0]}")
    print(f"  - Nombre de colonnes: {df.shape[1]}")

    # Vérifier les valeurs manquantes
    missing_count = df.isnull().sum().sum()
    print(f"  - Valeurs manquantes: {missing_count}")

    return df


def preprocess_data(df):
    """Prétraitement des données"""
    print("\n" + "=" * 60)
    print("2. PRÉTRAITEMENT DES DONNÉES")
    print("=" * 60)

    df_processed = df.copy()

    # Vérifier les valeurs manquantes
    missing_values = df_processed.isnull().sum().sum()
    print(f"✓ Valeurs manquantes: {missing_values}")

    if missing_values > 0:
        # Remplir les valeurs manquantes numériques avec la médiane
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)

        # Remplir les valeurs manquantes catégorielles avec le mode
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

        print(f"✓ Valeurs manquantes traitées")

    # Encoder les variables catégorielles
    label_encoders = {}
    categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()

    print(f"\n✓ Encodage des variables catégorielles:")
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"  - {col}: {len(le.classes_)} classes")

    return df_processed, label_encoders


def create_feature_engineering(df):
    """Créer des features composites pour améliorer les prédictions"""
    print("\n" + "=" * 60)
    print("3. FEATURE ENGINEERING (NOUVELLES VARIABLES)")
    print("=" * 60)

    df = df.copy()

    # 1. Stress Composite = Depression_Score * Nervous_Level
    df['Stress_Composite'] = df['Depression_Score'] * df['Nervous_Level'] / 100

    # 2. Fatigue Index = Sleep_Hours * Low_Energy
    df['Fatigue_Index'] = df['Sleep_Hours'] * df['Low_Energy']

    # 3. Social Depression Impact = SocialMedia_Hours * Depression_Score
    df['Social_Depression_Impact'] = df['SocialMedia_Hours'] * df['Depression_Score'] / 100

    # 4. Danger Score = Self_Harm + Worsening_Depression + Symptoms
    df['Danger_Score'] = df['Self_Harm'] + df['Worsening_Depression'] + (df['Symptoms'] / 10)

    # 5. Mental Health Vulnerability = Low_SelfEsteem * Low_Energy * Nervous_Level
    df['Mental_Vulnerability'] = df['Low_SelfEsteem'] * df['Low_Energy'] * df['Nervous_Level']

    # 6. Support Index = Mental_Health_Support + Coping_Methods
    df['Support_Index'] = df['Mental_Health_Support'] + df['Coping_Methods']

    # 7. Sleep Quality = abs(Sleep_Hours - 7.5) (optimal 7-8 heures)
    df['Sleep_Quality'] = abs(df['Sleep_Hours'] - 7.5)

    print(f"✓ 7 nouvelles features créées:")
    print(f"  - Stress_Composite")
    print(f"  - Fatigue_Index")
    print(f"  - Social_Depression_Impact")
    print(f"  - Danger_Score")
    print(f"  - Mental_Vulnerability")
    print(f"  - Support_Index")
    print(f"  - Sleep_Quality")

    return df


def prepare_features(df_processed):
    """Préparer les features et la target BINAIRE (0 ou 1 tentative = 0 | >=2 tentatives = 1)"""
    print("\n" + "=" * 60)
    print("4. PRÉPARATION DES FEATURES ET TARGET BINAIRE (0-1 tentative vs 2+ tentatives)")
    print("=" * 60)

    # === NOUVELLE TARGET CLINIQUE ===
    # Classe 0 : 0 ou 1 tentative → "risque modéré ou passé"
    # Classe 1 : >= 2 tentatives → "risque chronique / récurrent / très élevé"
    y = (df_processed['Suicide_Attempts'] >= 2).astype(int)

    print(f"✓ Nouvelle variable cible binaire (cliniquement pertinente):")
    print(f"  - Classe 0 (0 ou 1 tentative): {(y == 0).sum()} cas "
          f"({(y == 0).sum()/len(y)*100:.1f}%)")
    print(f"    → Dont 0 tentative: {(df_processed['Suicide_Attempts'] == 0).sum()}")
    print(f"    → Dont 1 seule tentative: {(df_processed['Suicide_Attempts'] == 1).sum()}")
    print(f"  - Classe 1 (≥2 tentatives - risque chronique): {(y == 1).sum()} cas "
          f"({(y == 1).sum()/len(y)*100:.1f}%)")

    # Features = tout sauf la colonne cible originale
    X = df_processed.drop('Suicide_Attempts', axis=1)

    print(f"\n✓ Features (X): {X.shape}")
    print(f"✓ Target (y): {y.shape}")
    print(f"\nColonnes utilisées comme features ({len(X.columns)} au total):")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i}. {col}")

    return X, y


def split_and_scale_data(X, y):
    """Diviser les données et appliquer le scaling"""
    print("\n" + "=" * 60)
    print("5. DIVISION TRAIN/TEST ET NORMALISATION")
    print("=" * 60)

    # Division AVANT scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"✓ Données divisées:")
    print(f"  - Train: {X_train.shape[0]} échantillons")
    print(f"  - Test: {X_test.shape[0]} échantillons")
    print(f"  - Train distribution: {(y_train == 0).sum()} vs {(y_train == 1).sum()}")
    print(f"  - Test distribution: {(y_test == 0).sum()} vs {(y_test == 1).sum()}")

    # Normalisation
    print(f"\n✓ Normalisation avec StandardScaler:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  - Scaler ajusté sur les données d'entraînement")
    print(f"  - Transformation appliquée à train et test")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train, y_train):
    """Entraîner plusieurs modèles avec class_weight pour gérer le déséquilibre"""
    print("\n" + "=" * 60)
    print("6. ENTRAÎNEMENT DES MODÈLES (AVEC class_weight='balanced')")
    print("=" * 60)

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Gère le déséquilibre
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    }

    trained_models = {}
    cv_scores = {}

    for name, model in models.items():
        print(f"\n→ Entraînement: {name}")

        # Entraîner le modèle
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
        cv_scores[name] = scores

        print(f"  ✓ Score F1 validation croisée: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return trained_models, cv_scores


def evaluate_models(trained_models, X_test, y_test):
    """Évaluer tous les modèles sur le test set"""
    print("\n" + "=" * 60)
    print("7. ÉVALUATION DES MODÈLES")
    print("=" * 60)

    results = {}

    for name, model in trained_models.items():
        print(f"\n→ Évaluation: {name}")

        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred
        }

        print(f"  ✓ Accuracy:  {accuracy:.4f}")
        print(f"  ✓ Precision: {precision:.4f}")
        print(f"  ✓ Recall:    {recall:.4f}")
        print(f"  ✓ F1-Score:  {f1:.4f}")
        print(f"  ✓ ROC-AUC:   {roc_auc:.4f}")

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Matrice de confusion:")
        print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")

        # Rapport de classification
        print(f"\n  Rapport de classification:")
        print(classification_report(y_test, y_pred,
                                   target_names=['Pas de risque', 'Risque'],
                                   zero_division=0))

    return results


def select_best_model(results, trained_models):
    """Sélectionner le meilleur modèle basé sur F1-score"""
    print("\n" + "=" * 60)
    print("8. SÉLECTION DU MEILLEUR MODÈLE")
    print("=" * 60)

    # Utiliser F1-score pour choisir le meilleur modèle
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]

    print(f"\n Meilleur modèle: {best_model_name}")
    print(f"   Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"   F1-Score:  {best_metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {best_metrics['roc_auc']:.4f}")

    print(f"\n Classement de tous les modèles (par F1-Score):")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    for i, (name, metrics) in enumerate(sorted_results, 1):
        print(f"   {i}. {name}")
        print(f"      F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    return best_model_name, best_model


def save_model_and_artifacts(model, scaler, label_encoders, model_name):
    """Sauvegarder le modèle, le scaler et les encoders"""
    print("\n" + "=" * 60)
    print("9. SAUVEGARDE DU MODÈLE ET DES ARTEFACTS")
    print("=" * 60)

    # Sauvegarder le modèle
    model_path = 'models/depression_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Modèle sauvegardé: {model_path}")
    print(f"  Type: {model_name}")

    # Sauvegarder le scaler
    scaler_path = 'models/scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler sauvegardé: {scaler_path}")

    # Sauvegarder les label encoders
    encoders_path = 'models/label_encoders.pkl'
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"✓ Label encoders sauvegardés: {encoders_path}")


def main():
    """Fonction principale"""
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION")
    print("Mental Health Classification - VERSION BINAIRE")
    print("=" * 60)

    try:
        # 1. Charger les données
        df = load_data('data/ipynb_checkpoints/Mental Health Classification.csv')

        # 2. Prétraitement
        df_processed, label_encoders = preprocess_data(df)

        # 3. Feature Engineering
        df_processed = create_feature_engineering(df_processed)

        # 4. Préparer les features et TARGET BINAIRE
        X, y = prepare_features(df_processed)

        # 5. Division et scaling
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)

        # 6. Entraîner les modèles (avec class_weight='balanced')
        trained_models, cv_scores = train_models(X_train, y_train)

        # 7. Évaluer les modèles
        results = evaluate_models(trained_models, X_test, y_test)

        # 8. Sélectionner le meilleur modèle
        best_model_name, best_model = select_best_model(results, trained_models)

        # 9. Sauvegarder
        save_model_and_artifacts(best_model, scaler, label_encoders, best_model_name)

        print("\n" + "=" * 60)
        print(" ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("=" * 60)
        print("\nLes fichiers suivants ont été créés:")
        print("  - models/depression_model.pkl")
        print("  - models/scaler.pkl")
        print("  - models/label_encoders.pkl")
        print("\nAMÉLIORATIONS APPORTÉES:")
        print("  ✓ Prédiction BINAIRE (Pas de risque vs Risque)")
        print("  ✓ 7 nouvelles features composites créées")

    except Exception as e:
        print(f"\n ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()