"""
Mental Health Classification - Script d'entraînement du modèle
Ce script charge les données, prétraite les features avec SMOTE,
entraîne plusieurs modèles et sauvegarde le meilleur modèle avec le scaler
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
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


def prepare_features(df_processed):
    """Préparer les features et la target"""
    print("\n" + "=" * 60)
    print("3. PRÉPARATION DES FEATURES")
    print("=" * 60)

    # Utiliser 'Suicide_Attempts' comme variable cible (c'est ce qu'on veut prédire)
    X = df_processed.drop('Suicide_Attempts', axis=1)
    y = df_processed['Suicide_Attempts']

    print(f"✓ Features (X): {X.shape}")
    print(f"✓ Target (y): {y.shape}")
    print(f"\n✓ Distribution de la cible AVANT preprocessing:")
    print(y.value_counts().sort_index())
    print(f"\nColonnes utilisées comme features ({len(X.columns)} au total):")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i}. {col}")

    return X, y


def split_and_scale_data(X, y):
    """Diviser les données et appliquer le scaling"""
    print("\n" + "=" * 60)
    print("4. DIVISION TRAIN/TEST (AVANT PREPROCESSING)")
    print("=" * 60)

    # ⚠️ IMPORTANT: Division AVANT SMOTE pour éviter data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"✓ Données divisées:")
    print(f"  - Train: {X_train.shape[0]} échantillons")
    print(f"  - Test: {X_test.shape[0]} échantillons")

    # Normalisation (fit sur train, puis appliqué à test)
    print(f"\n✓ Normalisation avec StandardScaler:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  - Scaler ajusté sur les données d'entraînement")
    print(f"  - Transformation appliquée à train et test")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test


def apply_smote(X_train_scaled, y_train):
    """Appliquer SMOTE pour équilibrer les classes"""
    print("\n" + "=" * 60)
    print("5. ÉQUILIBRAGE DES CLASSES AVEC SMOTE")
    print("=" * 60)

    print(f"✓ Distribution AVANT SMOTE:")
    print(f"  {pd.Series(y_train).value_counts().sort_index()}")

    # Appliquer SMOTE
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    print(f"\n✓ Distribution APRÈS SMOTE:")
    print(f"  {pd.Series(y_train_balanced).value_counts().sort_index()}")

    print(f"\n✓ SMOTE appliqué avec succès!")
    print(f"  - Nombre d'échantillons avant: {len(y_train)}")
    print(f"  - Nombre d'échantillons après: {len(y_train_balanced)}")

    return X_train_balanced, y_train_balanced


def train_models(X_train, y_train):
    """Entraîner plusieurs modèles et retourner leurs performances"""
    print("\n" + "=" * 60)
    print("6. ENTRAÎNEMENT DES MODÈLES")
    print("=" * 60)

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True)
    }

    trained_models = {}
    cv_scores = {}

    for name, model in models.items():
        print(f"\n→ Entraînement: {name}")

        # Entraîner le modèle
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        cv_scores[name] = scores

        print(f"  ✓ Score de validation croisée: {scores.mean():.4f} (+/- {scores.std():.4f})")

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

        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }

        print(f"  ✓ Accuracy:  {accuracy:.4f}")
        print(f"  ✓ Precision: {precision:.4f}")
        print(f"  ✓ Recall:    {recall:.4f}")
        print(f"  ✓ F1-Score:  {f1:.4f}")

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Matrice de confusion:")
        print(f"  {cm}")

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

    print(f"\n🏆 Meilleur modèle: {best_model_name}")
    print(f"   Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"   F1-Score:  {best_metrics['f1']:.4f}")

    print(f"\n📊 Classement de tous les modèles (par F1-Score):")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    for i, (name, metrics) in enumerate(sorted_results, 1):
        print(f"   {i}. {name}")
        print(f"      Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

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
    print("Mental Health Classification - Version avec SMOTE")
    print("=" * 60)

    try:
        # 1. Charger les données
        df = load_data('data/ipynb_checkpoints/Mental Health Classification.csv')

        # 2. Prétraitement
        df_processed, label_encoders = preprocess_data(df)

        # 3. Préparer les features
        X, y = prepare_features(df_processed)

        # 4. Division et scaling (AVANT SMOTE)
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test = split_and_scale_data(X, y)

        # 5. SMOTE sur les données d'entraînement
        X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)

        # 6. Entraîner les modèles (sur données équilibrées)
        trained_models, cv_scores = train_models(X_train_balanced, y_train_balanced)

        # 7. Évaluer les modèles (sur données test NON équilibrées)
        results = evaluate_models(trained_models, X_test_scaled, y_test)

        # 8. Sélectionner le meilleur modèle
        best_model_name, best_model = select_best_model(results, trained_models)

        # 9. Sauvegarder
        save_model_and_artifacts(best_model, scaler, label_encoders, best_model_name)

        print("\n" + "=" * 60)
        print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("=" * 60)
        print("\nLes fichiers suivants ont été créés:")
        print("  - models/depression_model.pkl")
        print("  - models/scaler.pkl")
        print("  - models/label_encoders.pkl")
        print("\nVous pouvez maintenant utiliser l'application Flask (app.py)")

    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()