"""
Mental Health Classification - Entraînement & Optimisation des Modèles
✓ Optimisation hyperparamètres (CV=4)
✓ Matrices de confusion
✓ Tableaux récapitulatifs
✓ Comparaison des modèles
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import warnings
warnings.filterwarnings('ignore')

# Configuration Seaborn
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# ==================== SECTION 1: CHARGEMENT & PRÉTRAITEMENT ====================

def load_and_preprocess(filepath):
    """Charger et prétraiter les données"""
    print("=" * 80)
    print("1. CHARGEMENT ET PRÉTRAITEMENT")
    print("=" * 80)

    df = pd.read_csv(filepath)
    print(f"✓ Dataset chargé: {df.shape}")

    # Prétraitement
    df_processed = df.copy()

    # Encoder les variables catégorielles
    label_encoders = {}
    for col in df_processed.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

    print(f"✓ Variables catégorielles encodées")

    return df_processed, label_encoders


# ==================== SECTION 2: FEATURE ENGINEERING ====================

def create_features(df):
    """Créer des features composites"""
    print("\n" + "=" * 80)
    print("2. FEATURE ENGINEERING")
    print("=" * 80)

    df = df.copy()

    df['Stress_Composite'] = df['Depression_Score'] * df['Nervous_Level'] / 100
    df['Fatigue_Index'] = df['Sleep_Hours'] * df['Low_Energy']
    df['Social_Depression_Impact'] = df['SocialMedia_Hours'] * df['Depression_Score'] / 100
    df['Danger_Score'] = df['Self_Harm'] + df['Worsening_Depression'] + (df['Symptoms'] / 10)
    df['Mental_Vulnerability'] = df['Low_SelfEsteem'] * df['Low_Energy'] * df['Nervous_Level']
    df['Support_Index'] = df['Mental_Health_Support'] + df['Coping_Methods']
    df['Sleep_Quality'] = abs(df['Sleep_Hours'] - 7.5)

    print(f"✓ 7 nouvelles features créées")

    return df


# ==================== SECTION 3: PRÉPARATION DONNÉES ====================

def prepare_data(df_processed):
    """Préparer X et y"""
    print("\n" + "=" * 80)
    print("3. PRÉPARATION DES DONNÉES")
    print("=" * 80)

    # Target binaire: 0 ou 1 tentative vs >=2 tentatives
    y = (df_processed['Suicide_Attempts'] >= 1).astype(int)

    print(f"✓ Classe 0 (0 tentative): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
    print(f"✓ Classe 1 (≥1 tentatives): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")

    X = df_processed.drop('Suicide_Attempts', axis=1)

    # Split et scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n✓ Train: {X_train_scaled.shape}")
    print(f"✓ Test: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ==================== SECTION 4: OPTIMISATION HYPERPARAMÈTRES ====================

def optimize_models(X_train, y_train):
    """Optimisation avec GridSearchCV (CV=4)"""
    print("\n" + "=" * 80)
    print("4. OPTIMISATION DES HYPERPARAMÈTRES (CV=4)")
    print("=" * 80)

    results_opt = {}

    # 1. Random Forest
    print("\n Random Forest...")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [15, 20, 25],
        'min_samples_split': [5, 10]
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        rf_params, cv=4, scoring='f1', n_jobs=-1, verbose=0
    )
    rf_grid.fit(X_train, y_train)
    print(f"  ✓ Meilleurs paramètres: {rf_grid.best_params_}")
    print(f"  ✓ Score CV (F1): {rf_grid.best_score_:.4f}")
    results_opt['Random Forest'] = rf_grid

    # 2. Gradient Boosting
    print("\n Gradient Boosting...")
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7]
    }
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_params, cv=4, scoring='f1', n_jobs=-1, verbose=0
    )
    gb_grid.fit(X_train, y_train)
    print(f"  ✓ Meilleurs paramètres: {gb_grid.best_params_}")
    print(f"  ✓ Score CV (F1): {gb_grid.best_score_:.4f}")
    results_opt['Gradient Boosting'] = gb_grid

    return results_opt


# ==================== SECTION 5: ÉVALUATION MODÈLES ====================

def evaluate_models(best_models, X_test, y_test):
    """Évaluation complète"""
    print("\n" + "=" * 80)
    print("5. ÉVALUATION DES MODÈLES")
    print("=" * 80)

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for idx, (name, model_grid) in enumerate(best_models.items()):
        print(f"\n→ {name}")

        # Utiliser le meilleur modèle
        model = model_grid.best_estimator_

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

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
            'predictions': y_pred,
            'proba': y_pred_proba,
            'model': model
        }

        print(f"  ✓ Accuracy:  {accuracy:.4f}")
        print(f"  ✓ Precision: {precision:.4f}")
        print(f"  ✓ Recall:    {recall:.4f}")
        print(f"  ✓ F1-Score:  {f1:.4f}")
        print(f"  ✓ ROC-AUC:   {roc_auc:.4f}")

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Matrice confusion: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

        # Visualisation
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Pas risque', 'Risque'])
        disp.plot(ax=axes[idx], cmap='Blues', values_format='d')
        axes[idx].set_title(f'Matrice Confusion - {name}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/10_MODELS_CONFUSION_MATRICES.png', dpi=300, bbox_inches='tight')
    print("\n✓ Matrice sauvegardée: 10_MODELS_CONFUSION_MATRICES.png")
    plt.show()

    return results


# ==================== SECTION 6: TABLEAUX RÉCAPITULATIFS ====================

def create_summary(results):
    """Tableaux récapitulatifs"""
    print("\n" + "=" * 80)
    print("6. TABLEAUX RÉCAPITULATIFS")
    print("=" * 80)

    summary_df = pd.DataFrame({
        'Modèle': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results.keys()],
        'Precision': [results[m]['precision'] for m in results.keys()],
        'Recall': [results[m]['recall'] for m in results.keys()],
        'F1-Score': [results[m]['f1'] for m in results.keys()],
        'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
    }).round(4)

    print("\n Comparaison des performances:")
    print(summary_df.to_string(index=False))

    # Visualisation tableau
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Tableau Récapitulatif des Performances', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('visualizations/11_MODELS_PERFORMANCE_SUMMARY.png', dpi=300, bbox_inches='tight')
    print("\n✓ Tableau sauvegardé: 11_MODELS_PERFORMANCE_SUMMARY.png")
    plt.show()

    # Sauvegarder CSV
    summary_df.to_csv('results/MODELS_PERFORMANCE_COMPARISON.csv', index=False)
    print("✓ CSV sauvegardé: MODELS_PERFORMANCE_COMPARISON.csv")

    return summary_df


# ==================== SECTION 7: COURBES ROC ====================

def plot_roc_curves(results, y_test):
    """Afficher les courbes ROC"""
    print("\n" + "=" * 80)
    print("7. COURBES ROC")
    print("=" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, result) in enumerate(results.items()):
        y_pred_proba = result['proba']

        disp = RocCurveDisplay.from_predictions(
            y_test, y_pred_proba, ax=axes[idx], color='steelblue', name=f'{name}'
        )
        axes[idx].set_title(f'Courbe ROC - {name}', fontsize=12, fontweight='bold')
        axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/12_MODELS_ROC_CURVES.png', dpi=300, bbox_inches='tight')
    print("\n✓ Courbes ROC sauvegardées: 12_MODELS_ROC_CURVES.png")
    plt.show()


# ==================== MAIN ====================

def main():
    """Fonction principale"""
    print("\n" + "=" * 80)
    print(" MENTAL HEALTH - ENTRAÎNEMENT & OPTIMISATION")
    print("=" * 80)

    try:
        import os
        os.makedirs('visualizations', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # 1. Charger et prétraiter
        df, encoders = load_and_preprocess('data/ipynb_checkpoints/Mental Health Classification.csv')

        # 2. Feature Engineering
        df = create_features(df)

        # 3. Préparer données
        X_train, X_test, y_train, y_test, scaler = prepare_data(df)

        # 4. Optimiser
        best_models = optimize_models(X_train, y_train)

        # 5. Évaluer
        results = evaluate_models(best_models, X_test, y_test)

        # 6. Résumé
        summary = create_summary(results)

        # 7. Courbes ROC
        plot_roc_curves(results, y_test)

        # 8. Sauvegarder meilleur modèle
        best_model_name = max(results, key=lambda x: results[x]['f1'])
        best_model = results[best_model_name]['model']

        with open('models/depression_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open('models/label_encoders.pkl', 'wb') as f:
            pickle.dump(encoders, f)

        print("\n" + "=" * 80)
        print(" PROJET TERMINÉ!")
        print("=" * 80)
        print(f"\n🏆 Meilleur modèle: {best_model_name}")
        print(f"   F1-Score: {results[best_model_name]['f1']:.4f}")
        print("\n Fichiers générés:")
        print("   ✓ 10_MODELS_CONFUSION_MATRICES.png")
        print("   ✓ 11_MODELS_PERFORMANCE_SUMMARY.png")
        print("   ✓ 12_MODELS_ROC_CURVES.png")
        print("   ✓ MODELS_PERFORMANCE_COMPARISON.csv")
        print("   ✓ models/")

    except Exception as e:
        print(f"\n ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()