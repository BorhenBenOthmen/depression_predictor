"""
EDA Visualization - Explorations et Visualisations des Données
✓ Exploration des données
✓ Détection valeurs aberrantes
✓ Feature selection et corrélation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Configuration Seaborn
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# ==================== SECTION 1: EXPLORATION DES DONNÉES ====================

def explore_data(df):
    """Exploration et visualisations initiales"""
    print("=" * 80)
    print(" EXPLORATION DES DONNÉES")
    print("=" * 80)

    print("\n Statistiques descriptives:")
    print(df.describe())

    print("\n Informations du dataset:")
    print(f"  - Dimensions: {df.shape}")
    print(f"  - Nombres de lignes: {df.shape[0]}")
    print(f"  - Nombres de colonnes: {df.shape[1]}")
    print(f"  - Valeurs manquantes totales: {df.isnull().sum().sum()}")

    # Visualisation: Distribution de la variable cible
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution des tentatives de suicide
    df['Suicide_Attempts'].value_counts().sort_index().plot(
        kind='bar', ax=axes[0], color='steelblue'
    )
    axes[0].set_title('Distribution des Tentatives de Suicide', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Nombre de tentatives')
    axes[0].set_ylabel('Fréquence')
    axes[0].grid(axis='y', alpha=0.3)

    # Distribution des types de dépression
    df['Depression_Type'].value_counts().plot(
        kind='bar', ax=axes[1], color='coral'
    )
    axes[1].set_title('Distribution des Types de Dépression', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Type de dépression')
    axes[1].set_ylabel('Fréquence')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/01_exploration_target.png', dpi=300, bbox_inches='tight')
    print("\n✓ Graphique sauvegardé: visualizations/01_exploration_target.png")
    plt.show()


# ==================== SECTION 2: DÉTECTION VALEURS ABERRANTES ====================

def detect_outliers(df):
    """Détection des valeurs aberrantes avec plusieurs méthodes"""
    print("\n" + "=" * 80)
    print("️ DÉTECTION DES VALEURS ABERRANTES")
    print("=" * 80)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Méthode IQR
    outliers_iqr = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        outliers_iqr[col] = outliers

    print("\n Valeurs aberrantes détectées (méthode IQR):")
    for col, count in sorted(outliers_iqr.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  - {col}: {count} aberrantes ({count/len(df)*100:.1f}%)")

    # Visualisation Boxplot - Partie 1 (10 premiers)
    numeric_cols_subset1 = numeric_cols[:10]
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols_subset1):
        sns.boxplot(data=df, y=col, ax=axes[idx], color='lightblue')
        axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/02_outliers_part1.png', dpi=300, bbox_inches='tight')
    print("\n✓ Graphique sauvegardé: visualizations/02_outliers_part1.png")
    plt.show()

    # Visualisation Boxplot - Partie 2 (10 suivants)
    if len(numeric_cols) > 10:
        numeric_cols_subset2 = numeric_cols[10:20]
        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        axes = axes.flatten()

        for idx, col in enumerate(numeric_cols_subset2):
            sns.boxplot(data=df, y=col, ax=axes[idx], color='lightgreen')
            axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('visualizations/02_outliers_part2.png', dpi=300, bbox_inches='tight')
        print("✓ Graphique sauvegardé: visualizations/02_outliers_part2.png")
        plt.show()

    return outliers_iqr


# ==================== SECTION 3: DISTRIBUTION DES VARIABLES ====================

def plot_distributions(df):
    """Visualiser les distributions des variables numériques"""
    print("\n" + "=" * 80)
    print(" DISTRIBUTION DES VARIABLES")
    print("=" * 80)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Histogrammes
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols[:12]):
        axes[idx].hist(df[col], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'Distribution de {col}', fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Valeur')
        axes[idx].set_ylabel('Fréquence')
        axes[idx].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/03_distributions.png', dpi=300, bbox_inches='tight')
    print("\n✓ Graphique sauvegardé: visualizations/03_distributions.png")
    plt.show()


# ==================== SECTION 4: SÉLECTION DES FEATURES ====================

def feature_selection_analysis(X, y):
    """Sélection des features les plus pertinentes"""
    print("\n" + "=" * 80)
    print(" SÉLECTION DES FEATURES")
    print("=" * 80)

    # Feature importance avec RandomForest
    print("\n Calcul de l'importance des features...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n✓ Top 15 features par importance:")
    print(feature_importance.head(15).to_string(index=False))

    # Visualisation Feature Importance
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title('Top 15 Features par Importance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/04_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Graphique sauvegardé: visualizations/04_feature_importance.png")
    plt.show()

    return feature_importance


# ==================== SECTION 5: MATRICE DE CORRÉLATION ====================

def plot_correlation_matrix(X):
    """Visualiser la matrice de corrélation"""
    print("\n" + "=" * 80)
    print(" MATRICE DE CORRÉLATION")
    print("=" * 80)

    # Matrice de corrélation complète
    X_df = pd.DataFrame(X)
    corr_matrix = X_df.corr()

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax,
                cbar_kws={'label': 'Corrélation'}, square=True)
    ax.set_title('Matrice de Corrélation Complète', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/05_correlation_matrix_full.png', dpi=300, bbox_inches='tight')
    print("\n✓ Graphique sauvegardé: visualizations/05_correlation_matrix_full.png")
    plt.show()

    # Matrice avec features les plus importantes seulement
    top_features = ['Danger_Score', 'Stress_Composite', 'Mental_Vulnerability',
                    'Depression_Score', 'Nervous_Level', 'Self_Harm',
                    'Low_SelfEsteem', 'Symptoms', 'Support_Index', 'Sleep_Quality']

    # Vérifier que les features existent
    available_features = [f for f in top_features if f in X_df.columns]

    if len(available_features) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_subset = X_df[available_features].corr()
        sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    ax=ax, cbar_kws={'label': 'Corrélation'}, square=True)
        ax.set_title('Matrice de Corrélation - Top Features', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('visualizations/05_correlation_matrix_top.png', dpi=300, bbox_inches='tight')
        print("✓ Graphique sauvegardé: visualizations/05_correlation_matrix_top.png")
        plt.show()


# ==================== SECTION 6: ANALYSE BIVARIÉE ====================

def bivariate_analysis(df):
    """Analyse bivariée: relations entre variables"""
    print("\n" + "=" * 80)
    print("🔍 ANALYSE BIVARIÉE")
    print("=" * 80)

    # Relation entre Depression_Score et Nervous_Level par rapport à Suicide_Attempts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Depression_Score vs Suicide_Attempts
    axes[0].scatter(df['Depression_Score'], df['Suicide_Attempts'],
                   alpha=0.5, c=df['Nervous_Level'], cmap='viridis', s=50)
    axes[0].set_xlabel('Depression Score', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Suicide Attempts', fontsize=11, fontweight='bold')
    axes[0].set_title('Depression Score vs Suicide Attempts\n(coloré par Nervous Level)',
                     fontsize=12, fontweight='bold')
    cbar = plt.colorbar(axes[0].collections[0], ax=axes[0])
    cbar.set_label('Nervous Level')
    axes[0].grid(alpha=0.3)

    # Plot 2: Sleep_Hours vs Suicide_Attempts
    axes[1].scatter(df['Sleep_Hours'], df['Suicide_Attempts'],
                   alpha=0.5, c=df['Low_Energy'], cmap='plasma', s=50)
    axes[1].set_xlabel('Sleep Hours', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Suicide Attempts', fontsize=11, fontweight='bold')
    axes[1].set_title('Sleep Hours vs Suicide Attempts\n(coloré par Low Energy)',
                     fontsize=12, fontweight='bold')
    cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
    cbar.set_label('Low Energy')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/06_bivariate_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Graphique sauvegardé: visualizations/06_bivariate_analysis.png")
    plt.show()


# ==================== FONCTION PRINCIPALE ====================

def main():
    """Fonction principale - Explorations et Visualisations"""
    print("\n" + "=" * 80)
    print("📊 EDA - EXPLORATIONS ET VISUALISATIONS DES DONNÉES")
    print("=" * 80)

    try:
        # Créer les dossiers
        import os
        os.makedirs('visualizations', exist_ok=True)

        # 1. Charger les données
        print("\n Chargement des données...")
        df = pd.read_csv('data/ipynb_checkpoints/Mental Health Classification.csv')
        print(f"✓ Dataset chargé: {df.shape}")

        # 2. Exploration
        explore_data(df)

        # 3. Détection aberrantes
        detect_outliers(df)

        # 4. Distributions
        plot_distributions(df)

        # 5. Feature selection (besoin de X, y préparés)
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        # Prétraitement minimal pour feature selection
        df_temp = df.copy()

        # Encoder
        for col in df_temp.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_temp[col] = le.fit_transform(df_temp[col])

        # Feature Engineering
        df_temp['Stress_Composite'] = df_temp['Depression_Score'] * df_temp['Nervous_Level'] / 100
        df_temp['Fatigue_Index'] = df_temp['Sleep_Hours'] * df_temp['Low_Energy']
        df_temp['Social_Depression_Impact'] = df_temp['SocialMedia_Hours'] * df_temp['Depression_Score'] / 100
        df_temp['Danger_Score'] = df_temp['Self_Harm'] + df_temp['Worsening_Depression'] + (df_temp['Symptoms'] / 10)
        df_temp['Mental_Vulnerability'] = df_temp['Low_SelfEsteem'] * df_temp['Low_Energy'] * df_temp['Nervous_Level']
        df_temp['Support_Index'] = df_temp['Mental_Health_Support'] + df_temp['Coping_Methods']
        df_temp['Sleep_Quality'] = abs(df_temp['Sleep_Hours'] - 7.5)

        X = df_temp.drop('Suicide_Attempts', axis=1)
        y = (df_temp['Suicide_Attempts'] >= 2).astype(int)

        # Feature selection
        feature_selection_analysis(X, y)

        # 6. Corrélation
        plot_correlation_matrix(X)

        # 7. Analyse bivariée
        bivariate_analysis(df)

        print("\n" + "=" * 80)
        print(" EXPLORATIONS TERMINÉES!")
        print("=" * 80)
        print("\n Fichiers générés dans visualizations/:")
        print("  ✓ 01_exploration_target.png")
        print("  ✓ 02_outliers_part1.png")
        print("  ✓ 02_outliers_part2.png")
        print("  ✓ 03_distributions.png")
        print("  ✓ 04_feature_importance.png")
        print("  ✓ 05_correlation_matrix_full.png")
        print("  ✓ 05_correlation_matrix_top.png")
        print("  ✓ 06_bivariate_analysis.png")

    except Exception as e:
        print(f"\n ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()