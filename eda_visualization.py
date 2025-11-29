"""
EDA Visualization - Explorations et Visualisations Complètes des Données
Noms de fichiers significatifs pour la présentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)


# ==================== 1. EXPLORATION DATASET ====================

def explore_dataset_overview(df):
    """Vue d'ensemble du dataset"""
    print("=" * 80)
    print(" EXPLORATION - VUE D'ENSEMBLE DU DATASET")
    print("=" * 80)

    print(f"\n Dimensions: {df.shape}")
    print(f" Nombre de lignes: {df.shape[0]}")
    print(f" Nombre de colonnes: {df.shape[1]}")
    print(f" Valeurs manquantes: {df.isnull().sum().sum()}")
    print(f" Types de données: {df.dtypes.value_counts().to_dict()}")

    # Statistiques par type
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    print(f"\n Variables numériques: {len(numeric_cols)}")
    print(f" Variables catégorielles: {len(categorical_cols)}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Nombre de colonnes par type
    type_counts = df.dtypes.value_counts()
    axes[0, 0].bar(type_counts.index.astype(str), type_counts.values, color='steelblue')
    axes[0, 0].set_title('Types de Données', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Nombre de colonnes')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. Statistiques manquantes
    missing = df.isnull().sum()
    axes[0, 1].text(0.5, 0.5, f'Valeurs manquantes:\n{missing.sum()}\n\nComplétude:\n{(1 - missing.sum()/(df.shape[0]*df.shape[1]))*100:.1f}%',
                   ha='center', va='center', fontsize=14, fontweight='bold', transform=axes[0, 1].transAxes)
    axes[0, 1].axis('off')

    # 3. Taille du dataset
    axes[1, 0].text(0.5, 0.5, f'Taille du Dataset:\n{df.shape[0]} lignes\n{df.shape[1]} colonnes\n\n{df.shape[0] * df.shape[1]} cellules',
                   ha='center', va='center', fontsize=14, fontweight='bold', transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')

    # 4. Distribution classe
    target_dist = df['Suicide_Attempts'].value_counts().sort_index()
    axes[1, 1].bar(target_dist.index, target_dist.values, color='coral')
    axes[1, 1].set_title('Distribution Variable Cible\n(Suicide_Attempts)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Nombre de tentatives')
    axes[1, 1].set_ylabel('Fréquence')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/01_DATASET_OVERVIEW.png', dpi=300, bbox_inches='tight')
    print("\n✓ Sauvegardé: 01_DATASET_OVERVIEW.png")
    plt.show()


# ==================== 2. DISTRIBUTION VARIABLE CIBLE ====================

def analyze_target_variable(df):
    """Analyse détaillée de la variable cible"""
    print("\n" + "=" * 80)
    print(" ANALYSE - VARIABLE CIBLE (SUICIDE_ATTEMPTS)")
    print("=" * 80)

    target = df['Suicide_Attempts']

    print(f"\n Statistiques:")
    print(f"  - Moyenne: {target.mean():.2f}")
    print(f"  - Médiane: {target.median():.2f}")
    print(f"  - Écart-type: {target.std():.2f}")
    print(f"  - Min: {target.min()}")
    print(f"  - Max: {target.max()}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribution
    target.value_counts().sort_index().plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Distribution Tentatives de Suicide', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Nombre de tentatives')
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. Distribution pie
    axes[0, 1].pie(target.value_counts(), labels=[f'{i} tentative(s)' for i in target.value_counts().index],
                   autopct='%1.1f%%', colors=['green', 'orange', 'red', 'darkred'])
    axes[0, 1].set_title('Proportion Tentatives de Suicide', fontsize=12, fontweight='bold')

    # 3. Histogram
    axes[1, 0].hist(target, bins=4, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Histogramme Tentatives de Suicide', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Nombre de tentatives')
    axes[1, 0].set_ylabel('Fréquence')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. Box plot
    axes[1, 1].boxplot(target, vert=True)
    axes[1, 1].set_title('Box Plot - Tentatives de Suicide', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Nombre de tentatives')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/02_TARGET_DISTRIBUTION.png', dpi=300, bbox_inches='tight')
    print("\n✓ Sauvegardé: 02_TARGET_DISTRIBUTION.png")
    plt.show()


# ==================== 3. VALEURS ABERRANTES ====================

def detect_and_visualize_outliers(df):
    """Détection complète des aberrantes"""
    print("\n" + "=" * 80)
    print("️ ANALYSE - VALEURS ABERRANTES")
    print("=" * 80)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    outliers_summary = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        outliers_summary[col] = {'count': outliers, 'percent': outliers/len(df)*100}

    print("\n Top 10 colonnes avec aberrantes:")
    sorted_outliers = sorted(outliers_summary.items(), key=lambda x: x[1]['count'], reverse=True)
    for col, stats in sorted_outliers[:10]:
        print(f"  - {col}: {stats['count']} ({stats['percent']:.1f}%)")

    # Boxplots - Partie 1
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols[:12]):
        sns.boxplot(data=df, y=col, ax=axes[idx], color='lightblue')
        axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)

    plt.suptitle('Détection Valeurs Aberrantes - Partie 1', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('visualizations/03_OUTLIERS_DETECTION_PART1.png', dpi=300, bbox_inches='tight')
    print("\n✓ Sauvegardé: 03_OUTLIERS_DETECTION_PART1.png")
    plt.show()

    # Boxplots - Partie 2
    if len(numeric_cols) > 12:
        fig, axes = plt.subplots(2, 5, figsize=(16, 8))
        axes = axes.flatten()

        for idx, col in enumerate(numeric_cols[12:22]):
            sns.boxplot(data=df, y=col, ax=axes[idx], color='lightgreen')
            axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)

        plt.suptitle('Détection Valeurs Aberrantes - Partie 2', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig('visualizations/03_OUTLIERS_DETECTION_PART2.png', dpi=300, bbox_inches='tight')
        print("✓ Sauvegardé: 03_OUTLIERS_DETECTION_PART2.png")
        plt.show()


# ==================== 4. DISTRIBUTIONS VARIABLES NUMÉRIQUES ====================

def analyze_numeric_distributions(df):
    """Analyse distributions variables numériques"""
    print("\n" + "=" * 80)
    print(" ANALYSE - DISTRIBUTIONS VARIABLES NUMÉRIQUES")
    print("=" * 80)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Histogrammes
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols[:12]):
        axes[idx].hist(df[col], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Valeur')
        axes[idx].set_ylabel('Fréquence')
        axes[idx].grid(axis='y', alpha=0.3)

        # Ajouter stats
        skewness = skew(df[col])
        kurt = kurtosis(df[col])
        axes[idx].text(0.98, 0.97, f'Skew: {skewness:.2f}\nKurt: {kurt:.2f}',
                      transform=axes[idx].transAxes, ha='right', va='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)

    plt.suptitle('Distributions Variables Numériques', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('visualizations/04_NUMERIC_DISTRIBUTIONS.png', dpi=300, bbox_inches='tight')
    print("\n✓ Sauvegardé: 04_NUMERIC_DISTRIBUTIONS.png")
    plt.show()


# ==================== 5. STATISTIQUES DESCRIPTIVES ====================

def descriptive_statistics(df):
    """Statistiques descriptives"""
    print("\n" + "=" * 80)
    print("📊 STATISTIQUES DESCRIPTIVES")
    print("=" * 80)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')

    stats_df = pd.DataFrame({
        'Colonne': numeric_cols,
        'Moyenne': [df[col].mean() for col in numeric_cols],
        'Médiane': [df[col].median() for col in numeric_cols],
        'Écart-type': [df[col].std() for col in numeric_cols],
        'Min': [df[col].min() for col in numeric_cols],
        'Max': [df[col].max() for col in numeric_cols],
        'Q25': [df[col].quantile(0.25) for col in numeric_cols],
        'Q75': [df[col].quantile(0.75) for col in numeric_cols]
    }).round(2)

    table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for i in range(len(stats_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Tableau Statistiques Descriptives', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('visualizations/05_DESCRIPTIVE_STATISTICS.png', dpi=300, bbox_inches='tight')
    print("\n✓ Sauvegardé: 05_DESCRIPTIVE_STATISTICS.png")
    plt.show()


# ==================== 6. FEATURE IMPORTANCE ====================

def feature_importance_analysis(X, y):
    """Importance des features"""
    print("\n" + "=" * 80)
    print(" SÉLECTION - IMPORTANCE DES FEATURES")
    print("=" * 80)

    print("\n Calcul de l'importance des features...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n✓ Top 15 features:")
    print(feature_importance.head(15).to_string(index=False))

    # Horizontal bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title('Top 15 Features par Importance (Random Forest)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/06_FEATURE_IMPORTANCE_TOP15.png', dpi=300, bbox_inches='tight')
    print("\n✓ Sauvegardé: 06_FEATURE_IMPORTANCE_TOP15.png")
    plt.show()

    # Toutes les features
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='coolwarm', ax=ax)
    ax.set_title('Importance de Toutes les Features', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/06_FEATURE_IMPORTANCE_ALL.png', dpi=300, bbox_inches='tight')
    print("✓ Sauvegardé: 06_FEATURE_IMPORTANCE_ALL.png")
    plt.show()

    return feature_importance


# ==================== 7. CORRÉLATION ====================

def correlation_analysis(X):
    """Matrice de corrélation"""
    print("\n" + "=" * 80)
    print(" ANALYSE - CORRÉLATION ENTRE FEATURES")
    print("=" * 80)

    X_df = pd.DataFrame(X)
    corr_matrix = X_df.corr()

    # Corrélation complète
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax,
                cbar_kws={'label': 'Corrélation'}, square=True)
    ax.set_title('Matrice Corrélation Complète - Toutes les Features', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/07_CORRELATION_MATRIX_FULL.png', dpi=300, bbox_inches='tight')
    print("\n✓ Sauvegardé: 07_CORRELATION_MATRIX_FULL.png")
    plt.show()

    # Corrélations fortes uniquement
    fig, ax = plt.subplots(figsize=(10, 8))

    # Trouver les corrélations fortes
    strong_corr = np.where(np.abs(corr_matrix) > 0.5)
    strong_corr_list = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                        for i, j in zip(*strong_corr) if i < j]

    if strong_corr_list:
        top_corr = sorted(strong_corr_list, key=lambda x: abs(x[2]), reverse=True)[:10]
        corr_pairs = [f"{x[0]} vs {x[1]}" for x in top_corr]
        corr_values = [x[2] for x in top_corr]

        colors = ['green' if x > 0 else 'red' for x in corr_values]
        ax.barh(corr_pairs, corr_values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Coefficient Corrélation')
        ax.set_title('Top 10 Corrélations Fortes entre Features', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig('visualizations/07_CORRELATION_STRONG_PAIRS.png', dpi=300, bbox_inches='tight')
        print("✓ Sauvegardé: 07_CORRELATION_STRONG_PAIRS.png")
        plt.show()


# ==================== 8. ANALYSE BIVARIÉE ====================

def bivariate_analysis(df):
    """Analyse bivariée"""
    print("\n" + "=" * 80)
    print(" ANALYSE - RELATIONS BIVARIÉES")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1
    axes[0, 0].scatter(df['Depression_Score'], df['Suicide_Attempts'],
                      alpha=0.5, c=df['Nervous_Level'], cmap='viridis', s=50)
    axes[0, 0].set_xlabel('Depression Score')
    axes[0, 0].set_ylabel('Suicide Attempts')
    axes[0, 0].set_title('Depression Score vs Suicide Attempts\n(coloré par Nervous Level)')
    axes[0, 0].grid(alpha=0.3)

    # Plot 2
    axes[0, 1].scatter(df['Sleep_Hours'], df['Suicide_Attempts'],
                      alpha=0.5, c=df['Low_Energy'], cmap='plasma', s=50)
    axes[0, 1].set_xlabel('Sleep Hours')
    axes[0, 1].set_ylabel('Suicide Attempts')
    axes[0, 1].set_title('Sleep Hours vs Suicide Attempts\n(coloré par Low Energy)')
    axes[0, 1].grid(alpha=0.3)

    # Plot 3
    axes[1, 0].scatter(df['Nervous_Level'], df['Suicide_Attempts'],
                      alpha=0.5, c=df['Depression_Score'], cmap='RdYlGn_r', s=50)
    axes[1, 0].set_xlabel('Nervous Level')
    axes[1, 0].set_ylabel('Suicide Attempts')
    axes[1, 0].set_title('Nervous Level vs Suicide Attempts\n(coloré par Depression Score)')
    axes[1, 0].grid(alpha=0.3)

    # Plot 4
    axes[1, 1].scatter(df['Self_Harm'], df['Suicide_Attempts'],
                      alpha=0.5, c=df['Low_SelfEsteem'], cmap='coolwarm', s=50)
    axes[1, 1].set_xlabel('Self Harm')
    axes[1, 1].set_ylabel('Suicide Attempts')
    axes[1, 1].set_title('Self Harm vs Suicide Attempts\n(coloré par Low SelfEsteem)')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/08_BIVARIATE_RELATIONSHIPS.png', dpi=300, bbox_inches='tight')
    print("\n✓ Sauvegardé: 08_BIVARIATE_RELATIONSHIPS.png")
    plt.show()


# ==================== 9. PAIRES VARIABLES ====================

def pairplot_analysis(df):
    """Pairplot des variables principales"""
    print("\n" + "=" * 80)
    print(" ANALYSE - PAIRPLOT DES VARIABLES PRINCIPALES")
    print("=" * 80)

    # Sélectionner les variables principales
    main_vars = ['Depression_Score', 'Nervous_Level', 'Sleep_Hours', 'Suicide_Attempts', 'Low_Energy']

    df_subset = df[main_vars].copy()

    pairplot = sns.pairplot(df_subset, diag_kind='hist', plot_kws={'alpha': 0.6},
                            diag_kws={'bins': 20, 'edgecolor': 'black'})
    pairplot.fig.suptitle('Pairplot Variables Principales', fontsize=14, fontweight='bold', y=1.00)

    plt.savefig('visualizations/09_PAIRPLOT_MAIN_VARIABLES.png', dpi=300, bbox_inches='tight')
    print("\n✓ Sauvegardé: 09_PAIRPLOT_MAIN_VARIABLES.png")
    plt.show()


# ==================== FONCTION PRINCIPALE ====================

def main():
    """Fonction principale"""
    print("\n" + "=" * 80)
    print(" EDA COMPLÈTE - EXPLORATIONS ET VISUALISATIONS ENRICHIES")
    print("=" * 80)

    try:
        import os
        os.makedirs('visualizations', exist_ok=True)

        # Charger données
        print("\n Chargement des données...")
        df = pd.read_csv('data/ipynb_checkpoints/Mental Health Classification.csv')
        print(f"✓ Dataset chargé: {df.shape}")

        # 1. Vue d'ensemble
        explore_dataset_overview(df)

        # 2. Variable cible
        analyze_target_variable(df)

        # 3. Aberrantes
        detect_and_visualize_outliers(df)

        # 4. Distributions
        analyze_numeric_distributions(df)

        # 5. Statistiques
        descriptive_statistics(df)

        # Préparation pour feature analysis
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        df_temp = df.copy()
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

        # 6. Feature Importance
        feature_importance_analysis(X, y)

        # 7. Corrélation
        correlation_analysis(X)

        # 8. Bivariée
        bivariate_analysis(df)

        # 9. Pairplot
        pairplot_analysis(df)

        print("\n" + "=" * 80)
        print(" EDA COMPLÈTE TERMINÉE!")
        print("=" * 80)

    except Exception as e:
        print(f"\n ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()