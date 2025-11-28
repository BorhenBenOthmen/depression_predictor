"""
Mental Health Classification - Application Flask (VERSION BINAIRE)
Prédiction: 2 niveaux (0=Pas de risque, 1=Risque)
Les probabilités retournées sont RÉELLES, du modèle
"""
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Chargement du modèle, scaler et encoders
MODEL_PATH = 'models/depression_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
ENCODERS_PATH = 'models/label_encoders.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(ENCODERS_PATH, 'rb') as f:
        label_encoders = pickle.load(f)
    print("Modèle, scaler et encoders chargés avec succès!")
    print(f"Modèle type: {type(model).__name__}")
    print(f"Prédiction: BINAIRE (2 niveaux)")
except Exception as e:
    print(f"Erreur lors du chargement des modèles: {e}")
    print("Veuillez d'abord exécuter model_training.py")

# Mappage des 2 niveaux de risque
RISK_LEVELS = {
    0: {
        'label': 'Pas de Risque',
        'color': 'green',
        'description': 'Pas de tentatives de suicide signalées. Continuez à surveiller votre santé mentale.',
        'urgency': 'Normal - Suivi régulier recommandé'
    },
    1: {
        'label': 'Risque Identifié',
        'color': 'red',
        'description': 'Risque de suicide identifié. Une consultation professionnelle est fortement recommandée.',
        'urgency': 'URGENT - Cherchez de l\'aide immédiatement'
    }
}

@app.route('/')
def home():

    """Page d'accueil avec le formulaire de prédiction"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour faire des prédictions"""
    try:
        # Récupérer les données du formulaire
        data = request.get_json()

        # Préparer les features
        features_dict = {
            'Gender': int(data.get('Gender', 0)),
            'Age': int(data.get('Age', 25)),
            'Education_Level': int(data.get('Education_Level', 0)),
            'Employment_Status': int(data.get('Employment_Status', 0)),
            'Depression_Type': int(data.get('Depression_Type', 0)),
            'Symptoms': int(data.get('Symptoms', 0)),
            'Low_Energy': int(data.get('Low_Energy', 0)),
            'Low_SelfEsteem': int(data.get('Low_SelfEsteem', 0)),
            'Search_Depression_Online': int(data.get('Search_Depression_Online', 0)),
            'Worsening_Depression': int(data.get('Worsening_Depression', 0)),
            'Your overeating level': int(data.get('Your_overeating_level', 0)),
            'How many times you eat ': int(data.get('How_many_times_you_eat', 0)),
            'SocialMedia_Hours': int(data.get('SocialMedia_Hours', 0)),
            'SocialMedia_WhileEating': int(data.get('SocialMedia_WhileEating', 0)),
            'Sleep_Hours': int(data.get('Sleep_Hours', 6)),
            'Nervous_Level': int(data.get('Nervous_Level', 0)),
            'Depression_Score': int(data.get('Depression_Score', 0)),
            'Coping_Methods': int(data.get('Coping_Methods', 0)),
            'Self_Harm': int(data.get('Self_Harm', 0)),
            'Mental_Health_Support': int(data.get('Mental_Health_Support', 0))
        }

        # Créer un DataFrame
        input_df = pd.DataFrame([features_dict])

        # CRÉER LES 7 NOUVELLES FEATURES (Feature Engineering)
        input_df['Stress_Composite'] = input_df['Depression_Score'] * input_df['Nervous_Level'] / 100
        input_df['Fatigue_Index'] = input_df['Sleep_Hours'] * input_df['Low_Energy']
        input_df['Social_Depression_Impact'] = input_df['SocialMedia_Hours'] * input_df['Depression_Score'] / 100
        input_df['Danger_Score'] = input_df['Self_Harm'] + input_df['Worsening_Depression'] + (input_df['Symptoms'] / 10)
        input_df['Mental_Vulnerability'] = input_df['Low_SelfEsteem'] * input_df['Low_Energy'] * input_df['Nervous_Level']
        input_df['Support_Index'] = input_df['Mental_Health_Support'] + input_df['Coping_Methods']
        input_df['Sleep_Quality'] = abs(input_df['Sleep_Hours'] - 7.5)

        # Normaliser
        input_scaled = scaler.transform(input_df)

        # PRÉDICTION BINAIRE
        prediction = model.predict(input_scaled)[0]  # Retourne 0 ou 1
        prediction_proba = model.predict_proba(input_scaled)[0]  # [prob_0, prob_1]

        # Probabilités réelles
        risk_probabilities = {
            'Risk_Level_0': float(prediction_proba[0]),
            'Risk_Level_1': float(prediction_proba[1])
        }

        # Informations sur le risque
        risk_level = int(prediction)
        risk_info = RISK_LEVELS[risk_level]

        # Générer les recommandations
        recommendations = get_recommendations(risk_level, features_dict)

        return jsonify({
            'success': True,
            'risk_level': risk_level,
            'risk_label': risk_info['label'],
            'risk_color': risk_info['color'],
            'risk_description': risk_info['description'],
            'urgency': risk_info['urgency'],
            'probabilities': risk_probabilities,
            'recommendations': recommendations
        })

    except Exception as e:
        print(f"Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Erreur lors de la prédiction: {str(e)}'
        }), 400

def get_recommendations(risk_level, features):
    """Générer des recommandations basées sur le niveau de risque et les features"""
    recommendations = []

    if risk_level == 0:
        recommendations = [
            "Continuez à surveiller votre santé mentale",
            "Maintenez vos habitudes de bien-être",
            "Restez en contact avec votre réseau social",
            "Consultez un professionnel une fois par an pour un suivi"
        ]
    else:
        recommendations = [
            "URGENT - Cherchez de l'aide immédiatement",
            "France: 3114 (gratuit, 24h/24, 7j/7)",
            "SAMU: 15 | Pompiers: 18",
            "Consultez un professionnel de santé mentale dès que possible",
            "Contactez votre médecin ou un psychiatre",
            "Prévenez quelqu'un de confiance de votre situation"
        ]

    if features.get('Sleep_Hours', 6) < 6:
        recommendations.append("Améliorez votre hygiène de sommeil (visez 7-9 heures)")
    if features.get('Sleep_Hours', 6) > 10:
        recommendations.append("Un sommeil excessif peut être un signe - Consultez un médecin")
    if features.get('SocialMedia_Hours', 0) > 4:
        recommendations.append("Réduisez votre temps sur les réseaux sociaux (max 2-3 heures/jour)")
    if features.get('Nervous_Level', 0) > 7:
        recommendations.append("Pratiquez des techniques de relaxation (respiration, yoga, méditation)")
    if features.get('Low_Energy', 0) == 1:
        recommendations.append("Faible énergie signalée - Pratiquez une activité physique régulière")
    if features.get('Low_SelfEsteem', 0) == 1:
        recommendations.append("Basse estime de soi - Envisagez une thérapie cognitivo-comportementale")
    if features.get('Self_Harm', 0) == 1:
        recommendations.insert(0, "AUTO-BLESSURES SIGNALÉES - Appelez une ligne de crise immédiatement!")
    if features.get('Mental_Health_Support', 0) == 0:
        recommendations.append("Cherchez un soutien professionnel ou rejoignez un groupe de soutien")
    if features.get('Coping_Methods', 0) == 0:
        recommendations.append("Développez des stratégies d'adaptation saines (exercice, hobby, socialisation)")
    if features.get('Depression_Score', 0) > 20:
        recommendations.append("Score de dépression élevé - Consultation médicale urgente")

    return recommendations

@app.route('/info')
def info():
    """Page d'information sur l'application"""
    try:
        model_info = {
            'model_type': type(model).__name__,
            'features_count': len(scaler.mean_),
            'target': 'Suicide_Attempts (BINAIRE)',
            'risk_levels': list(range(2)),
            'risk_labels': [RISK_LEVELS[i]['label'] for i in range(2)]
        }
    except:
        model_info = {
            'model_type': 'Modèle non chargé',
            'features_count': 0,
            'target': 'Suicide_Attempts',
            'risk_levels': list(range(2)),
            'risk_labels': [RISK_LEVELS[i]['label'] for i in range(2)]
        }
    return render_template('info.html', model_info=model_info)

@app.route('/health')
def health():
    """Endpoint de santé"""
    return jsonify({
        'status': 'healthy',
        'message': 'Application Mental Health Classification en marche'
    })

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Dossier 'templates' créé")
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Dossier 'models' créé")

    print("\n" + "=" * 60)
    print("APPLICATION FLASK - MENTAL HEALTH CLASSIFICATION")
    print("=" * 60)
    print("\nServeur démarré!")
    print("Ouvrez votre navigateur: http://127.0.0.1:5000")
    print("Modèle: Classification BINAIRE (2 niveaux)")
    print("Probabilités: 2 prédictions réelles du modèle")
    print("Cohérence: Training → App → HTML (PARFAITE)")
    print("\nAppuyez sur Ctrl+C pour arrêter le serveur\n")
    app.run(debug=True, host='0.0.0.0', port=5000)