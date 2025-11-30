"""
Mental Health Classification - Application Flask (VERSION TUNISIE)
Prédiction: 2 niveaux (0=Pas de risque, 1=Risque)
Les probabilités retournées sont RÉELLES, du modèle
Numéros d'urgence Tunisiens + Améliorations UX
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
        'label': 'Pas de Risque Identifié',
        'color': 'green',
        'description': 'Selon notre évaluation, aucun signe critique de risque suicidaire n\'a été détecté. Continuez à surveiller votre santé mentale et maintenez vos habitudes de bien-être.',
        'urgency': 'Suivi régulier recommandé'
    },
    1: {
        'label': 'Risque Identifié - Action Immédiate Recommandée',
        'color': 'red',
        'description': 'Notre évaluation a détecté des indicateurs de risque. Il est fortement recommandé de contacter immédiatement un professionnel de santé mentale ou une ligne d\'écoute.',
        'urgency': 'ACTION IMMÉDIATE REQUISE'
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
            "Continuez à surveiller votre santé mentale régulièrement",
            "Maintenez vos habitudes de bien-être et d'activité physique",
            "Restez en contact régulier avec votre réseau social",
            "Consultez un professionnel de santé pour un suivi annuel"
        ]
    else:
        recommendations = [
            "CONTACTEZ IMMÉDIATEMENT UN PROFESSIONNEL DE SANTÉ",
            "Ressources en Tunisie:",
            "  - Numéro National d'Écoute: 1445 (gratuit, 24h/24)",
            "  - Urgences Médicales: 15",
            "  - Pompiers/Secours: 198",
            "  - Samu Social: 215 836 666",
            "Contactez votre médecin généraliste ou un psychiatre",
            "Prévenez un membre de la famille ou un proche de confiance"
        ]

    # Recommandations spécifiques basées sur les features détectées
    if features.get('Sleep_Hours', 6) < 5:
        recommendations.append("IMPORTANT: Votre sommeil est très insuffisant (moins de 5h/nuit). Cela peut aggraver votre état mental. Cherchez de l'aide")
    elif features.get('Sleep_Hours', 6) < 6:
        recommendations.append("Votre sommeil semble insuffisant. Essayez d'améliorer votre hygiène de sommeil (visez 7-9 heures)")
    elif features.get('Sleep_Hours', 6) > 12:
        recommendations.append("Un sommeil excessif (plus de 12h/nuit) peut être un signe de dépression. Consultez un médecin")

    if features.get('SocialMedia_Hours', 0) > 6:
        recommendations.append("Vous passez beaucoup de temps sur les réseaux sociaux (plus de 6h/jour). Réduisez ce temps, cela peut affecter votre santé mentale")

    if features.get('Nervous_Level', 0) > 8:
        recommendations.append("Vous signalerez un niveau d'anxiété très élevé. Pratiquez la méditation, la respiration profonde ou contactez un professionnel")

    if features.get('Low_Energy', 0) == 1:
        recommendations.append("Vous avez signalé une faible énergie. Essayez une activité physique régulière (même 20 minutes de marche par jour)")

    if features.get('Low_SelfEsteem', 0) == 1:
        recommendations.append("Une faible estime de soi a été détectée. Une thérapie ou un groupe de soutien pourrait vous aider")

    if features.get('Self_Harm', 0) == 1:
        recommendations.insert(0, "ALERTE: Auto-mutilation signalée. Contactez immédiatement un professionnel au 1445 ou allez à l'hôpital le plus proche")

    if features.get('Depression_Score', 0) > 25:
        recommendations.insert(0, "ALERTE: Score de dépression très élevé détecté. Une intervention médicale est urgente")

    if features.get('Mental_Health_Support', 0) == 0:
        recommendations.append("Vous n'avez pas signalé de soutien professionnel. N'hésitez pas à chercher de l'aide - c'est important!")

    if features.get('Coping_Methods', 0) == 0:
        recommendations.append("Développez des stratégies d'adaptation: exercice, loisirs, socialisation, ou autres activités que vous aimez")

    if features.get('Worsening_Depression', 0) == 1:
        recommendations.append("Vous avez indiqué une aggravation récente. Ne temporisez pas - contactez un professionnel au plus tôt")

    return recommendations

@app.route('/info')
def info():
    """Page d'information sur l'application"""
    try:
        model_info = {
            'model_type': type(model).__name__,
            'features_count': len(scaler.mean_),
            'target': 'Évaluation du Risque Suicidaire',
            'risk_levels': list(range(2)),
            'risk_labels': [RISK_LEVELS[i]['label'] for i in range(2)]
        }
    except:
        model_info = {
            'model_type': 'Modèle non chargé',
            'features_count': 0,
            'target': 'Évaluation du Risque Suicidaire',
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
    print("Version: Tunisie - Numéros d'urgence locaux")
    print("Modèle: Classification BINAIRE (2 niveaux)")
    print("\nAppuyez sur Ctrl+C pour arrêter le serveur\n")
    app.run(debug=True, host='0.0.0.0', port=5000)