"""
Mental Health Classification - Application Flask
Application web pour prédire le risque de suicide basé sur les caractéristiques de l'utilisateur
VERSION MODIFIÉE: Prédiction de Suicide_Attempts
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Charger le modèle, scaler et encoders
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
    print("✓ Modèle, scaler et encoders chargés avec succès!")
    print(f"✓ Modèle type: {type(model).__name__}")
except Exception as e:
    print(f"❌ Erreur lors du chargement des modèles: {e}")
    print("Veuillez d'abord exécuter model_training.py")

# Mappage des niveaux de risque
RISK_LEVELS = {
    0: {
        'label': 'Risque Faible',
        'color': 'green',
        'description': 'Pas de tentatives de suicide signalées',
        'urgency': 'Normal'
    },
    1: {
        'label': 'Risque Modéré',
        'color': 'yellow',
        'description': 'Une tentative de suicide signalée',
        'urgency': 'À surveiller'
    },
    2: {
        'label': 'Risque Élevé',
        'color': 'orange',
        'description': 'Deux tentatives de suicide signalées',
        'urgency': 'Consultation recommandée'
    },
    3: {
        'label': 'Risque Critique',
        'color': 'red',
        'description': 'Trois tentatives de suicide signalées',
        'urgency': '🚨 URGENT - Contactez une ligne de crise'
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

        # Préparer les features (SANS Suicide_Attempts qui est la TARGET)
        # IMPORTANT: Créer avec exactement les mêmes noms de colonnes que dans le dataset original
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

        # Vérifier les colonnes
        print(f"Colonnes du DataFrame: {list(input_df.columns)}")
        print(f"Colonnes du scaler: {list(scaler.get_feature_names_out())}")

        # S'assurer que les colonnes sont dans le bon ordre et correspondent au scaler
        try:
            # Utiliser les noms exacts du scaler
            expected_columns = list(scaler.get_feature_names_out())
            input_df = input_df[expected_columns]
        except Exception as e:
            print(f"Erreur lors du réagencement des colonnes: {e}")
            raise

        # Normaliser
        input_scaled = scaler.transform(input_df)

        # Prédiction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Niveau de risque prédit (0, 1, 2, ou 3)
        suicide_risk_level = int(prediction[0])

        # Obtenir les probabilités pour chaque classe
        risk_probabilities = {
            f'Risk_Level_{i}': float(prediction_proba[0][i])
            for i in range(len(prediction_proba[0]))
        }

        # Informations sur le risque
        risk_info = RISK_LEVELS.get(suicide_risk_level, RISK_LEVELS[0])

        # Générer les recommandations
        recommendations = get_recommendations(suicide_risk_level, features_dict)

        return jsonify({
            'success': True,
            'risk_level': suicide_risk_level,
            'risk_label': risk_info['label'],
            'risk_color': risk_info['color'],
            'risk_description': risk_info['description'],
            'urgency': risk_info['urgency'],
            'probabilities': risk_probabilities,
            'recommendations': recommendations
        })

    except Exception as e:
        print(f"Erreur: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Erreur lors de la prédiction: {str(e)}'
        }), 400


def get_recommendations(risk_level, features):
    """Générer des recommandations basées sur le niveau de risque et les features"""
    recommendations = []

    # Recommandations basées sur le niveau de risque
    risk_recommendations = {
        0: [
            "✓ Continuez à surveiller votre santé mentale",
            "✓ Maintenez vos habitudes de bien-être",
            "✓ Restez en contact avec votre réseau social",
            "✓ Consultez un professionnel une fois par an"
        ],
        1: [
            "⚠️ Consultez un professionnel de santé mentale dès que possible",
            "⚠️ Envisagez une thérapie cognitivo-comportementale (TCC)",
            "⚠️ Parlez à votre médecin des options de traitement",
            "⚠️ Établissez une routine quotidienne stable"
        ],
        2: [
            "🔴 Consultation URGENTE avec un psychiatre recommandée",
            "🔴 Envisagez un traitement médicamenteux",
            "🔴 Réduisez votre exposition aux facteurs de stress",
            "🔴 Restez entouré(e) et en contact régulier avec votre réseau",
            "🔴 Envisagez un groupe de soutien"
        ],
        3: [
            "🚨 URGENT - Contactez immédiatement une ligne de crise",
            "🚨 France: 3114 (gratuit, 24h/24, 7j/7)",
            "🚨 Appelez le SAMU (15) ou les pompiers (18) si nécessaire",
            "🚨 Allez aux urgences si vous avez des pensées suicidaires",
            "🚨 Prévenez quelqu'un de confiance immédiatement"
        ]
    }

    recommendations.extend(risk_recommendations.get(risk_level, []))

    # Recommandations spécifiques basées sur les features
    if features.get('Sleep_Hours', 6) < 6:
        recommendations.append("💤 Améliorez votre hygiène de sommeil (visez 7-9 heures)")

    if features.get('Sleep_Hours', 6) > 10:
        recommendations.append("💤 Un sommeil excessif peut être un signe - Consultez un médecin")

    if features.get('SocialMedia_Hours', 0) > 4:
        recommendations.append("📱 Réduisez votre temps sur les réseaux sociaux (max 2-3 heures/jour)")

    if features.get('Nervous_Level', 0) > 7:
        recommendations.append("🧘 Pratiquez des techniques de relaxation (respiration, yoga, méditation)")

    if features.get('Low_Energy', 0) == 1:
        recommendations.append("⚡ Faible énergie signalée - Pratiquez une activité physique régulière")

    if features.get('Low_SelfEsteem', 0) == 1:
        recommendations.append("💪 Basse estime de soi - Envisagez une thérapie cognitivo-comportementale")

    if features.get('Self_Harm', 0) == 1:
        recommendations.insert(0, "🚨 AUTO-BLESSURES SIGNALÉES - Appelez une ligne de crise immédiatement!")

    if features.get('Mental_Health_Support', 0) == 0:
        recommendations.append("🤝 Cherchez un soutien professionnel ou rejoignez un groupe de soutien")

    if features.get('Coping_Methods', 0) == 0:
        recommendations.append("🎯 Développez des stratégies d'adaptation saines (exercice, hobby, socialisation)")

    return recommendations


@app.route('/info')
def info():
    """Page d'information sur l'application"""
    try:
        model_info = {
            'model_type': type(model).__name__,
            'features_count': len(scaler.mean_),
            'target': 'Suicide_Attempts',
            'risk_levels': list(range(4)),
            'risk_labels': [RISK_LEVELS[i]['label'] for i in range(4)]
        }
    except:
        model_info = {
            'model_type': 'Modèle non chargé',
            'features_count': 0,
            'target': 'Suicide_Attempts',
            'risk_levels': list(range(4)),
            'risk_labels': [RISK_LEVELS[i]['label'] for i in range(4)]
        }

    return render_template('info.html', model_info=model_info)


@app.route('/health')
def health():
    """Endpoint de santé pour vérifier que le serveur est en marche"""
    return jsonify({
        'status': 'healthy',
        'message': 'Application Mental Health Classification en marche'
    })


if __name__ == '__main__':
    # Créer le dossier templates s'il n'existe pas
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("✓ Dossier 'templates' créé")

    if not os.path.exists('models'):
        os.makedirs('models')
        print("✓ Dossier 'models' créé")

    print("\n" + "=" * 60)
    print("🌐 APPLICATION FLASK - MENTAL HEALTH CLASSIFICATION")
    print("=" * 60)
    print("\n✓ Serveur démarré!")
    print("✓ Ouvrez votre navigateur: http://127.0.0.1:5000")
    print("✓ Variable prédite: Suicide_Attempts")
    print("✓ Niveaux de risque: 0 (Faible) → 1 (Modéré) → 2 (Élevé) → 3 (Critique)")
    print("\nAppuyez sur Ctrl+C pour arrêter le serveur\n")

    app.run(debug=True, host='0.0.0.0', port=5000)