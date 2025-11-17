"""
Mental Health Classification - Application Flask
Application web pour prédire le type de dépression basé sur les caractéristiques de l'utilisateur
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
except Exception as e:
    print(f"❌ Erreur lors du chargement des modèles: {e}")
    print("Veuillez d'abord exécuter model_training.py")

# Configuration des features et leurs options
FEATURE_CONFIG = {
    'Gender': ['Male', 'Female', 'Other'],
    'Education_Level': ['High School', 'Bachelor', 'Master', 'PhD', 'None'],
    'Employment_Status': ['Employed', 'Unemployed', 'Student', 'Self-employed'],
    'Symptoms': ['Anxiety', 'Fatigue', 'Sadness', 'Insomnia', 'Loss of interest'],
    'Low_Energy': ['Yes', 'No'],
    'Low_SelfEsteem': ['Yes', 'No'],
    'Search_Depression_Online': ['Yes', 'No'],
    'Worsening_Depression': ['Yes', 'No'],
    'SocialMedia_WhileEating': ['Yes', 'No'],
    'Coping_Methods': ['Exercise', 'Therapy', 'Medication', 'Social support', 'None'],
    'Self_Harm': ['Yes', 'No'],
    'Mental_Health_Support': ['Yes', 'No'],
    'Suicide_Attempts': ['Yes', 'No']
}


@app.route('/')
def home():
    """Page d'accueil avec le formulaire de prédiction"""
    return render_template('index.html', config=FEATURE_CONFIG)


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour faire des prédictions"""
    try:
        # Récupérer les données du formulaire
        data = request.get_json()

        # Préparer les features dans le bon ordre
        features_dict = {
            'Gender': data.get('Gender'),
            'Age': float(data.get('Age', 0)),
            'Education_Level': data.get('Education_Level'),
            'Employment_Status': data.get('Employment_Status'),
            'Symptoms': data.get('Symptoms'),
            'Low_Energy': data.get('Low_Energy'),
            'Low_SelfEsteem': data.get('Low_SelfEsteem'),
            'Search_Depression_Online': data.get('Search_Depression_Online'),
            'Worsening_Depression': data.get('Worsening_Depression'),
            'Your overeating level': float(data.get('Your_overeating_level', 0)),
            'How many times you eat': float(data.get('How_many_times_you_eat', 0)),
            'SocialMedia_Hours': float(data.get('SocialMedia_Hours', 0)),
            'SocialMedia_WhileEating': data.get('SocialMedia_WhileEating'),
            'Sleep_Hours': float(data.get('Sleep_Hours', 0)),
            'Nervous_Level': float(data.get('Nervous_Level', 0)),
            'Depression_Score': float(data.get('Depression_Score', 0)),
            'Coping_Methods': data.get('Coping_Methods'),
            'Self_Harm': data.get('Self_Harm'),
            'Mental_Health_Support': data.get('Mental_Health_Support'),
            'Suicide_Attempts': data.get('Suicide_Attempts')
        }

        # Créer un DataFrame
        input_df = pd.DataFrame([features_dict])

        # Encoder les variables catégorielles
        categorical_columns = ['Gender', 'Education_Level', 'Employment_Status', 'Symptoms',
                               'Low_Energy', 'Low_SelfEsteem', 'Search_Depression_Online',
                               'Worsening_Depression', 'SocialMedia_WhileEating', 'Coping_Methods',
                               'Self_Harm', 'Mental_Health_Support', 'Suicide_Attempts']

        for col in categorical_columns:
            if col in label_encoders and col in input_df.columns:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except ValueError:
                    # Si la valeur n'est pas dans les classes connues, utiliser la première classe
                    input_df[col] = 0

        # Normaliser
        input_scaled = scaler.transform(input_df)

        # Prédiction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Décoder la prédiction
        depression_type = label_encoders['Depression_Type'].inverse_transform(prediction)[0]

        # Obtenir les probabilités pour chaque classe
        classes = label_encoders['Depression_Type'].classes_
        probabilities = {classes[i]: float(prediction_proba[0][i]) for i in range(len(classes))}

        # Recommandations basées sur le type de dépression
        recommendations = get_recommendations(depression_type, features_dict)

        return jsonify({
            'success': True,
            'prediction': depression_type,
            'probabilities': probabilities,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


def get_recommendations(depression_type, features):
    """Générer des recommandations basées sur le type de dépression et les features"""
    recommendations = []

    # Recommandations générales
    base_recommendations = {
        'Major Depression': [
            "Consultez un professionnel de santé mentale dès que possible",
            "Envisagez une thérapie cognitivo-comportementale (TCC)",
            "Parlez à votre médecin des options de traitement médicamenteux",
            "Établissez une routine quotidienne stable"
        ],
        'Persistent Depressive Disorder': [
            "Maintenez un suivi régulier avec un thérapeute",
            "Développez des stratégies d'adaptation à long terme",
            "Pratiquez la pleine conscience et la méditation",
            "Rejoignez un groupe de soutien"
        ],
        'Bipolar Disorder': [
            "Consultez un psychiatre spécialisé dans les troubles bipolaires",
            "Maintenez un journal de l'humeur",
            "Établissez une routine de sommeil régulière",
            "Évitez l'alcool et les substances"
        ],
        'Seasonal Affective Disorder': [
            "Exposez-vous à la lumière naturelle autant que possible",
            "Envisagez la luminothérapie",
            "Maintenez une activité physique régulière",
            "Planifiez des activités sociales"
        ],
        'Postpartum Depression': [
            "Parlez à votre médecin immédiatement",
            "Acceptez l'aide de votre entourage",
            "Rejoignez un groupe de soutien pour jeunes parents",
            "Ne restez pas seule avec vos émotions"
        ]
    }

    recommendations.extend(base_recommendations.get(depression_type, [
        "Consultez un professionnel de santé mentale",
        "Prenez soin de votre bien-être général"
    ]))

    # Recommandations spécifiques basées sur les features
    if features.get('Sleep_Hours', 0) < 6:
        recommendations.append("⚠️ Améliorez votre hygiène de sommeil (visez 7-9 heures)")

    if features.get('SocialMedia_Hours', 0) > 4:
        recommendations.append("⚠️ Réduisez votre temps sur les réseaux sociaux")

    if features.get('Nervous_Level', 0) > 7:
        recommendations.append("⚠️ Pratiquez des techniques de relaxation (respiration, yoga)")

    if features.get('Mental_Health_Support') == 'No':
        recommendations.append("⚠️ Cherchez un soutien professionnel en santé mentale")

    if features.get('Self_Harm') == 'Yes' or features.get('Suicide_Attempts') == 'Yes':
        recommendations.insert(0, "🚨 URGENT: Contactez immédiatement une ligne de crise (ex: 3114 en France)")

    return recommendations


@app.route('/info')
def info():
    """Page d'information sur l'application"""
    model_info = {
        'model_type': type(model).__name__,
        'features_count': len(scaler.mean_),
        'depression_types': list(label_encoders['Depression_Type'].classes_)
    }
    return render_template('info.html', model_info=model_info)


if __name__ == '__main__':
    # Créer le dossier templates s'il n'existe pas
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("✓ Dossier 'templates' créé")

    print("\n" + "=" * 60)
    print("🌐 APPLICATION FLASK - MENTAL HEALTH CLASSIFICATION")
    print("=" * 60)
    print("\n✓ Serveur démarré!")
    print("✓ Ouvrez votre navigateur: http://127.0.0.1:5000")
    print("\nAppuyez sur Ctrl+C pour arrêter le serveur\n")

    app.run(debug=True, host='0.0.0.0', port=5000)