


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
from config import MOTS_CRITIQUES

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False


def entrainer_xgboost(df_feedback):
    """
    Entraîne un modèle XGBoost (ou GradientBoostingClassifier en fallback)
    sur les feedbacks de validation utilisateur.
    """
    if df_feedback.empty or len(df_feedback) < 5:
        return None, None, None

    try:
        # Features numériques
        features_num = df_feedback[['niveau_num', 'longueur_message', 'frequence_flux']].fillna(0)
        scaler = StandardScaler()
        features_num_norm = scaler.fit_transform(features_num)

        # Features textuelles TF-IDF
        corpus = (
            df_feedback['composant'].fillna('') + ' ' +
            df_feedback['requete'].fillna('') + ' ' +
            df_feedback['message'].fillna('')
        )
        vectorizer = TfidfVectorizer(vocabulary=MOTS_CRITIQUES, lowercase=True)
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Fusion des features
        features_finales = sp.hstack([tfidf_matrix, sp.csr_matrix(features_num_norm)], format='csr')
        X = features_finales
        y = df_feedback['label'].astype(int)

        # Entraînement du modèle
        if HAS_XGB:
            modele = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            modele = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )

        modele.fit(X, y)
        return modele, scaler, vectorizer
    except Exception as e:
        return None, None, None


def predire_xgboost(df_anomalies, modele, scaler, vectorizer):
    """
    Prédit si les anomalies sont des vraies anomalies (1) ou des faux positifs (0)
    en utilisant le modèle XGBoost entraîné.
    """
    if df_anomalies.empty or modele is None or scaler is None or vectorizer is None:
        return None, None

    try:
        # Features numériques
        features_num = df_anomalies[['niveau_num', 'longueur_message', 'frequence_flux']].fillna(0)
        features_num_norm = scaler.transform(features_num)

        # Features textuelles TF-IDF
        corpus = (
            df_anomalies['composant'].fillna('') + ' ' +
            df_anomalies['requete'].fillna('') + ' ' +
            df_anomalies['message'].fillna('')
        )
        tfidf_matrix = vectorizer.transform(corpus)

        # Fusion des features
        features_finales = sp.hstack([tfidf_matrix, sp.csr_matrix(features_num_norm)], format='csr')

        # Prédiction
        preds = modele.predict(features_finales)
        
        # Probabilités
        if hasattr(modele, "predict_proba"):
            probs = modele.predict_proba(features_finales)
            confiances = [probs[i][pred] for i, pred in enumerate(preds)]
        else:
            confiances = [1.0] * len(preds)

        return preds, confiances
    except Exception as e:
        return None, None


def get_stats_xgboost(modele):
    """
    Retourne des métadonnées sur le modèle entraîné.
    """
    if modele is None:
        return {}

    try:
        if HAS_XGB:
            return {
                'n_estimators': modele.n_estimators,
                'max_depth': modele.max_depth if modele.max_depth is not None else 6,
                'type': 'XGBoost'
            }
        else:
            return {
                'n_estimators': modele.n_estimators,
                'max_depth': modele.max_depth,
                'type': 'GradientBoosting'
            }
    except Exception:
        return {'n_estimators': 100, 'max_depth': 3, 'type': 'Inconnu'}
