import pandas as pd
from config import NIVEAU_MAP
import streamlit as st

@st.cache_data
def extraire_features(df):
    """Préparer les données pour l'intelligence artificielle"""
    if df.empty:
        return df

    # Mapping numérique pour les niveaux
    niveau_num_map = {'V': 0, 'D': 1, 'I': 2, 'W': 3, 'E': 4, 'F': 5}
    df['niveau_num']       = df['niveau'].map(niveau_num_map).fillna(2)
    df['longueur_message'] = df['message'].apply(len)

    # Fréquence par composant (discriminant pour Spring Boot)
    df['frequence_composant'] = df['composant'].map(df['composant'].value_counts())
    df['frequence_flux']      = df['frequence_composant']

    # ── Features spécifiques Nginx ──────────────────────────────────────
    # Extraire le code HTTP depuis le message Nginx
    import re
    def extraire_status(msg):
        m = re.search(r'Status:\s*(\d{3})', str(msg))
        if m:
            return int(m.group(1))
        return 200  # défaut : OK

    def extraire_taille(msg):
        m = re.search(r'Taille:\s*(\d+)', str(msg))
        if m:
            return int(m.group(1))
        return 0

    mask_nginx = df['source'] == 'Nginx'
    if mask_nginx.any():
        df.loc[mask_nginx, 'http_status'] = (
            df.loc[mask_nginx, 'message'].apply(extraire_status)
        )
        df.loc[mask_nginx, 'taille_reponse'] = (
            df.loc[mask_nginx, 'message'].apply(extraire_taille)
        )
        # Fréquence par IP (anomalie = IP qui fait beaucoup de requêtes)
        ip_freq = df.loc[mask_nginx, 'ip'].map(
            df.loc[mask_nginx, 'ip'].value_counts()
        )
        df.loc[mask_nginx, 'frequence_ip'] = ip_freq

        # Niveau_num renforcé pour les erreurs HTTP
        def niveau_from_status(status):
            if status >= 500: return 5   # critique
            if status >= 400: return 4   # warning fort
            return 2                     # normal
        df.loc[mask_nginx, 'niveau_num'] = (
            df.loc[mask_nginx, 'http_status'].apply(niveau_from_status)
        )
        # Longueur du message = taille réponse (plus discriminante que len(msg))
        df.loc[mask_nginx, 'longueur_message'] = df.loc[mask_nginx, 'taille_reponse']
        # Fréquence_flux = fréquence IP pour Nginx
        df.loc[mask_nginx, 'frequence_flux'] = ip_freq

    # Remplir les colonnes optionnelles pour Spring Boot
    for col in ['http_status', 'taille_reponse', 'frequence_ip']:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    return df