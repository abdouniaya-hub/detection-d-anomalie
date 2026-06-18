import re
import pandas as pd
import streamlit as st
from config import NIVEAU_MAP, MOIS_MAP

@st.cache_data
def lire_fichier(filepath):
    """Lire un fichier log ligne par ligne."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.readlines()

def parser_springboot(logs):
    """
    Parser les logs Spring Boot OCP.
    Les champs [user:] [email:] [ip:] [trace:] [requete:] sont rendus
    OPTIONNELS pour accepter tous les formats rencontrés en production.
    """
    entries  = []
    rejetees = 0

    pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2})\s+'            # groupe 1  : date
        r'(\d{2}):(\d{2}):(\d{2})\.\d+\s+'   # groupes 2-4 : heure:min:sec
        r'(\w+)\s+'                            # groupe 5  : niveau (ERROR, WARN…)
        r'(\d+)\s+---\s+'                     # groupe 6  : PID
        r'\[([^\]]+)\]\s+'                    # groupe 7  : thread
        r'(\S+)\s*:\s+'                       # groupe 8  : composant
        r'(?:\[user:([^\]]*)\]\s+)?'          # groupe 9  : userId    (optionnel)
        r'(?:\[email:([^\]]*)\]\s+)?'         # groupe 10 : email     (optionnel)
        r'(?:\[ip:([^\]]*)\]\s+)?'            # groupe 11 : ip        (optionnel)
        r'(?:\[trace:([^\]]*)\]\s+)?'         # groupe 12 : traceId   (optionnel)
        r'(?:\[([^\]]*)\]\s+)?'               # groupe 13 : requete   (optionnel)
        r'(.+)'                               # groupe 14 : message
    )

    for log in logs:
        ligne = log.strip()
        if not ligne:
            continue
        m = pattern.match(ligne)
        if m:
            niv      = m.group(5).strip().upper()
            niveau   = NIVEAU_MAP.get(niv, 'I')

            entries.append({
                'source':    'Spring Boot',
                'date':      m.group(1),
                'heure':     int(m.group(2)),
                'minute':    int(m.group(3)),
                'seconde':   int(m.group(4)),
                'niveau':    niveau,
                'pid':       int(m.group(6)),
                'thread':    m.group(7).strip(),
                'composant': m.group(8).strip(),
                'user_id':   (m.group(9)  or 'N/A').strip(),
                'email':     (m.group(10) or 'N/A').strip(),
                'ip':        (m.group(11) or 'N/A').strip(),
                'trace_id':  (m.group(12) or 'N/A').strip(),
                'requete':   (m.group(13) or '-').strip(),
                'message':   m.group(14).strip(),
            })
        else:
            rejetees += 1

    df = pd.DataFrame(entries)

    if not df.empty:
        valeurs_valides = set(NIVEAU_MAP.values())
        df['niveau'] = df['niveau'].apply(
            lambda x: x if x in valeurs_valides else 'I'
        )

    return df, rejetees


def parser_nginx(logs):
    """
    Parser les logs Nginx (format Combined Log Format).
    
    CORRECTIONS :
    - composant = path de la requête (ex: /api/automates/ligne5)
      → permet à frequence_composant de détecter les pics de volume
    - message enrichi avec mots-clés sémantiques selon le status HTTP
      → permet au TF-IDF de détecter les anomalies textuelles
    """
    entries  = []
    rejetees = 0

    pattern = re.compile(
        r'(\S+)\s+-\s+'                                                      # groupe 1  : IP
        r'(\S+)\s+'                                                          # groupe 2  : utilisateur
        r'\[(\d{2})/(\w{3})/(\d{4}):(\d{2}):(\d{2}):(\d{2})\s[^\]]+\]\s+' # groupes 3-8 : date/heure
        r'"(\w+)\s+(\S+)\s+HTTP/[\d.]+"\s+'                                 # groupes 9-10 : méthode + path
        r'(\d{3})\s+'                                                        # groupe 11 : status HTTP
        r'(\d+)'                                                             # groupe 12 : taille réponse
    )

    def status_vers_niveau(status):
        """Convertit un code HTTP en niveau de log."""
        if status >= 500: return 'E'
        if status >= 400: return 'W'
        return 'I'

    def enrichir_message(methode, path, status, taille):
        """
        Construit un message enrichi avec des mots-clés sémantiques
        selon le code HTTP, pour que le TF-IDF puisse détecter les anomalies.
        Les mots-clés importants (timeout, erreur…) sont placés AVANT le chemin
        afin d'être visibles même quand la colonne est tronquée.
        """
        # Préfixe sémantique selon le status — visible dès le début du message
        prefixe = ""

        if status in (408, 504):
            prefixe = "[timeout] "
        elif status in (502, 503):
            prefixe = "[indisponible] "
        elif status >= 500:
            prefixe = "[erreur serveur] "
        elif status == 403:
            prefixe = "[forbidden] "
        elif status == 404:
            prefixe = "[not found] "
        elif status == 413:
            prefixe = "[payload trop grand] "

        base = f"{prefixe}Requete HTTP {methode} {path} - Status: {status} - Taille: {taille} octets"

        # Mots-clés pour le TF-IDF (anomalie textuelle)
        if status >= 500 and status < 600:
            if status in (502, 503, 504):
                base += " erreur exception failed service indisponible"
            else:
                base += " erreur exception failed"
        elif status == 403:
            base += " forbidden"
        elif status == 404:
            base += " erreur"
        elif status == 413:
            base += " depasse"

        if status in (408, 504):
            base += " timeout"

        # Buffer overflow / requête anormalement longue (taille > 2000)
        if int(taille) > 2000:
            base += " anormale depasse"

        return base

    for log in logs:
        ligne = log.strip()
        if not ligne:
            continue
        m = pattern.match(ligne)
        if m:
            status  = int(m.group(11))
            taille  = m.group(12)
            methode = m.group(9)
            path    = m.group(10)
            mois    = MOIS_MAP.get(m.group(4), '01')
            date    = f"{m.group(5)}-{mois}-{m.group(3)}"

            entries.append({
                'source':    'Nginx',
                'date':      date,
                'heure':     int(m.group(6)),
                'minute':    int(m.group(7)),
                'seconde':   int(m.group(8)),
                'niveau':    status_vers_niveau(status),
                'pid':       0,
                'thread':    'nginx-worker',
                # ← FIX : composant = path pour que frequence_composant
                #          détecte les pics de volume par endpoint
                'composant': path,
                'user_id':   'N/A',
                'email':     'N/A',
                'ip':        m.group(1),
                'trace_id':  'N/A',
                'requete':   f"{methode} {path}",
                # ← FIX : message enrichi avec mots-clés sémantiques
                'message':   enrichir_message(methode, path, status, taille),
            })
        else:
            rejetees += 1

    return pd.DataFrame(entries), rejetees