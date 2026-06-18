import sqlite3
import pandas as pd
import os
from datetime import datetime
from config import DB_PATH

def init_db():
    """Crée la base de données et les tables si elles n'existent pas."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    TEXT,
            inserted_at   TEXT,
            source        TEXT,
            date          TEXT,
            heure         INTEGER,
            minute        INTEGER,
            seconde       INTEGER,
            niveau        TEXT,
            composant     TEXT,
            ip            TEXT,
            user_id       TEXT,
            email         TEXT,
            trace_id      TEXT,
            requete       TEXT,
            message       TEXT,
            anomalie      INTEGER,
            score         REAL
        )
    """)
    conn.commit()
    conn.close()

def sauvegarder_logs(df: pd.DataFrame, session_id: str):
    """Insère tous les logs du DataFrame dans la base SQLite."""
    if df.empty:
        return 0
    conn = sqlite3.connect(DB_PATH)
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cols = ['source','date','heure','minute','seconde','niveau','composant',
            'ip','user_id','email','trace_id','requete','message','anomalie','score']
    
    # Garder seulement les colonnes présentes dans df
    cols_ok = [c for c in cols if c in df.columns]
    df_save = df[cols_ok].copy()
    df_save['session_id']  = session_id
    df_save['inserted_at'] = now
    
    # Remplir les colonnes absentes
    for col in cols:
        if col not in df_save.columns:
            df_save[col] = None
            
    df_save.to_sql('logs', conn, if_exists='append', index=False)
    conn.close()
    return len(df_save)

def charger_historique(limit=500):
    """Charge les derniers logs depuis la base."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT * FROM logs ORDER BY id DESC LIMIT {limit}",
        conn
    )
    conn.close()
    return df

def stats_db():
    """Retourne des statistiques globales sur la base."""
    if not os.path.exists(DB_PATH):
        return {}
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM logs")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM logs WHERE anomalie = -1")
    anomalies = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT session_id) FROM logs")
    sessions = c.fetchone()[0]
    c.execute("SELECT inserted_at FROM logs ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    last = row[0] if row else "—"
    conn.close()
    return {"total": total, "anomalies": anomalies, "sessions": sessions, "last": last}

def vider_db():
    """Supprime tous les enregistrements de la base."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM logs")
    conn.commit()
    conn.close()