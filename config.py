# ==============================================================
# CONFIGURATION GLOBALE
# ==============================================================

# Base de données
DB_PATH = "logs_anomalies.db"

# Vocabulaire métier (OCP) forcé pour réduire le bruit lexical
# Spring Boot + Nginx
MOTS_CRITIQUES = [
    # Spring Boot
    'timeout', 'communication', 'depasse', 'anormale',
    'pression', 'temperature', 'reset', 'delete',
    'exception', 'refused', 'failed', 'forbidden',
    'verification', 'demarrage', 'erreur',
    # Nginx / HTTP
    'error', 'critical', 'warning', 'automates',
    'capteurs', 'modbus', 'api', 'gateway',
    'unauthorized', 'internal', 'unavailable',
    'bad', 'request', 'nginx',
]

# Mapping des niveaux de log (Spring Boot -> Code court)
NIVEAU_MAP = {
    'TRACE': 'V', 'DEBUG': 'D', 'INFO': 'I',
    'WARN': 'W',  'WARNING': 'W',
    'ERROR': 'E', 'FATAL': 'F', 'CRITICAL': 'F',
}

# Mapping des mois (Nginx)
MOIS_MAP = {
    'Jan':'01','Feb':'02','Mar':'03','Apr':'04',
    'May':'05','Jun':'06','Jul':'07','Aug':'08',
    'Sep':'09','Oct':'10','Nov':'11','Dec':'12'
}