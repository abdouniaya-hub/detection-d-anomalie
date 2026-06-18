"""
theme_glass.py — Thème glassmorphism warm gray EXACT
Palette extraite de l'image de référence (RonDesignLab #09) :
  - Fond : #2a2520 warm anthracite
  - Cartes : rgba(255,255,255,0.07) + blur
  - Texte : #f5f2ed / rgba(245,242,237,0.45)
  - Bordures : rgba(255,255,255,0.10)
  - Pills actives : rgba(255,255,255,0.15)
"""

import streamlit as st
import base64
import os

# Image de fond active — modifier ce nom pour changer le background
BACKGROUND_IMAGE = "back1.jpg"


def inject_theme():
    """Injecte le CSS global du thème glassmorphism warm gray dans l'application Streamlit."""

    # Image de fond : BACKGROUND_IMAGE en priorité, puis fallbacks (bg.jpg, back1.jpeg…)
    bg_css = ""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mime_by_ext = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    bg_candidates = [BACKGROUND_IMAGE, "bg.jpg", "bg.jpeg", "bg.png", "back1.jpeg", "background.jpg"]
    bg_path = next(
        (os.path.join(base_dir, name) for name in bg_candidates if os.path.exists(os.path.join(base_dir, name))),
        None,
    )
    if bg_path:
        ext = os.path.splitext(bg_path)[1].lower()
        mime = mime_by_ext.get(ext, "image/jpeg")
        with open(bg_path, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode()
        bg_css = f"background-image: url('data:{mime};base64,{bg_b64}');"
    else:
        bg_css = """background: linear-gradient(
            135deg,
            #2f2a25 0%,
            #2a2520 40%,
            #322c27 70%,
            #26211d 100%
        );"""

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');

        html, body {{
            margin: 0; padding: 0;
        }}

        [data-testid="stAppViewContainer"] {{
            {bg_css}
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            min-height: 100vh !important;
        }}

        [data-testid="stAppViewContainer"]::before {{
            content: '';
            position: fixed;
            inset: 0;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            background: rgba(22, 18, 15, 0.50);
            z-index: 0;
            pointer-events: none;
        }}

        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {{
            background: transparent !important;
            position: relative;
            z-index: 1;
        }}

        /* SIDEBAR */
        [data-testid="stSidebar"] {{
            background: rgba(22, 18, 15, 0.75) !important;
            border-right: 0.5px solid rgba(255, 255, 255, 0.07) !important;
            backdrop-filter: blur(24px) !important;
            -webkit-backdrop-filter: blur(24px) !important;
        }}

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] span:not([data-testid="stIconMaterial"]):not([data-testid="stIconEmoji"]) {{
            color: rgba(245, 242, 237, 0.65) !important;
            font-family: 'DM Sans', 'Segoe UI', sans-serif !important;
        }}

        .sidebar-logo {{
            text-align: center;
            padding: 24px 0 16px;
        }}

        .sidebar-title {{
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.12em;
            color: rgba(245, 242, 237, 0.92) !important;
            margin-top: 8px;
        }}

        .sidebar-subtitle {{
            font-size: 11px;
            color: rgba(245, 242, 237, 0.38) !important;
            margin-top: 4px;
        }}

        .sidebar-footer {{
            font-size: 11px;
            color: rgba(245, 242, 237, 0.28) !important;
            text-align: center;
            padding: 16px 0;
            border-top: 0.5px solid rgba(255,255,255,0.05);
            margin-top: 24px;
        }}

        /* TEXTE GLOBAL */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: #f5f2ed !important;
            font-family: 'DM Sans', 'Segoe UI', sans-serif !important;
            font-weight: 500 !important;
        }}

        p, li,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span:not([data-testid="stIconMaterial"]):not([data-testid="stIconEmoji"]) {{
            color: rgba(245, 242, 237, 0.68) !important;
            font-family: 'DM Sans', 'Segoe UI', sans-serif !important;
        }}

        /* Icônes Material Streamlit — restaurer la police (sinon "upload" s'affiche en texte) */
        [data-testid="stIconMaterial"],
        [data-testid="stIconEmoji"],
        [data-testid="stSidebar"] [data-testid="stIconMaterial"],
        [data-testid="stFileUploader"] [data-testid="stIconMaterial"],
        [data-baseweb="button"] [data-testid="stIconMaterial"] {{
            font-family: "Material Symbols Rounded", sans-serif !important;
            font-weight: 400 !important;
            font-style: normal !important;
            font-feature-settings: "liga" !important;
            -webkit-font-feature-settings: "liga" !important;
            font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24 !important;
            text-transform: none !important;
            letter-spacing: normal !important;
            line-height: 1 !important;
            user-select: none !important;
            -webkit-font-smoothing: antialiased !important;
        }}

        /* Bouton Upload : masquer le libellé texte en doublon (l'icône suffit) */
        [data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] {{
            display: none !important;
        }}

        /* BANNIERE ACCUEIL */
        .welcome-banner {{
            background: rgba(255, 255, 255, 0.06);
            border: 0.5px solid rgba(255, 255, 255, 0.10);
            border-radius: 16px;
            padding: 22px 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}

        .welcome-banner h2 {{
            color: #f5f2ed !important;
            font-size: 18px !important;
            margin: 0 0 4px !important;
        }}

        .welcome-banner p {{
            color: rgba(245, 242, 237, 0.45) !important;
            font-size: 13px !important;
            margin: 0 !important;
        }}

        .date-badge {{
            background: rgba(255, 255, 255, 0.08);
            border: 0.5px solid rgba(255, 255, 255, 0.12);
            border-radius: 10px;
            padding: 8px 16px;
            font-size: 13px;
            color: rgba(245, 242, 237, 0.55) !important;
        }}

        /* CARTES KPI */
        .kpi-card {{
            background: rgba(255, 255, 255, 0.07) !important;
            border: 0.5px solid rgba(255, 255, 255, 0.10) !important;
            border-radius: 16px !important;
            padding: 22px 20px !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            transition: border-color 0.2s ease, background 0.2s ease !important;
        }}

        .kpi-card:hover {{
            background: rgba(255, 255, 255, 0.10) !important;
            border-color: rgba(255, 255, 255, 0.18) !important;
        }}

        .kpi-label {{
            font-size: 10px !important;
            letter-spacing: 0.11em !important;
            text-transform: uppercase !important;
            color: rgba(245, 242, 237, 0.35) !important;
            margin-bottom: 10px !important;
        }}

        .kpi-value {{
            font-size: 32px !important;
            font-weight: 500 !important;
            color: #f5f2ed !important;
            line-height: 1 !important;
        }}

        .kpi-sub {{
            font-size: 11px !important;
            color: rgba(245, 242, 237, 0.28) !important;
            margin-top: 6px !important;
        }}

        /* SECTION TITRE */
        .section-titre {{
            font-size: 11px !important;
            font-weight: 500 !important;
            letter-spacing: 0.10em !important;
            text-transform: uppercase !important;
            color: rgba(245, 242, 237, 0.35) !important;
            margin-bottom: 14px !important;
            padding-bottom: 10px !important;
            border-bottom: 0.5px solid rgba(255, 255, 255, 0.06) !important;
        }}

        /* ALERTES */
        div[data-testid="stAlert"] {{
            border-radius: 10px !important;
            border-left: none !important;
        }}

        /* BOUTONS */
        .stButton > button {{
            background: rgba(255, 255, 255, 0.08) !important;
            border: 0.5px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 10px !important;
            color: rgba(245, 242, 237, 0.80) !important;
            font-family: 'DM Sans', 'Segoe UI', sans-serif !important;
            font-size: 13px !important;
            transition: all 0.2s ease !important;
        }}

        .stButton > button:hover {{
            background: rgba(255, 255, 255, 0.13) !important;
            border-color: rgba(255, 255, 255, 0.24) !important;
            color: #f5f2ed !important;
        }}

        /* INPUTS */
        .stTextInput input {{
            background: rgba(255, 255, 255, 0.07) !important;
            border: 0.5px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 10px !important;
            color: rgba(245, 242, 237, 0.88) !important;
        }}

        [data-baseweb="select"] {{
            background: rgba(255, 255, 255, 0.07) !important;
        }}

        [data-baseweb="select"] div,
        [data-baseweb="select"] span {{
            background: transparent !important;
            color: rgba(245, 242, 237, 0.88) !important;
        }}

        /* DATAFRAME */
        [data-testid="stDataFrame"] {{
            background: rgba(255, 255, 255, 0.04) !important;
            border: 0.5px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }}

        /* EXPANDER */
        .streamlit-expanderHeader {{
            background: rgba(255, 255, 255, 0.05) !important;
            border: 0.5px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 10px !important;
            color: rgba(240, 237, 232, 0.70) !important;
        }}

        .streamlit-expanderContent {{
            background: rgba(255, 255, 255, 0.03) !important;
            border: 0.5px solid rgba(255, 255, 255, 0.06) !important;
            border-top: none !important;
            border-radius: 0 0 10px 10px !important;
        }}

        /* METRIQUES */
        [data-testid="stMetricLabel"] p {{
            color: rgba(245, 242, 237, 0.35) !important;
            font-size: 11px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.09em !important;
        }}

        [data-testid="stMetricValue"] {{
            color: #f5f2ed !important;
            font-size: 26px !important;
            font-weight: 500 !important;
        }}

        /* PROGRESS BAR */
        .stProgress > div > div {{
            background: rgba(210, 200, 188, 0.60) !important;
            border-radius: 4px !important;
        }}

        .stProgress > div {{
            background: rgba(255, 255, 255, 0.07) !important;
            border-radius: 4px !important;
        }}

        /* FILE UPLOADER */
        [data-testid="stFileUploader"] {{
            background: rgba(255, 255, 255, 0.04) !important;
            border: 0.5px dashed rgba(255, 255, 255, 0.14) !important;
            border-radius: 14px !important;
        }}

        /* MULTISELECT TAGS */
        [data-baseweb="tag"] {{
            background: rgba(255, 255, 255, 0.12) !important;
            border-radius: 8px !important;
        }}

        [data-baseweb="tag"] span {{
            color: rgba(245, 242, 237, 0.88) !important;
        }}

        /* SCROLLBAR */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: rgba(255,255,255,0.03); }}
        ::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.12); border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.20); }}
        </style>
        """,
        unsafe_allow_html=True,
    )