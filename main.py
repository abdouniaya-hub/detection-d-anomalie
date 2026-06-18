from datetime import datetime
import streamlit as st
import pandas as pd
import os

from config import DB_PATH
from database.db_manager import init_db, sauvegarder_logs, stats_db, charger_historique, vider_db
from backend.parser import lire_fichier, parser_springboot, parser_nginx
from backend.feature_extraction import extraire_features
from backend.ml_model import (
    detecter_anomalies,
    detecter_ensemble,
    extraire_mots_techniques_tfidf,
    evaluer_performance_modele,
    afficher_graphique_apprentissage,
    evaluer_modeles_ensemble,
    benchmarker_modeles,
)
from backend.xgboost_model import (
    entrainer_xgboost,
    predire_xgboost,
    get_stats_xgboost,
)
from frontend.dashboard import afficher_dashboard
from validation.human_validation import (
    afficher_interface_validation,
    collecter_feedbacks_pour_xgboost,
    afficher_statut_xgboost,
)
from theme_glass import inject_theme


# ==============================================================
# CONFIG PAGE
# ==============================================================
st.set_page_config(
    page_title="Log Analytics — OCP El Jadida",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()
init_db()

for key in ("modele_xgb", "scaler_xgb", "vectorizer_xgb"):
    if key not in st.session_state:
        st.session_state[key] = None

# ==============================================================
# SIDEBAR
# ==============================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:3rem;">📊</div>
        <div class="sidebar-title">LOG ANALYTICS</div>
        <div class="sidebar-subtitle">Détection d'anomalies</div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "",
        ["🏠 Accueil", "📁 Upload", "📊 Dashboard", "🚨 Anomalies", "📋 Logs", "✅ Validation", "🔬 Évaluation", "🗄️ DB"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-footer">PFE OCP El Jadida 2026</div>', unsafe_allow_html=True)

# ==============================================================
# BANNIÈRE (cachée sur l'onglet Accueil)
# ==============================================================
if menu != "🏠 Accueil":
    now = datetime.now()
    st.markdown(f"""
<div class="welcome-banner">
    <div class="welcome-text">
        <h2>👋 Hello Aya !</h2>
        <p>Bienvenue sur votre plateforme de détection d'anomalies</p>
    </div>
    <div class="date-badge">📅 {now.strftime("%A %d %B %Y")}</div>
</div>
""", unsafe_allow_html=True)

# ==============================================================
# CONSTANTES
# ==============================================================
NIVEAUX_DISPO = ["I", "W", "E"]
SOURCES_DISPO = ["Spring Boot", "Nginx"]

# ==============================================================
# ONGLET ACCUEIL — fond seul, aucun contenu
# ==============================================================
if menu == "🏠 Accueil":
    accueil_html = (
        "<style>[data-testid='stMainBlockContainer']{padding-top:0!important;}</style>"
        "<div style='position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:10;text-align:center;'>"
        "<div style='font-size:11px;letter-spacing:0.18em;text-transform:uppercase;"
        "color:rgba(245,242,237,0.45);margin-bottom:14px;font-family:DM Sans,sans-serif;'>"
        "Plateforme de détection d'anomalies</div>"
        "<div style='font-size:52px;font-weight:600;color:#f5f2ed;line-height:1.1;"
        "font-family:DM Sans,sans-serif;letter-spacing:-0.5px;'>Log Analytics</div>"
        "<div style='font-size:28px;font-weight:300;color:rgba(245,242,237,0.55);"
        "font-family:DM Sans,sans-serif;margin-top:4px;'>FST AL HOCEIMA</div>"
        "<div style='width:60px;height:2px;background:rgba(245,242,237,0.25);"
        "margin:22px auto;border-radius:2px;'></div>"
        "<div style='font-size:13px;color:rgba(245,242,237,0.38);"
        "font-family:DM Sans,sans-serif;font-weight:300;'>"
        "Détection intelligente &nbsp;·&nbsp; Isolation Forest &nbsp;·&nbsp; LOF &nbsp;·&nbsp; Random Forest"
        "</div></div>"
    )
    st.markdown(accueil_html, unsafe_allow_html=True)
    st.stop()

# ==============================================================
# ONGLET UPLOAD
# ==============================================================
if menu == "📁 Upload":
    st.markdown("### 📁 Chargement des fichiers logs")
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        upload_sb = st.file_uploader("📂 Logs Spring Boot", type=["log", "txt"], key="upload_sb")
    with col_up2:
        upload_ng = st.file_uploader("📂 Logs Nginx", type=["log", "txt"], key="upload_ng")

    st.markdown("---")
    st.markdown("### ⚙️ Paramètres d'analyse")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        contamination = st.slider("🎯 Sensibilité anomalies", 0.01, 0.20, 0.05, 0.01, key="contamination_slider")
    with col_f2:
        sources_selectionnees = st.multiselect("📌 Sources", SOURCES_DISPO, default=SOURCES_DISPO, key="sources_select")
    with col_f3:
        niveaux_selectionnes = st.multiselect("🏷️ Niveaux", NIVEAUX_DISPO, default=NIVEAUX_DISPO, key="niveaux_select")

    # Lecture unique des fichiers → bytes dans session_state
    if upload_sb is not None:
        content = upload_sb.read()
        if content:
            st.session_state["bytes_sb"] = content
            st.session_state["name_sb"] = upload_sb.name
    if upload_ng is not None:
        content = upload_ng.read()
        if content:
            st.session_state["bytes_ng"] = content
            st.session_state["name_ng"] = upload_ng.name

    if "bytes_sb" in st.session_state or "bytes_ng" in st.session_state:
        st.success("✅ Fichiers chargés ! Naviguez vers un autre onglet.")

    st.session_state["contamination_value"] = contamination
    st.session_state["sources_value"] = sources_selectionnees
    st.session_state["niveaux_value"] = niveaux_selectionnes

else:
    contamination = st.session_state.get("contamination_value", 0.05)
    sources_selectionnees = st.session_state.get("sources_value", SOURCES_DISPO)
    niveaux_selectionnes = st.session_state.get("niveaux_value", NIVEAUX_DISPO)

# ==============================================================
# ÉCRITURE FICHIERS TEMPORAIRES
# ==============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

path_sb = os.path.join(BASE_DIR, "Echantillon_logs_springboot.log")
path_ng = os.path.join(BASE_DIR, "Echantillon_logs_nginx.log")
name_sb = "default_sb"
name_ng = "default_ng"

if "bytes_sb" in st.session_state:
    path_sb = os.path.join(BASE_DIR, "temp_sb.log")
    name_sb = st.session_state.get("name_sb", "uploaded_sb")
    with open(path_sb, "wb") as f:
        f.write(st.session_state["bytes_sb"])

if "bytes_ng" in st.session_state:
    path_ng = os.path.join(BASE_DIR, "temp_ng.log")
    name_ng = st.session_state.get("name_ng", "uploaded_ng")
    with open(path_ng, "wb") as f:
        f.write(st.session_state["bytes_ng"])

sb_existe = os.path.exists(path_sb)
ng_existe = os.path.exists(path_ng)

# ==============================================================
# TRAITEMENT
# ==============================================================
if sb_existe or ng_existe:
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    @st.cache_data(ttl=300, show_spinner="⏳ Analyse en cours...")
    def traiter_logs(path_sb, path_ng, contamination, sb_existe, ng_existe, name_sb, name_ng):
        dfs = []
        if sb_existe and os.path.exists(path_sb):
            logs_sb, _ = parser_springboot(lire_fichier(path_sb))
            if not logs_sb.empty:
                dfs.append(logs_sb)
        if ng_existe and os.path.exists(path_ng):
            logs_ng, _ = parser_nginx(lire_fichier(path_ng))
            if not logs_ng.empty:
                dfs.append(logs_ng)
        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)
        if df.empty:
            return None
        df = extraire_features(df)
        df = detecter_ensemble(df, contamination)        # ← 3 modèles + vote majoritaire
        df = extraire_mots_techniques_tfidf(df, top_n=5)
        return df

    from backend.ml_model import MAX_LIGNES_ANALYSE
    df = traiter_logs(path_sb, path_ng, contamination, sb_existe, ng_existe, name_sb, name_ng)

    if df is not None and not df.empty:
        # Filtrage
        df_filtre = df[
            df["niveau"].isin(niveaux_selectionnes) &
            df["source"].isin(sources_selectionnees)
        ]
        anomalies = df_filtre[df_filtre["anomalie"] == -1]
        normaux   = df_filtre[df_filtre["anomalie"] == 1]

        def cols_ok(d, souhaites):
            return [c for c in souhaites if c in d.columns]

        # ── ONGLETS ───────────────────────────────────────────────────────

        if menu in ("📁 Upload", "🏠 Accueil"):
            st.markdown("---")
            st.markdown("### 💾 Enregistrement dans la base de données")
            cle_save = f"sauvegarde_{st.session_state.session_id}"
            deja = st.session_state.get(cle_save, False)
            col_s1, col_s2 = st.columns([3, 1])
            with col_s1:
                st.info(f"📊 **{len(df)}** logs analysés — **{len(anomalies)}** anomalies détectées")
            with col_s2:
                if deja:
                    st.button("✅ Sauvegardé", disabled=True)
                else:
                    if st.button("💾 Sauvegarder", type="primary"):
                        nb = sauvegarder_logs(df, st.session_state.session_id)
                        st.session_state[cle_save] = True
                        st.success(f"Enregistré ! ({nb} logs)")
                        st.rerun()

        elif menu == "📊 Dashboard":
            afficher_dashboard(df, df_filtre, anomalies, normaux)

        elif menu == "🚨 Anomalies":
            st.markdown("### 🚨 Liste des anomalies détectées")
            if anomalies.empty:
                st.success("✅ Aucune anomalie détectée !")
            else:
                nb_ng = int((anomalies["source"] == "Nginx").sum())
                nb_sb = int((anomalies["source"] == "Spring Boot").sum())
                nb_total = len(anomalies)

                # ── Ligne 1 : boutons de filtre source ────────────────────
                st.markdown(
                    """
                    <div style='font-size:13px;color:rgba(245,242,237,0.55);
                                margin-bottom:8px;letter-spacing:0.05em;'>
                        🔎 Filtrer par source
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col_b0, col_b1, col_b2, col_spacer = st.columns([1.4, 1.4, 1.6, 4])
                with col_b0:
                    btn_toutes = st.button(
                        f"🌐 Toutes ({nb_total})",
                        key="anom_btn_toutes",
                        use_container_width=True,
                        type="primary" if st.session_state.get("anom_source_actif", "Toutes") == "Toutes" else "secondary",
                    )
                with col_b1:
                    btn_nginx = st.button(
                        f"🔵 Nginx ({nb_ng})",
                        key="anom_btn_nginx",
                        use_container_width=True,
                        type="primary" if st.session_state.get("anom_source_actif") == "Nginx" else "secondary",
                    )
                with col_b2:
                    btn_sb = st.button(
                        f"🟠 SpringBoot ({nb_sb})",
                        key="anom_btn_sb",
                        use_container_width=True,
                        type="primary" if st.session_state.get("anom_source_actif") == "Spring Boot" else "secondary",
                    )

                # Mise à jour de la source active selon le bouton cliqué
                if btn_toutes:
                    st.session_state["anom_source_actif"] = "Toutes"
                    st.rerun()
                if btn_nginx:
                    st.session_state["anom_source_actif"] = "Nginx"
                    st.rerun()
                if btn_sb:
                    st.session_state["anom_source_actif"] = "Spring Boot"
                    st.rerun()

                source_choisie = st.session_state.get("anom_source_actif", "Toutes")

                # ── Ligne 2 : barre de recherche + toggle périmètre ───────
                col_search, col_toggle = st.columns([4, 2])
                with col_search:
                    recherche_anom = st.text_input(
                        "🔍 Rechercher dans les messages",
                        placeholder="ex : timeout, temperature, 500, GET /api...",
                        key="anom_recherche"
                    )
                with col_toggle:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                    chercher_tous = st.toggle(
                        "🔎 Chercher dans TOUS les logs",
                        value=False,
                        key="anom_toggle_tous",
                        help="Activez pour chercher dans tous les logs, pas seulement les anomalies détectées par le ML"
                    )

                # ── Appliquer les filtres ──────────────────────────────────
                # Périmètre : anomalies seules ou tous les logs
                base_recherche = df_filtre if chercher_tous else anomalies

                anom_affichees = base_recherche if source_choisie == "Toutes" \
                    else base_recherche[base_recherche["source"] == source_choisie]

                if recherche_anom.strip():
                    masque = (
                        anom_affichees["message"].str.contains(
                            recherche_anom, case=False, na=False, regex=False
                        ) |
                        anom_affichees["composant"].str.contains(
                            recherche_anom, case=False, na=False, regex=False
                        )
                    )
                    anom_affichees = anom_affichees[masque]

                # ── Résultat ──────────────────────────────────────────────
                nb_res = len(anom_affichees)
                LABEL_SOURCE = {
                    "Toutes": "Toutes",
                    "Nginx": "Nginx",
                    "Spring Boot": "SpringBoot",
                }
                label_affiche = LABEL_SOURCE.get(source_choisie, source_choisie)
                perimetre_label = "tous les logs" if chercher_tous else "anomalies ML"

                if recherche_anom.strip():
                    st.caption(f"{nb_res} résultat(s) pour « {recherche_anom} » dans {label_affiche} ({perimetre_label})")
                else:
                    st.caption(f"{nb_res} entrée(s) affichée(s) — source : {label_affiche} — {perimetre_label}")

                if anom_affichees.empty:
                    if recherche_anom.strip() and not chercher_tous:
                        st.warning(
                            f"⚠️ Aucune anomalie ML contient « {recherche_anom} ». "
                            f"Activez **🔎 Chercher dans TOUS les logs** pour chercher dans l'ensemble des logs."
                        )
                    else:
                        st.info("Aucun résultat ne correspond à votre recherche.")
                else:
                    cols_anom = ["source", "date", "heure", "niveau",
                                 "composant", "message", "score_if", "nb_votes"]
                    # Colonne anomalie visible si on cherche dans tous les logs
                    if chercher_tous:
                        cols_anom = ["source", "date", "heure", "niveau",
                                     "composant", "message", "anomalie", "score_if", "nb_votes"]
                    st.dataframe(
                        anom_affichees[cols_ok(anom_affichees, cols_anom)],
                        use_container_width=True
                    )

        elif menu == "📋 Logs":
            st.markdown("### 📋 Tous les logs")
            recherche = st.text_input("🔍 Rechercher dans les messages", "")
            df_affiche = (
                df_filtre[df_filtre["message"].str.contains(recherche, case=False, na=False)]
                if recherche else df_filtre
            )
            souhaites = ["source", "date", "heure", "niveau", "message", "anomalie"]
            st.dataframe(df_affiche[cols_ok(df_affiche, souhaites)], use_container_width=True)

        elif menu == "✅ Validation":
            st.markdown("### ✅ Validation humaine des anomalies")
            col_titre, col_statut = st.columns([3, 1])
            with col_statut:
                afficher_statut_xgboost()

            if anomalies.empty:
                st.success("✅ Aucune anomalie à valider !")
            else:
                st.session_state["anomalies_a_valider"] = anomalies.copy()
                afficher_interface_validation(anomalies)

                st.markdown("---")
                st.markdown("### 🤖 Apprentissage supervisé (XGBoost)")
                df_feedback = collecter_feedbacks_pour_xgboost()

                if len(df_feedback) >= 5:
                    st.success(f"✅ {len(df_feedback)} validations — entraînement XGBoost...")
                    modele, scaler, vectorizer = entrainer_xgboost(df_feedback)
                    if modele is not None:
                        st.session_state["modele_xgb"]     = modele
                        st.session_state["scaler_xgb"]     = scaler
                        st.session_state["vectorizer_xgb"] = vectorizer
                        stats = get_stats_xgboost(modele)
                        if stats:
                            st.markdown(f"**Modèle :** {stats['type']} — {stats['n_estimators']} arbres, profondeur {stats['max_depth']}")

                        idx_valides  = set(df_feedback.index)
                        non_validees = anomalies.drop(
                            index=[i for i in idx_valides if i in anomalies.index],
                            errors="ignore"
                        ).head(10)
                        if not non_validees.empty:
                            preds, probs = predire_xgboost(non_validees, modele, scaler, vectorizer)
                            if preds is not None:
                                st.markdown("**🔮 Prédictions XGBoost sur anomalies restantes :**")
                                for i, (_, row) in enumerate(non_validees.iterrows()):
                                    if i >= len(preds):
                                        break
                                    conf = probs[i] if probs is not None else 0.5
                                    msg  = str(row.get("message", ""))[:80]
                                    if preds[i] == 1:
                                        st.markdown(f"⚠️ **Anomalie probable** ({conf*100:.0f}%) : {msg}...")
                                    else:
                                        st.markdown(f"✅ **Faux positif probable** ({conf*100:.0f}%) : {msg}...")
                else:
                    st.info(f"ℹ️ {len(df_feedback)}/5 validations reçues.")

                if not df_feedback.empty:
                    st.markdown("---")
                    st.markdown("### 📊 Récapitulatif")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("✅ Confirmées", int((df_feedback["label"] == 1).sum()))
                    with c2:
                        st.metric("❌ Rejetées", int((df_feedback["label"] == 0).sum()))

        elif menu == "🔬 Évaluation":
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
            st.markdown("### 📊 Évaluation des performances")

            # ── Onglets internes ──────────────────────────────────────────
            tab_ens, tab_xgb = st.tabs(["🤖 Modèles Ensemble (IF / LOF / RF)", "🚀 XGBoost supervisé"])

            # ── TAB 1 : Évaluation comparative des 3 modèles ─────────────
            with tab_ens:
                st.markdown("#### Résultats des 3 modèles de détection non-supervisée")

                if df_filtre.empty:
                    st.info("Aucune donnée filtrée à évaluer.")
                else:
                    cols_modeles = ["anomalie_if", "anomalie_lof", "anomalie_rf", "nb_votes"]
                    if not all(c in df_filtre.columns for c in cols_modeles):
                        st.warning("Les colonnes des 3 modèles ne sont pas disponibles. Relancez l'analyse.")
                    else:
                        eval_res = evaluer_modeles_ensemble(df_filtre)
                        nb_if  = int((df_filtre["anomalie_if"]  == -1).sum())
                        nb_lof = int((df_filtre["anomalie_lof"] == -1).sum())
                        nb_rf  = int((df_filtre["anomalie_rf"]  == -1).sum())
                        nb_ens = int((df_filtre["anomalie"]     == -1).sum())
                        total_f = len(df_filtre)

                        # ── KPI compteurs ─────────────────────────────────
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("🌲 Isolation Forest", nb_if,
                                      f"{nb_if/total_f*100:.1f}% anomalies")
                        with c2:
                            st.metric("📍 LOF", nb_lof,
                                      f"{nb_lof/total_f*100:.1f}% anomalies")
                        with c3:
                            st.metric("🌳 Random Forest", nb_rf,
                                      f"{nb_rf/total_f*100:.1f}% anomalies")
                        with c4:
                            st.metric("🗳️ Vote Ensemble", nb_ens,
                                      f"{nb_ens/total_f*100:.1f}% — résultat final")

                        st.markdown("---")

                        # ── 1. Tableau comparatif Précision/Rappel/F1 ─────
                        st.markdown("#### 📋 Tableau comparatif — Précision / Rappel / F1 / Kappa / MCC")
                        if eval_res:
                            mdf = eval_res["metrics_df"].copy()
                            fmt_cols = ["Précision","Rappel","F1-Score","Accuracy"]
                            mdf_pct = mdf[fmt_cols].map(lambda x: f"{x*100:.1f}%")
                            # Kappa et MCC en valeur brute (pas de %)
                            for col in ["Kappa", "MCC"]:
                                if col in mdf.columns:
                                    mdf_pct[col] = mdf[col].map(lambda x: f"{x:.3f}")
                            mdf_pct["Anomalies détectées"] = mdf["Anomalies"].astype(str)
                            st.dataframe(mdf_pct, use_container_width=True)
                        st.markdown("---")

                        # ── 1b. Temps + mémoire ───────────────────────────
                        st.markdown("#### ⏱️ Temps d'entraînement & inférence — Consommation mémoire")
                        if 'perf_bench' not in st.session_state:
                            with st.spinner("Benchmark en cours sur échantillon..."):
                                st.session_state['perf_bench'] = benchmarker_modeles(df_filtre, contamination)
                        perf = st.session_state['perf_bench'].copy()
                        if perf:
                            n_ech = perf.pop('_n_echantillon', len(df_filtre))
                            rows_perf = []
                            for nom, v in perf.items():
                                rows_perf.append({
                                    "Modèle":           nom,
                                    "Train (s)":        v["train_s"],
                                    "Inférence (s)":    v["infer_s"],
                                    "Mém. train (KB)":  v["mem_train_kb"],
                                    "Mém. inférence (KB)": v["mem_infer_kb"],
                                })
                            df_perf = pd.DataFrame(rows_perf).set_index("Modèle")
                            st.dataframe(df_perf, use_container_width=True)
                            st.caption(f"Mesuré sur un échantillon de {n_ech} logs (représentatif du dataset complet)")

                            # Mini graphique barres temps
                            fig_t, axes = plt.subplots(1, 2, figsize=(10, 3.5))
                            fig_t.patch.set_alpha(0.0)
                            noms  = list(perf.keys())
                            t_tr  = [perf[n]["train_s"]        for n in noms]
                            t_inf = [perf[n]["infer_s"]         for n in noms]
                            m_tr  = [perf[n]["mem_train_kb"]    for n in noms]
                            m_inf = [perf[n]["mem_infer_kb"]    for n in noms]
                            x_p   = np.arange(len(noms))
                            w_p   = 0.35

                            # Temps
                            axes[0].bar(x_p - w_p/2, t_tr,  w_p, label="Entraînement", color="#c8b48a", alpha=0.85)
                            axes[0].bar(x_p + w_p/2, t_inf, w_p, label="Inférence",    color="#70b890", alpha=0.85)
                            axes[0].set_xticks(x_p); axes[0].set_xticklabels(noms, fontsize=8)
                            axes[0].set_ylabel("Secondes"); axes[0].set_title("Temps (s)")
                            axes[0].legend(fontsize=8, framealpha=0.15, labelcolor="white")
                            axes[0].spines["top"].set_visible(False)
                            axes[0].spines["right"].set_visible(False)

                            # Mémoire
                            axes[1].bar(x_p - w_p/2, m_tr,  w_p, label="Entraînement", color="#c8983a", alpha=0.85)
                            axes[1].bar(x_p + w_p/2, m_inf, w_p, label="Inférence",    color="#d06050", alpha=0.85)
                            axes[1].set_xticks(x_p); axes[1].set_xticklabels(noms, fontsize=8)
                            axes[1].set_ylabel("KB"); axes[1].set_title("Mémoire (KB)")
                            axes[1].legend(fontsize=8, framealpha=0.15, labelcolor="white")
                            axes[1].spines["top"].set_visible(False)
                            axes[1].spines["right"].set_visible(False)

                            plt.tight_layout()
                            st.pyplot(fig_t, use_container_width=True)
                            plt.close(fig_t)
                        else:
                            st.warning("Impossible de calculer le benchmark sur ces données.")
                        st.markdown("---")

                        # ── 2. Graphique comparatif barres groupées ───────
                        st.markdown("#### 📊 Graphique de comparaison des performances")
                        if eval_res:
                            mdf_raw  = eval_res["metrics_df"]
                            modeles  = mdf_raw.index.tolist()
                            met_keys = ["Précision", "Rappel", "F1-Score"]
                            bar_cols = ["#c8b48a", "#70b890", "#d06050"]
                            x = np.arange(len(modeles))
                            w = 0.25
                            fig_cmp, ax_cmp = plt.subplots(figsize=(10, 4.5))
                            fig_cmp.patch.set_alpha(0.0)
                            for i, met in enumerate(met_keys):
                                vals = mdf_raw[met].values
                                bars = ax_cmp.bar(x + i*w, vals, w,
                                                  label=met, color=bar_cols[i],
                                                  alpha=0.85, zorder=3)
                                for bar, v in zip(bars, vals):
                                    ax_cmp.text(
                                        bar.get_x() + bar.get_width()/2,
                                        bar.get_height() + 0.015,
                                        f"{v*100:.0f}%",
                                        ha="center", va="bottom",
                                        fontsize=8, color="white")
                            ax_cmp.set_xticks(x + w)
                            ax_cmp.set_xticklabels(modeles, fontsize=9)
                            ax_cmp.set_ylim(0, 1.18)
                            ax_cmp.set_ylabel("Score")
                            ax_cmp.legend(loc="upper right", fontsize=9,
                                          framealpha=0.15, labelcolor="white")
                            ax_cmp.spines["top"].set_visible(False)
                            ax_cmp.spines["right"].set_visible(False)
                            plt.tight_layout()
                            st.pyplot(fig_cmp, use_container_width=True)
                            plt.close(fig_cmp)
                        st.markdown("---")

                        # ── 3. Réduction des alertes par le vote ──────────
                        st.markdown("#### 🔕 Réduction des alertes par le vote d'ensemble")
                        if eval_res:
                            red = eval_res["reduction"]
                            rc1, rc2, rc3, rc4 = st.columns(4)
                            with rc1:
                                st.metric("🌲 IF seul", red["nb_if"],
                                          f"−{red['reduction_vs_if']}% vs ensemble")
                            with rc2:
                                st.metric("📍 LOF seul", red["nb_lof"],
                                          f"−{red['reduction_vs_lof']}% vs ensemble")
                            with rc3:
                                st.metric("🌳 RF seul", red["nb_rf"],
                                          f"−{red['reduction_vs_rf']}% vs ensemble")
                            with rc4:
                                st.metric("🗳️ Ensemble final", red["nb_ensemble"],
                                          f"−{red['reduction_vs_union']}% vs union")
                            st.caption(
                                f"Union brute des 3 modèles : **{red['nb_union']}** alertes  →  "
                                f"Vote ensemble : **{red['nb_ensemble']}** alertes  →  "
                                f"Réduction : **{red['reduction_vs_union']}%** de faux positifs éliminés"
                            )
                        st.markdown("---")

                        # ── 4. Matrices de confusion ───────────────────────
                        st.markdown("#### 🔲 Matrices de confusion (référence = Vote Ensemble)")
                        if eval_res:
                            conf = eval_res["confusion"]
                            col_m1, col_m2, col_m3 = st.columns(3)
                            for col_m, (nom_mod, vals) in zip(
                                [col_m1, col_m2, col_m3], conf.items()
                            ):
                                with col_m:
                                    st.markdown(f"**{nom_mod}**")
                                    tp, fp = vals["TP"], vals["FP"]
                                    fn, tn = vals["FN"], vals["TN"]
                                    fig_cm, ax_cm = plt.subplots(figsize=(3.2, 2.8))
                                    fig_cm.patch.set_alpha(0.0)
                                    mat = np.array([[tp, fp], [fn, tn]])
                                    ax_cm.imshow(mat, cmap="YlOrBr", aspect="auto")
                                    ax_cm.set_xticks([0, 1])
                                    ax_cm.set_yticks([0, 1])
                                    ax_cm.set_xticklabels(["Prédit +", "Prédit −"], fontsize=8)
                                    ax_cm.set_yticklabels(["Réel +", "Réel −"], fontsize=8)
                                    for (r, c_), lbl, val in [
                                        ((0,0),"TP",tp),((0,1),"FP",fp),
                                        ((1,0),"FN",fn),((1,1),"TN",tn)
                                    ]:
                                        ax_cm.text(c_, r, f"{lbl}\n{val}",
                                                   ha="center", va="center",
                                                   fontsize=10, fontweight="bold",
                                                   color="white")
                                    ax_cm.set_title(nom_mod, fontsize=8, pad=4)
                                    plt.tight_layout()
                                    st.pyplot(fig_cm, use_container_width=True)
                                    plt.close(fig_cm)
                        st.markdown("---")

                        # ── 5. Synthèse justificative ─────────────────────
                        st.markdown("#### 📝 Synthèse — Justification du vote d'ensemble")
                        if eval_res:
                            red    = eval_res["reduction"]
                            mdf_r  = eval_res["metrics_df"]
                            f1_if  = mdf_r.loc["Isolation Forest","F1-Score"]
                            f1_lof = mdf_r.loc["LOF","F1-Score"]
                            f1_rf  = mdf_r.loc["Random Forest","F1-Score"]
                            best   = max([("Isolation Forest",f1_if),
                                          ("LOF",f1_lof),
                                          ("Random Forest",f1_rf)],
                                         key=lambda x: x[1])

                            # Accord inter-modèles
                            agr = eval_res.get("agreement", {})
                            kappa_txt = (f"Kappa moyen inter-modèles : **{agr['kappa_mean']:.3f}**"
                                         if agr.get("kappa_mean") is not None
                                         else "")

                            st.markdown(f"""
<div style="background:rgba(200,180,138,0.08);border-left:3px solid #c8b48a;
            padding:16px 20px;border-radius:6px;font-size:14px;
            color:rgba(245,242,237,0.82);line-height:1.8;">

<b>Pourquoi le vote d'ensemble ?</b><br><br>
Les trois modèles non-supervisés sont <b>indépendants</b> et ont des forces complémentaires :<br>
• <b>Isolation Forest</b> — détecte les points globalement isolés par arbres aléatoires (F1 = {f1_if*100:.1f}%)<br>
• <b>LOF</b> — repère les anomalies locales par densité de voisinage (F1 = {f1_lof*100:.1f}%)<br>
• <b>Random Forest</b> — apprend la frontière de la zone normale via un noyau RBF (F1 = {f1_rf*100:.1f}%)<br><br>
Le meilleur modèle individuel est <b>{best[0]}</b> (F1 = {best[1]*100:.1f}%), mais aucun modèle seul
n'est suffisant. Le vote majoritaire (≥ 2/3) réduit les faux positifs de
<b>{red['reduction_vs_union']:.0f}%</b> par rapport à l'union brute ({red['nb_union']} → {red['nb_ensemble']} alertes).<br>
{kappa_txt}<br><br>
Cette approche est particulièrement adaptée au contexte OCP El Jadida où la fiabilité
des alertes est critique : mieux vaut quelques faux négatifs qu'une avalanche de faux positifs.
</div>
""", unsafe_allow_html=True)

                        st.markdown("---")
                        # ── Distribution des votes ─────────────────────────
                        st.markdown("#### 🗳️ Distribution des votes par anomalie confirmée")
                        votes_dist = df_filtre[df_filtre["anomalie"] == -1]["nb_votes"].value_counts().sort_index()
                        if not votes_dist.empty:
                            fig_v, ax_v = plt.subplots(figsize=(6, 3))
                            fig_v.patch.set_alpha(0.0)
                            ax_v.bar([f"{v}/3 votes" for v in votes_dist.index],
                                     votes_dist.values,
                                     color=["#c8983a", "#d06050"], width=0.5, zorder=3)
                            ax_v.set_ylabel("Nombre d'anomalies")
                            ax_v.spines["top"].set_visible(False)
                            ax_v.spines["right"].set_visible(False)
                            for bar, val in zip(ax_v.patches, votes_dist.values):
                                ax_v.text(bar.get_x() + bar.get_width()/2,
                                          bar.get_height() + votes_dist.values.max()*0.02,
                                          str(val), ha="center", color="white", fontsize=10)
                            plt.tight_layout()
                            st.pyplot(fig_v, use_container_width=True)
                            plt.close(fig_v)

                        accord = int(((df_filtre["anomalie_if"] == df_filtre["anomalie_lof"]) &
                                      (df_filtre["anomalie_lof"] == df_filtre["anomalie_rf"])).sum())
                        pct_accord = accord / total_f * 100 if total_f > 0 else 0
                        st.metric("✅ Lignes où les 3 modèles s'accordent",
                                  f"{accord}/{total_f}", f"{pct_accord:.1f}%")
                        if pct_accord >= 80:
                            st.success("Bonne cohérence entre les modèles.")
                        elif pct_accord >= 60:
                            st.warning("Cohérence modérée — considérez ajuster la contamination.")
                        else:
                            st.error("Faible cohérence — les modèles divergent beaucoup.")

                        # Kappa inter-modèles si disponible
                        if eval_res and eval_res.get("agreement", {}).get("kappa_mean") is not None:
                            kappa_m = eval_res["agreement"]["kappa_mean"]
                            if kappa_m >= 0.6:
                                kappa_label = "accord fort"
                            elif kappa_m >= 0.4:
                                kappa_label = "accord modéré"
                            else:
                                kappa_label = "accord faible"
                            st.metric("🤝 Kappa moyen inter-modèles", f"{kappa_m:.3f}",
                                      kappa_label)

            # ── TAB 2 : XGBoost supervisé ─────────────────────────────────
            with tab_xgb:
                st.markdown("#### Évaluation du modèle XGBoost supervisé")

                df_feedback = collecter_feedbacks_pour_xgboost()
                nb_fb = len(df_feedback)

                st.progress(min(nb_fb / 5, 1.0))
                st.caption(f"{nb_fb}/5 validations minimum requises")

                if st.session_state.get("modele_xgb") is not None:
                    st.success("✅ Modèle XGBoost entraîné et prêt !")
                    stats = get_stats_xgboost(st.session_state.modele_xgb)
                    if stats:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🤖 Type", stats['type'])
                        with col2:
                            st.metric("🌳 Arbres", stats['n_estimators'])
                        with col3:
                            st.metric("📏 Profondeur", stats['max_depth'])
                else:
                    st.info("🤖 Aucun modèle XGBoost entraîné. Validez au moins 5 anomalies dans l'onglet **Validation**.")

                if nb_fb >= 5:
                    st.markdown("---")
                    with st.spinner("Calcul des performances en cours..."):
                        results, _ = evaluer_performance_modele(df_feedback)

                    if results:
                        st.markdown("#### 📈 Métriques de performance")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🎯 Accuracy",  f"{results['accuracy']*100:.1f}%")
                        with col2:
                            st.metric("⚡ Précision", f"{results['precision']*100:.1f}%")
                        with col3:
                            st.metric("🔍 Rappel",    f"{results['recall']*100:.1f}%")
                        with col4:
                            st.metric("📈 F1-Score",  f"{results['f1']*100:.1f}%")

                        # ROC-AUC + mode d'évaluation
                        col_auc, col_mode = st.columns(2)
                        with col_auc:
                            if results.get('roc_auc') is not None:
                                st.metric("📐 ROC-AUC", f"{results['roc_auc']:.3f}")
                            else:
                                st.info("ROC-AUC non disponible (une seule classe présente).")
                        with col_mode:
                            st.info(f"📊 Mode d'évaluation : **{results.get('split_mode','—')}** "
                                    f"(test : {results.get('test_size',0)} échantillons)")

                        st.markdown("---")
                        st.markdown("#### 🔬 Validation croisée")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Score moyen", f"{results['cv_mean']*100:.1f}%")
                        with col2:
                            st.metric("Écart-type",  f"± {results['cv_std']*100:.1f}%")

                        st.markdown("---")
                        accuracy = results['accuracy']
                        if accuracy >= 0.9:
                            st.success("✅ **Excellent !** Modèle très fiable (≥90%)")
                        elif accuracy >= 0.75:
                            st.info("👍 **Bon modèle** (75–90%)")
                        elif accuracy >= 0.6:
                            st.warning("⚠️ **Modèle moyen** (60–75%)")
                        else:
                            st.error("❌ **Modèle faible** (<60%)")

                        st.markdown("---")
                        st.markdown("#### 📈 Courbe d'apprentissage")
                        afficher_graphique_apprentissage(df_feedback)

                        st.markdown("#### 📊 Progression données")
                        st.progress(min(results['n_samples'] / 50, 1.0))
                        if results['n_samples'] < 30:
                            st.info(f"💡 Encore {30 - results['n_samples']} validations recommandées pour un modèle robuste.")
                        else:
                            st.success("🎉 Données suffisantes pour un modèle fiable.")
                    else:
                        st.error("Erreur lors du calcul des métriques.")
                else:
                    st.warning(f"⚠️ Il manque {5 - nb_fb} validation(s) pour évaluer le modèle.")

        elif menu == "🗄️ DB":
            st.markdown("### 🗄️ Base de données")
            stats = stats_db()
            if stats:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("📊 Total logs",   stats.get("total",    0))
                with c2:
                    st.metric("🚨 Anomalies",    stats.get("anomalies", 0))
                with c3:
                    st.metric("👥 Sessions",     stats.get("sessions",  0))
                with c4:
                    st.metric("🕐 Dernier log",  stats.get("last",     "—"))
            st.markdown("---")
            db_limit = st.slider("Limite d'affichage", 50, 2000, 500, 50)
            df_hist  = charger_historique(limit=db_limit)
            if not df_hist.empty:
                st.dataframe(df_hist, use_container_width=True, height=400)
                csv = df_hist.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Télécharger CSV", csv,
                    file_name=f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("Aucune donnée.")
            st.markdown("---")
            if st.button("🗑️ Vider la base", type="secondary"):
                vider_db()
                st.success("Base vidée !")
                st.rerun()

    else:
        if menu not in ("📁 Upload", "🏠 Accueil"):
            st.warning("⚠️ Aucune donnée. Vérifiez le format de vos fichiers.")
else:
    if menu not in ("📁 Upload", "🏠 Accueil"):
        st.info("📁 Veuillez uploader des fichiers logs dans l'onglet 'Upload'.")