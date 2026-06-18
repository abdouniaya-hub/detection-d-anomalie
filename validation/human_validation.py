import hashlib
import streamlit as st
import pandas as pd

ANOMALIES_PAR_PAGE = 10


def _key(*parts):
    raw = "_".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _safe(row, col, default="N/A"):
    return row[col] if col in row.index else default


def _badge(val):
    return "🔴 ANOMALIE" if val == -1 else "🟢 NORMAL"


# ============================================================
# RÉSUMÉ KPI D'UN MODÈLE (lecture seule)
# ============================================================
def _afficher_kpi_modele(df_all, col_anomalie, col_score, label_score):
    if col_anomalie not in df_all.columns:
        st.warning("Colonnes du modèle non disponibles.")
        return
    nb_anom = int((df_all[col_anomalie] == -1).sum())
    nb_norm = int((df_all[col_anomalie] == 1).sum())
    total   = len(df_all)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🔴 Anomalies détectées", nb_anom, f"{nb_anom/total*100:.1f}%")
    with c2:
        st.metric("🟢 Normaux", nb_norm, f"{nb_norm/total*100:.1f}%")
    with c3:
        if col_score in df_all.columns:
            st.metric(label_score, f"{df_all[col_score].mean():.4f}")
        else:
            st.metric(label_score, "N/A")


# ============================================================
# LISTE PAGINÉE — LECTURE SEULE (onglets IF / LOF / SVM)
# ============================================================
def _afficher_liste_readonly(df_modele, col_score, prefixe_page):
    """Affiche les anomalies d'un modèle en lecture seule, sans boutons de validation."""
    if df_modele.empty:
        st.success("✅ Ce modèle n'a détecté aucune anomalie.")
        return

    total    = len(df_modele)
    df_reset = df_modele.reset_index(drop=True)
    nb_pages = max(1, (total + ANOMALIES_PAR_PAGE - 1) // ANOMALIES_PAR_PAGE)

    page_key = f"page_ro_{prefixe_page}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    page  = min(st.session_state[page_key], nb_pages - 1)
    debut = page * ANOMALIES_PAR_PAGE
    fin   = min(debut + ANOMALIES_PAR_PAGE, total)

    # Navigation
    col_prev, col_mid, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("←", key=_key(prefixe_page, "ro_prev", page), disabled=(page == 0)):
            st.session_state[page_key] = max(0, page - 1)
            st.rerun()
    with col_mid:
        st.markdown(f"**Page {page + 1} / {nb_pages}** — {total} anomalie(s)")
    with col_next:
        if st.button("→", key=_key(prefixe_page, "ro_next", page), disabled=(page >= nb_pages - 1)):
            st.session_state[page_key] = min(nb_pages - 1, page + 1)
            st.rerun()

    st.markdown("---")

    page_df = df_reset.iloc[debut:fin]
    for i, (_, row) in enumerate(page_df.iterrows()):
        num = debut + i + 1
        with st.expander(f"🔍 Anomalie #{num}", expanded=False):
            # Score du modèle
            if col_score in row.index:
                st.caption(f"Score : {row[col_score]:.4f}")

            # Votes des 3 modèles si disponibles
            if all(c in row.index for c in ["anomalie_if", "anomalie_lof", "anomalie_rf"]):
                cv1, cv2, cv3 = st.columns(3)
                with cv1:
                    st.caption(f"🌲 IF : {_badge(row['anomalie_if'])}")
                with cv2:
                    st.caption(f"📍 LOF : {_badge(row['anomalie_lof'])}")
                with cv3:
                    st.caption(f"🌳 RF : {_badge(row['anomalie_rf'])}")
                if "nb_votes" in row.index:
                    st.info(f"🗳️ Votes : **{int(row['nb_votes'])}/3**")

            st.markdown("---")
            st.write("**Message :**",   _safe(row, "message"))
            st.write("**Composant :**", _safe(row, "composant"))
            st.write("**Niveau :**",    _safe(row, "niveau"))
            st.write("**Source :**",    _safe(row, "source"))


# ============================================================
# LISTE PAGINÉE — VALIDATION (onglet Vote Ensemble uniquement)
# ============================================================
def _afficher_liste_validation(df_ens):
    """Affiche les anomalies du vote ensemble avec boutons Confirmer / Rejeter."""
    if df_ens.empty:
        st.success("✅ Aucune anomalie à valider.")
        return

    if "validations" not in st.session_state:
        st.session_state.validations = {}

    total    = len(df_ens)
    df_reset = df_ens.reset_index(drop=True)
    nb_pages = max(1, (total + ANOMALIES_PAR_PAGE - 1) // ANOMALIES_PAR_PAGE)

    page_key = "page_val_ens"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    page  = min(st.session_state[page_key], nb_pages - 1)
    debut = page * ANOMALIES_PAR_PAGE
    fin   = min(debut + ANOMALIES_PAR_PAGE, total)

    # Navigation
    col_prev, col_mid, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("←", key=_key("ens_val", "prev", page), disabled=(page == 0)):
            st.session_state[page_key] = max(0, page - 1)
            st.rerun()
    with col_mid:
        st.markdown(f"**Page {page + 1} / {nb_pages}** — {total} anomalie(s)")
    with col_next:
        if st.button("→", key=_key("ens_val", "next", page), disabled=(page >= nb_pages - 1)):
            st.session_state[page_key] = min(nb_pages - 1, page + 1)
            st.rerun()

    # Boutons de page (Confirmer tout / Rejeter tout)
    cp1, cp2, cp3 = st.columns([2, 2, 2])
    with cp1:
        if st.button("✔✔ Confirmer cette page",
                     key=_key("ens_val", "ok_page", page),
                     use_container_width=True):
            for i in range(debut, fin):
                st.session_state.validations[f"val_ens_{i}"] = "confirmée"
            st.rerun()
    with cp2:
        if st.button("✖✖ Rejeter cette page",
                     key=_key("ens_val", "no_page", page),
                     use_container_width=True):
            for i in range(debut, fin):
                st.session_state.validations[f"val_ens_{i}"] = "rejetée"
            st.rerun()
    with cp3:
        if st.button("🔄 Réinitialiser tout",
                     key=_key("ens_val", "reset"),
                     use_container_width=True):
            keys_del = [k for k in list(st.session_state.validations.keys())
                        if k.startswith("val_ens_")]
            for k in keys_del:
                del st.session_state.validations[k]
            st.rerun()

    st.markdown("---")

    page_df = df_reset.iloc[debut:fin]
    for local_i, (_, row) in enumerate(page_df.iterrows()):
        global_i  = debut + local_i
        cle       = f"val_ens_{global_i}"
        statut    = st.session_state.validations.get(cle)
        action_key = _key("ens_val", "action", global_i)

        icone = "✅" if statut == "confirmée" else ("❌" if statut == "rejetée" else "⏳")

        with st.expander(f"{icone} Anomalie #{global_i + 1}", expanded=(statut is None)):
            # Votes des 3 modèles
            if all(c in row.index for c in ["anomalie_if", "anomalie_lof", "anomalie_rf"]):
                cv1, cv2, cv3 = st.columns(3)
                with cv1:
                    st.caption(f"🌲 IF : {_badge(row['anomalie_if'])}")
                with cv2:
                    st.caption(f"📍 LOF : {_badge(row['anomalie_lof'])}")
                with cv3:
                    st.caption(f"🌳 RF : {_badge(row['anomalie_rf'])}")
            if "nb_votes" in row.index:
                st.info(f"🗳️ Vote final : **{int(row['nb_votes'])}/3**")
            if "score_if" in row.index:
                st.caption(f"Score IF : {row['score_if']:.4f}")

            st.markdown("---")
            st.write("**Message :**",   _safe(row, "message"))
            st.write("**Composant :**", _safe(row, "composant"))
            st.write("**Niveau :**",    _safe(row, "niveau"))
            st.write("**Source :**",    _safe(row, "source"))

            # Boutons de validation
            if statut is None:
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("✔ Confirmer",
                                 key=_key("ens_val", "ok", global_i),
                                 use_container_width=True):
                        st.session_state[action_key] = ("confirmer", cle)
                with b2:
                    if st.button("✖ Rejeter",
                                 key=_key("ens_val", "no", global_i),
                                 use_container_width=True):
                        st.session_state[action_key] = ("rejeter", cle)
            else:
                b1, b2 = st.columns([3, 1])
                with b1:
                    st.info(f"Statut : {statut}")
                with b2:
                    if st.button("↩ Modifier",
                                 key=_key("ens_val", "edit", global_i),
                                 use_container_width=True):
                        st.session_state[action_key] = ("modifier", cle)

    # Appliquer les actions après la boucle
    need_rerun = False
    for local_i in range(debut, fin):
        ak = _key("ens_val", "action", local_i)
        if ak in st.session_state:
            action, cle = st.session_state.pop(ak)
            if action == "confirmer":
                st.session_state.validations[cle] = "confirmée"
            elif action == "rejeter":
                st.session_state.validations[cle] = "rejetée"
            elif action == "modifier":
                st.session_state.validations.pop(cle, None)
            need_rerun = True
    if need_rerun:
        st.rerun()


# ============================================================
# INTERFACE PRINCIPALE DE VALIDATION
# ============================================================
def afficher_interface_validation(anomalies: pd.DataFrame):

    st.markdown(
        '<div class="section-titre">✅ Validation Humaine des Anomalies</div>',
        unsafe_allow_html=True,
    )

    if anomalies.empty:
        st.success("Aucune anomalie à valider")
        return

    if "validations" not in st.session_state:
        st.session_state.validations = {}

    has_models = all(c in anomalies.columns for c in
                     ["anomalie_if", "anomalie_lof", "anomalie_rf"])

    if not has_models:
        st.warning("Les colonnes des 3 modèles ne sont pas disponibles.")
        _afficher_liste_validation(anomalies)
        return

    df_if  = anomalies[anomalies["anomalie_if"]  == -1].copy()
    df_lof = anomalies[anomalies["anomalie_lof"] == -1].copy()
    df_rf  = anomalies[anomalies["anomalie_rf"]  == -1].copy()
    df_ens = anomalies.copy()   # déjà filtrées anomalie == -1 par main.py

    # KPIs comparatifs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🌲 Isolation Forest", len(df_if),  "anomalies")
    with c2:
        st.metric("📍 LOF",              len(df_lof), "anomalies")
    with c3:
        st.metric("🌳 Random Forest",    len(df_rf),  "anomalies")
    with c4:
        st.metric("🗳️ Vote Ensemble",    len(df_ens), "anomalies (≥2/3)")

    st.markdown("---")

    tab_if, tab_lof, tab_rf, tab_ens_tab = st.tabs([
        f"🌲 Isolation Forest ({len(df_if)})",
        f"📍 LOF ({len(df_lof)})",
        f"🌳 Random Forest ({len(df_rf)})",
        f"🗳️ ✅ Vote Ensemble — Validation ({len(df_ens)})",
    ])

    # ── Onglets lecture seule ─────────────────────────────────────────
    with tab_if:
        st.markdown("### 🌲 Isolation Forest")
        st.caption("Isole les anomalies par arbres aléatoires. Score bas = forte anomalie.")
        _afficher_kpi_modele(anomalies, "anomalie_if", "score_if", "Score moyen IF")
        st.markdown("---")
        _afficher_liste_readonly(df_if, "score_if", "if")

    with tab_lof:
        st.markdown("### 📍 LOF — Local Outlier Factor")
        st.caption("Compare la densité locale d'un point à ses voisins. Score élevé = anomalie.")
        _afficher_kpi_modele(anomalies, "anomalie_lof", "score_lof", "Score moyen LOF")
        st.markdown("---")
        _afficher_liste_readonly(df_lof, "score_lof", "lof")

    with tab_rf:
        st.markdown("### 🌳 Random Forest")
        st.caption("Apprend la frontière de la zone normale (pseudo-labels IF∩LOF). Score élevé = anomalie probable.")
        _afficher_kpi_modele(anomalies, "anomalie_rf", "score_rf", "Score moyen OCSVM")
        st.markdown("---")
        _afficher_liste_readonly(df_rf, "score_rf", "svm")

    # ── Onglet validation (Vote Ensemble uniquement) ──────────────────
    with tab_ens_tab:
        st.markdown("### 🗳️ Vote Ensemble — Validation humaine")
        st.caption("Seules les anomalies retenues par ≥ 2 modèles sur 3 sont présentées ici.")

        if "nb_votes" in df_ens.columns:
            votes_2 = int((df_ens["nb_votes"] == 2).sum())
            votes_3 = int((df_ens["nb_votes"] == 3).sum())
            nb_val  = sum(1 for k in st.session_state.validations if k.startswith("val_ens_"))
            cv1, cv2, cv3 = st.columns(3)
            with cv1:
                st.metric("🟡 2/3 votes", votes_2, "accord partiel")
            with cv2:
                st.metric("🔴 3/3 votes", votes_3, "unanimité")
            with cv3:
                st.metric("✅ Validées", nb_val, f"/{len(df_ens)}")

        st.progress(
            min(sum(1 for k in st.session_state.validations if k.startswith("val_ens_"))
                / max(len(df_ens), 1), 1.0)
        )
        st.markdown("---")
        _afficher_liste_validation(df_ens)


# ============================================================
# FEEDBACKS XGBOOST — uniquement depuis le Vote Ensemble
# ============================================================
def collecter_feedbacks_pour_xgboost():
    if not st.session_state.get("validations"):
        return pd.DataFrame()

    df = st.session_state.get("anomalies_a_valider", pd.DataFrame()).copy()
    if df.empty:
        return pd.DataFrame()

    df_reset = df.reset_index(drop=True)
    labels   = []

    for i in range(len(df_reset)):
        val = st.session_state.validations.get(f"val_ens_{i}")
        if val == "confirmée":
            labels.append(1)
        elif val == "rejetée":
            labels.append(0)
        else:
            labels.append(None)

    df_reset["label"] = labels
    df_reset = df_reset.dropna(subset=["label"])
    if not df_reset.empty:
        df_reset["label"] = df_reset["label"].astype(int)
    return df_reset


# ============================================================
# STATUS XGBOOST
# ============================================================
def afficher_statut_xgboost():
    if st.session_state.get("modele_xgb") is not None:
        st.success("✅ XGBoost entraîné")
    else:
        st.info("ℹ️ XGBoost en attente")
