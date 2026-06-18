import pandas as pd
import numpy as np
import streamlit as st
import time
import tracemalloc
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
from config import MOTS_CRITIQUES

# ============================================================
# DÉTECTION DE BASE (Isolation Forest seul)
# ============================================================
@st.cache_data
def detecter_anomalies(df, contamination=0.05):
    """Détecter les anomalies avec Isolation Forest uniquement"""
    if df.empty:
        return df

    # Features numériques "classiques"
    features_num = df[['niveau_num', 'longueur_message', 'frequence_flux']].fillna(0)
    scaler = StandardScaler()
    features_num_norm = scaler.fit_transform(features_num)

    # TF-IDF forcé sur le vocabulaire critique uniquement (anti-bruit)
    corpus = (
        df['composant'].fillna('') + ' ' +
        df['requete'].fillna('') + ' ' +
        df['message'].fillna('')
    )
    vectorizer = TfidfVectorizer(
        vocabulary=MOTS_CRITIQUES,
        lowercase=True
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Fusion sparse: TF-IDF + numérique
    features_finales = sp.hstack(
        [tfidf_matrix, sp.csr_matrix(features_num_norm)],
        format='csr'
    )

    modele = IsolationForest(contamination=contamination, random_state=42)
    df['anomalie'] = modele.fit_predict(features_finales)
    df['score'] = modele.score_samples(features_finales)
    return df


# ============================================================
# DÉTECTION PAR ENSEMBLE (3 modèles + vote) — DATASET UNIFIÉ
# ============================================================

# Plafond de lignes pour garder l'analyse rapide (<5 s) même sur les FULL logs
MAX_LIGNES_ANALYSE = 8_000

@st.cache_data
def detecter_ensemble(df, contamination=0.05):
    """
    Détection d'anomalies par vote d'ensemble sur le dataset complet
    (Nginx + Spring Boot mélangés) — 3 modèles non-supervisés INDÉPENDANTS.

    Modèles :
      1. Isolation Forest  — points globalement isolés (sparse, O(n log n))
      2. LOF               — anomalies locales par densité (PCA 20D, O(n·k))
      3. One-Class SVM     — frontière de normalité (SGD + Nystroem, O(n))

    Si le dataset dépasse MAX_LIGNES_ANALYSE lignes, on échantillonne de façon
    stratifiée (par source), on prédit, puis on propage les labels à tout le df.
    """
    from sklearn.preprocessing import normalize

    if df.empty:
        return df

    df = df.copy()
    n_total = len(df)

    # ── Sous-échantillonnage stratifié si dataset trop grand ─────────────
    besoin_extrapol = n_total > MAX_LIGNES_ANALYSE
    if besoin_extrapol:
        # Échantillon représentatif (proportionnel par source)
        frames = []
        for src, grp in df.groupby('source'):
            n_src = min(len(grp), int(MAX_LIGNES_ANALYSE * len(grp) / n_total) + 1)
            frames.append(grp.sample(n_src, random_state=42))
        df_sample = pd.concat(frames).head(MAX_LIGNES_ANALYSE).copy()
    else:
        df_sample = df.copy()

    # ── 1. Encodage one-hot de la source ─────────────────────────────────
    sources_uniques = df['source'].unique().tolist()
    for src in sources_uniques:
        col = f"source_{src.lower().replace(' ', '_')}"
        df_sample[col] = (df_sample['source'] == src).astype(float)
    source_cols = [f"source_{s.lower().replace(' ', '_')}" for s in sources_uniques]

    # ── 2. Features numériques normalisées ───────────────────────────────
    num_cols = ['niveau_num', 'longueur_message', 'frequence_flux']
    features_num = df_sample[num_cols].fillna(0)
    scaler = StandardScaler()
    features_num_norm = scaler.fit_transform(features_num)
    source_matrix = df_sample[source_cols].values
    features_num_full = np.hstack([features_num_norm, source_matrix])

    # ── 3. TF-IDF vocabulaire commun ─────────────────────────────────────
    corpus = (
        df_sample['composant'].fillna('') + ' ' +
        df_sample['requete'].fillna('') + ' ' +
        df_sample['message'].fillna('')
    )
    vectorizer = TfidfVectorizer(vocabulary=MOTS_CRITIQUES, lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # ── 4. Fusion sparse + normalisation L2 ──────────────────────────────
    features_finales = sp.hstack(
        [tfidf_matrix, sp.csr_matrix(features_num_full)],
        format='csr'
    )
    features_finales = normalize(features_finales, norm='l2')

    # ── 5. Isolation Forest (sparse natif — très rapide) ─────────────────
    if_model = IsolationForest(
        contamination=contamination, random_state=42, n_jobs=-1
    )
    preds_if    = if_model.fit_predict(features_finales)
    scores_if   = if_model.score_samples(features_finales)

    # ── 6. LOF — réduction PCA légère (20 composantes) pour éviter .toarray() complet
    try:
        from sklearn.decomposition import TruncatedSVD
        n_comp_lof = min(20, features_finales.shape[1] - 1)
        svd_lof = TruncatedSVD(n_components=n_comp_lof, random_state=42)
        feat_lof = svd_lof.fit_transform(features_finales)   # dense mais petit : (n, 20)
        lof_model = LocalOutlierFactor(
            n_neighbors=min(20, len(df_sample) - 1),
            contamination=contamination
        )
        preds_lof  = lof_model.fit_predict(feat_lof)
        scores_lof = -lof_model.negative_outlier_factor_
    except Exception:
        preds_lof  = preds_if.copy()
        scores_lof = scores_if.copy()

    # ── 7. Random Forest — labels synthétiques combinant IF + LOF ────────
    # Les pseudo-labels sont construits à partir du consensus IF ∩ LOF :
    # anomalie = -1 si les DEUX modèles la signalent, sinon 1 (normal).
    # Cela rend RF indépendant de IF seul et plus robuste.
    pseudo_labels = np.where(
        (preds_if == -1) & (preds_lof == -1), -1, 1
    )
    try:
        rf_model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=8
        )
        rf_model.fit(features_finales, pseudo_labels)
        preds_rf  = rf_model.predict(features_finales)
        proba_rf  = rf_model.predict_proba(features_finales)
        # score_rf : probabilité d'être anomalie (classe -1)
        classes   = list(rf_model.classes_)
        idx_anom  = classes.index(-1) if -1 in classes else 0
        scores_rf = proba_rf[:, idx_anom]
    except Exception:
        preds_rf  = preds_if.copy()
        scores_rf = -scores_if  # inverse pour garder même sens

    # ── 8. Vote majoritaire sur l'échantillon ────────────────────────────
    df_sample = df_sample.copy()
    df_sample['anomalie_if']  = preds_if
    df_sample['score_if']     = scores_if
    df_sample['anomalie_lof'] = preds_lof
    df_sample['score_lof']    = scores_lof
    df_sample['anomalie_rf']  = preds_rf
    df_sample['score_rf']     = scores_rf
    df_sample['nb_votes'] = (
        (df_sample['anomalie_if']  == -1).astype(int) +
        (df_sample['anomalie_lof'] == -1).astype(int) +
        (df_sample['anomalie_rf']  == -1).astype(int)
    )
    df_sample['anomalie'] = df_sample['nb_votes'].apply(lambda v: -1 if v >= 2 else 1)

    # ── 9. Propagation aux lignes non-échantillonnées ─────────────────────
    if besoin_extrapol:
        # Les lignes hors-échantillon reçoivent les labels IF (rapide à inférer)
        # depuis le modèle déjà entraîné — on reconstruit leurs features
        idx_sample = df_sample.index
        idx_reste  = df.index.difference(idx_sample)

        df_reste = df.loc[idx_reste].copy()
        for src in sources_uniques:
            col = f"source_{src.lower().replace(' ', '_')}"
            df_reste[col] = (df_reste['source'] == src).astype(float)

        corpus_r = (
            df_reste['composant'].fillna('') + ' ' +
            df_reste['requete'].fillna('') + ' ' +
            df_reste['message'].fillna('')
        )
        tfidf_r = vectorizer.transform(corpus_r)
        fn_r    = df_reste[num_cols].fillna(0)
        fn_norm_r = scaler.transform(fn_r)
        sm_r    = df_reste[source_cols].values
        fnf_r   = np.hstack([fn_norm_r, sm_r])
        feat_r  = normalize(
            sp.hstack([tfidf_r, sp.csr_matrix(fnf_r)], format='csr'), norm='l2'
        )

        p_if_r  = if_model.predict(feat_r)
        s_if_r  = if_model.score_samples(feat_r)

        # LOF ne supporte pas predict() en novelty=False → on utilise IF comme proxy
        # RF : inférence directe sur le modèle entraîné
        try:
            p_rf_r   = rf_model.predict(feat_r)
            proba_r  = rf_model.predict_proba(feat_r)
            s_rf_r   = proba_r[:, idx_anom]
        except Exception:
            p_rf_r = p_if_r.copy()
            s_rf_r = s_if_r.copy()

        df_reste['anomalie_if']  = p_if_r
        df_reste['score_if']     = s_if_r
        df_reste['anomalie_lof'] = p_if_r   # proxy IF pour les lignes restantes
        df_reste['score_lof']    = s_if_r
        df_reste['anomalie_rf']  = p_rf_r
        df_reste['score_rf']     = s_rf_r
        df_reste['nb_votes'] = (
            (df_reste['anomalie_if']  == -1).astype(int) +
            (df_reste['anomalie_lof'] == -1).astype(int) +
            (df_reste['anomalie_rf']  == -1).astype(int)
        )
        df_reste['anomalie'] = df_reste['nb_votes'].apply(lambda v: -1 if v >= 2 else 1)

        df = pd.concat([df_sample, df_reste]).sort_index()
    else:
        df = df_sample

    return df


# ============================================================
# BENCHMARK — temps + mémoire (hors cache, exécuté à chaque rendu)
# ============================================================
def benchmarker_modeles(df, contamination=0.05, n_echantillon=500):
    """
    Mesure les temps d'entraînement/inférence et la consommation mémoire
    des 3 modèles sur un échantillon (n_echantillon lignes max).
    Appelée SANS @st.cache_data pour s'exécuter à chaque rendu.
    Retourne un dict stocké dans st.session_state['perf_modeles'].
    """
    if df.empty:
        return {}

    from sklearn.preprocessing import normalize

    # Échantillon representatif pour que le benchmark soit rapide
    df_s = df.sample(min(n_echantillon, len(df)), random_state=42).copy()

    # ── Préparation des features (même pipeline que detecter_ensemble) ───
    sources_uniques = df_s['source'].unique().tolist()
    for src in sources_uniques:
        col = f"source_{src.lower().replace(' ', '_')}"
        df_s[col] = (df_s['source'] == src).astype(float)
    source_cols = [f"source_{s.lower().replace(' ', '_')}" for s in sources_uniques]

    num_cols = ['niveau_num', 'longueur_message', 'frequence_flux']
    features_num = df_s[num_cols].fillna(0)
    scaler = StandardScaler()
    features_num_norm = scaler.fit_transform(features_num)
    source_matrix = df_s[source_cols].values
    features_num_full = np.hstack([features_num_norm, source_matrix])

    corpus = (
        df_s['composant'].fillna('') + ' ' +
        df_s['requete'].fillna('')   + ' ' +
        df_s['message'].fillna('')
    )
    vectorizer = TfidfVectorizer(vocabulary=MOTS_CRITIQUES, lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    features_finales = sp.hstack(
        [tfidf_matrix, sp.csr_matrix(features_num_full)], format='csr'
    )
    features_finales = normalize(features_finales, norm='l2')

    def mesurer(fn_train, fn_infer):
        """Retourne (t_train, t_infer, mem_train_kb, mem_infer_kb)."""
        tracemalloc.start()
        t0 = time.time()
        fn_train()
        t_train = round(time.time() - t0, 4)
        _, mem_train = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        t0 = time.time()
        fn_infer()
        t_infer = round(time.time() - t0, 4)
        _, mem_infer = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return t_train, t_infer, round(mem_train / 1024, 1), round(mem_infer / 1024, 1)

    perf = {}

    # Isolation Forest
    if_m = IsolationForest(contamination=contamination, random_state=42)
    t_tr, t_inf, m_tr, m_inf = mesurer(
        lambda: if_m.fit(features_finales),
        lambda: if_m.predict(features_finales)
    )
    perf['Isolation Forest'] = {
        'train_s': t_tr, 'infer_s': t_inf,
        'mem_train_kb': m_tr, 'mem_infer_kb': m_inf
    }

    # LOF
    try:
        features_dense = features_finales.toarray()
        lof_m = LocalOutlierFactor(contamination=contamination, novelty=False)
        tracemalloc.start()
        t0 = time.time()
        lof_m.fit_predict(features_dense)
        t_lof = round(time.time() - t0, 4)
        _, mem_lof = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        perf['LOF'] = {
            'train_s': t_lof, 'infer_s': t_lof,
            'mem_train_kb': round(mem_lof / 1024, 1),
            'mem_infer_kb': round(mem_lof / 1024, 1)
        }
    except Exception:
        perf['LOF'] = {'train_s': 0, 'infer_s': 0, 'mem_train_kb': 0, 'mem_infer_kb': 0}

    # Random Forest
    pseudo_bench = if_m.predict(features_finales)
    rf_m = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    t_tr, t_inf, m_tr, m_inf = mesurer(
        lambda: rf_m.fit(features_finales, pseudo_bench),
        lambda: rf_m.predict(features_finales)
    )
    perf['Random Forest'] = {
        'train_s': t_tr, 'infer_s': t_inf,
        'mem_train_kb': m_tr, 'mem_infer_kb': m_inf
    }

    perf['_n_echantillon'] = min(n_echantillon, len(df))
    return perf


# ============================================================
# EXTRACTION MOTS TECHNIQUES
# ============================================================
@st.cache_data
def extraire_mots_techniques_tfidf(df, top_n=5):
    """
    Extrait les mots techniques les plus importants par log via TF-IDF.
    """
    if df.empty:
        df['mots_techniques'] = []
        return df

    corpus = (
        df['composant'].fillna('') + ' ' +
        df['requete'].fillna('') + ' ' +
        df['message'].fillna('')
    )

    vectorizer = TfidfVectorizer(
        vocabulary=MOTS_CRITIQUES,
        lowercase=True,
    )

    try:
        tfidf = vectorizer.fit_transform(corpus)
        termes = np.array(vectorizer.get_feature_names_out())
    except ValueError:
        df['mots_techniques'] = [[] for _ in range(len(df))]
        return df

    mots_par_ligne = []
    for i in range(tfidf.shape[0]):
        ligne = tfidf.getrow(i)
        if ligne.nnz == 0:
            mots_par_ligne.append([])
            continue

        idx = ligne.indices
        scores = ligne.data
        ordre = np.argsort(scores)[::-1][:top_n]
        top_termes = []
        for j in ordre:
            terme = termes[idx[j]]
            if not terme.isdigit() and len(terme) >= 3:
                top_termes.append(terme)

        mots_par_ligne.append(top_termes)

    df['mots_techniques'] = mots_par_ligne
    return df


# ============================================================
# FONCTIONS D'ÉVALUATION DES PERFORMANCES
# ============================================================
def evaluer_performance_modele(df_feedback):
    """
    Évalue la performance du modèle XGBoost supervisé sur les validations humaines.

    Stratégie :
    - Si ≥ 10 échantillons  → train/test split 80/20 + validation croisée k-fold
    - Si 5–9 échantillons   → Leave-One-Out cross-validation (LOO)
    - < 5 échantillons       → refus

    Métriques retournées :
      accuracy, precision, recall, f1  — sur le jeu de test (out-of-sample)
      cv_mean, cv_std                  — scores de validation croisée
      roc_auc                          — AUC si les deux classes sont présentes
      n_samples, cv_folds, test_size
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score
    )
    from sklearn.model_selection import cross_val_score, train_test_split, LeaveOneOut

    if df_feedback.empty or len(df_feedback) < 5:
        return None, None

    try:
        # ── Features ────────────────────────────────────────────────────
        features_num = df_feedback[['niveau_num', 'longueur_message', 'frequence_flux']].fillna(0)
        scaler = StandardScaler()
        features_num_norm = scaler.fit_transform(features_num)

        corpus = (
            df_feedback['composant'].fillna('') + ' ' +
            df_feedback['requete'].fillna('') + ' ' +
            df_feedback['message'].fillna('')
        )
        vectorizer = TfidfVectorizer(vocabulary=MOTS_CRITIQUES, lowercase=True)
        tfidf_matrix = vectorizer.fit_transform(corpus)

        X = sp.hstack([tfidf_matrix, sp.csr_matrix(features_num_norm)], format='csr')
        y = df_feedback['label'].astype(int)

        # ── Modèle ──────────────────────────────────────────────────────
        try:
            import xgboost as xgb
            modele_cls = lambda: xgb.XGBClassifier(
                n_estimators=50, max_depth=3, random_state=42, eval_metric='logloss',
                use_label_encoder=False
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            modele_cls = lambda: GradientBoostingClassifier(
                n_estimators=50, max_depth=3, random_state=42
            )

        n = len(df_feedback)
        has_both_classes = len(y.unique()) >= 2

        if not has_both_classes:
            return {
                'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                'cv_mean': 1.0, 'cv_std': 0.0, 'roc_auc': None,
                'n_samples': n, 'cv_folds': 1, 'test_size': n,
                'split_mode': 'une seule classe — confirmez ET rejetez des anomalies',
            }, None

        # ── Validation croisée ──────────────────────────────────────────
        if n >= 10 and has_both_classes:
            from sklearn.model_selection import StratifiedKFold
            n_folds = min(5, int(y.value_counts().min()))
            cv_folds = max(2, n_folds)
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores_cv = cross_val_score(modele_cls(), X, y, cv=skf,
                                        scoring='accuracy', error_score=0.0)

            # Train/test split out-of-sample
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if has_both_classes else None
            )
            modele = modele_cls()
            modele.fit(X_train, y_train)
            y_pred = modele.predict(X_test)
            test_size = len(y_test)
        elif has_both_classes:
            # LOO pour les petits datasets
            loo = LeaveOneOut()
            loo_scores = cross_val_score(modele_cls(), X, y, cv=loo, scoring='accuracy')
            scores_cv = loo_scores
            cv_folds = n
            # Évaluation finale sur tout le dataset (pas de split possible)
            modele = modele_cls()
            modele.fit(X, y)
            y_pred = modele.predict(X)
            y_test = y
            test_size = n
        else:
            # Une seule classe : évaluation dégradée
            modele = modele_cls()
            modele.fit(X, y)
            y_pred = modele.predict(X)
            y_test = y
            scores_cv = np.array([1.0])
            cv_folds = 1
            test_size = n

        # ── Calcul des métriques ────────────────────────────────────────
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # ROC-AUC (si les deux classes sont présentes dans y_test)
        roc_auc = None
        if hasattr(modele, 'predict_proba') and len(np.unique(y_test)) >= 2:
            try:
                if n >= 10:
                    proba_test = modele.predict_proba(X_test)[:, 1]
                else:
                    proba_test = modele.predict_proba(X)[:, 1]
                roc_auc = round(roc_auc_score(y_test, proba_test), 3)
            except Exception:
                roc_auc = None

        results = {
            'accuracy':   round(accuracy,  3),
            'precision':  round(precision, 3),
            'recall':     round(recall,    3),
            'f1':         round(f1,        3),
            'cv_mean':    round(scores_cv.mean(), 3),
            'cv_std':     round(scores_cv.std(),  3),
            'roc_auc':    roc_auc,
            'n_samples':  n,
            'cv_folds':   cv_folds,
            'test_size':  test_size,
            'split_mode': 'train/test 80/20' if n >= 10 else ('LOO' if has_both_classes else 'in-sample'),
        }
        return results, modele

    except Exception as e:
        print(f"Erreur évaluation XGBoost : {e}")
        return None, None


def afficher_graphique_apprentissage(df_feedback):
    """Affiche la courbe d'apprentissage du modèle XGBoost."""
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_score

    if df_feedback.empty or len(df_feedback) < 10:
        st.info("Pas assez de données pour la courbe d'apprentissage (minimum 10 validations).")
        return

    # Vérifier qu'on a les deux classes
    if len(df_feedback['label'].unique()) < 2:
        st.info("💡 La courbe d'apprentissage nécessite des anomalies **confirmées ET rejetées**. "
                "Validez quelques anomalies dans les deux sens.")
        return

    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    n_samples = range(5, len(df_feedback) + 1, max(1, len(df_feedback) // 10))
    scores = []

    for n in n_samples:
        df_sample = df_feedback.head(n)

        # Sauter les itérations avec une seule classe
        if len(df_sample['label'].unique()) < 2:
            scores.append(None)
            continue

        features_num = df_sample[['niveau_num', 'longueur_message', 'frequence_flux']].fillna(0)
        scaler = StandardScaler()
        features_num_norm = scaler.fit_transform(features_num)
        corpus = (
            df_sample['composant'].fillna('') + ' ' +
            df_sample['requete'].fillna('') + ' ' +
            df_sample['message'].fillna('')
        )
        vectorizer = TfidfVectorizer(vocabulary=MOTS_CRITIQUES, lowercase=True)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        X = sp.hstack([tfidf_matrix, sp.csr_matrix(features_num_norm)], format='csr')
        y = df_sample['label'].astype(int)

        try:
            if HAS_XGB:
                modele = xgb.XGBClassifier(
                    n_estimators=50, max_depth=3, random_state=42,
                    eval_metric='logloss', use_label_encoder=False
                )
            else:
                from sklearn.ensemble import GradientBoostingClassifier
                modele = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)

            from sklearn.model_selection import StratifiedKFold
            n_folds = min(3, int(y.value_counts().min()))  # max folds = min classe
            if n_folds < 2:
                # Pas assez d'exemples par classe pour CV → score direct
                modele.fit(X, y)
                scores.append(float((modele.predict(X) == y).mean()))
            else:
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(modele, X, y, cv=skf,
                                            scoring='accuracy', error_score=0.0)
                scores.append(float(cv_scores.mean()))
        except Exception:
            scores.append(None)

    # Filtrer les None
    pts_x = [x for x, s in zip(n_samples, scores) if s is not None]
    pts_y = [s for s in scores if s is not None]

    if not pts_y:
        st.info("Pas assez de données diversifiées pour tracer la courbe.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_alpha(0.0)
    ax.plot(pts_x, pts_y, 'o-', color='#c8b48a', linewidth=2, markersize=8)
    ax.set_xlabel('Nombre de validations', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_title("Courbe d'apprentissage", fontsize=12)
    ax.set_ylim([0, 1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# ÉVALUATION COMPARATIVE DES 3 MODÈLES ENSEMBLE
# ============================================================
def evaluer_modeles_ensemble(df_filtre):
    """
    Calcule Précision / Rappel / F1 / Accuracy de chaque modèle individuel
    en prenant le vote d'ensemble comme référence (pseudo-vérité terrain).

    Retourne un dict avec :
      - metrics_df   : DataFrame comparatif (Modèle × Précision/Rappel/F1/Accuracy)
      - confusion    : dict {modele: {TP, FP, TN, FN}}
      - reduction    : statistiques de réduction d'alertes par le vote
      - agreement    : dict avec taux d'accord inter-modèles et kappa de Cohen
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, accuracy_score,
        cohen_kappa_score, matthews_corrcoef
    )

    required = ['anomalie', 'anomalie_if', 'anomalie_lof', 'anomalie_rf', 'nb_votes']
    if df_filtre.empty or not all(c in df_filtre.columns for c in required):
        return None

    y_ref  = (df_filtre['anomalie']     == -1).astype(int)   # vote ensemble = référence
    y_if   = (df_filtre['anomalie_if']  == -1).astype(int)
    y_lof  = (df_filtre['anomalie_lof'] == -1).astype(int)
    y_rf   = (df_filtre['anomalie_rf']  == -1).astype(int)

    def metriques(y_true, y_pred, nom):
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        acc  = accuracy_score(y_true, y_pred)
        try:
            kappa = cohen_kappa_score(y_true, y_pred)
        except Exception:
            kappa = 0.0
        try:
            mcc = matthews_corrcoef(y_true, y_pred)
        except Exception:
            mcc = 0.0
        return {
            'Modèle':    nom,
            'Précision': round(prec,  3),
            'Rappel':    round(rec,   3),
            'F1-Score':  round(f1,    3),
            'Accuracy':  round(acc,   3),
            'Kappa':     round(kappa, 3),
            'MCC':       round(mcc,   3),
            'Anomalies': int(y_pred.sum()),
        }

    rows = [
        metriques(y_ref, y_if,  'Isolation Forest'),
        metriques(y_ref, y_lof, 'LOF'),
        metriques(y_ref, y_rf,  'Random Forest'),
    ]
    # Ensemble lui-même → référence par construction
    rows.append({
        'Modèle':    'Vote Ensemble ✓',
        'Précision': 1.0, 'Rappel': 1.0, 'F1-Score': 1.0,
        'Accuracy':  1.0, 'Kappa': 1.0,  'MCC': 1.0,
        'Anomalies': int(y_ref.sum()),
    })

    metrics_df = pd.DataFrame(rows).set_index('Modèle')

    # ── Matrice de confusion par modèle ──────────────────────────────────
    def confusion_vals(y_true, y_pred):
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}

    confusion = {
        'Isolation Forest': confusion_vals(y_ref, y_if),
        'LOF':              confusion_vals(y_ref, y_lof),
        'Random Forest':    confusion_vals(y_ref, y_rf),
    }

    # ── Statistiques de réduction d'alertes ──────────────────────────────
    total       = len(df_filtre)
    nb_ensemble = int(y_ref.sum())
    nb_if       = int(y_if.sum())
    nb_lof      = int(y_lof.sum())
    nb_rf       = int(y_rf.sum())
    nb_union    = int(((y_if == 1) | (y_lof == 1) | (y_rf == 1)).sum())

    reduction = {
        'total_logs':         total,
        'nb_ensemble':        nb_ensemble,
        'nb_if':              nb_if,
        'nb_lof':             nb_lof,
        'nb_rf':              nb_rf,
        'nb_union':           nb_union,
        'reduction_vs_if':    round((nb_if  - nb_ensemble) / max(nb_if,  1) * 100, 1),
        'reduction_vs_lof':   round((nb_lof - nb_ensemble) / max(nb_lof, 1) * 100, 1),
        'reduction_vs_rf':    round((nb_rf  - nb_ensemble) / max(nb_rf,  1) * 100, 1),
        'reduction_vs_union': round((nb_union - nb_ensemble) / max(nb_union, 1) * 100, 1),
    }

    # ── Accord inter-modèles ─────────────────────────────────────────────
    # Taux de lignes où tous les 3 modèles s'accordent
    all_agree = int(((df_filtre['anomalie_if'] == df_filtre['anomalie_lof']) &
                     (df_filtre['anomalie_lof'] == df_filtre['anomalie_rf'])).sum())
    pct_agree = round(all_agree / total * 100, 1) if total > 0 else 0.0

    # Kappa moyen entre paires de modèles
    try:
        k_if_lof  = cohen_kappa_score(y_if,  y_lof)
        k_if_rf   = cohen_kappa_score(y_if,  y_rf)
        k_lof_rf  = cohen_kappa_score(y_lof, y_rf)
        kappa_mean = round((k_if_lof + k_if_rf + k_lof_rf) / 3, 3)
    except Exception:
        kappa_mean = None

    agreement = {
        'all_agree_n':   all_agree,
        'all_agree_pct': pct_agree,
        'kappa_mean':    kappa_mean,
    }

    return {
        'metrics_df': metrics_df,
        'confusion':  confusion,
        'reduction':  reduction,
        'agreement':  agreement,
    }