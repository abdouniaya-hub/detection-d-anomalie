"""
dashboard/dashboard.py — Onglet Dashboard avec palette warm/dark glassmorphism
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# PALETTE — warm dark glassmorphism (inspiré de l'image)
# ============================================================
COLORS = {
    'primary':        '#c8b48a',   # beige doré — barres principales
    'secondary':      '#a09070',   # taupe — barres secondaires
    'primary_dark':   '#e8d8b8',   # beige clair — textes forts
    'secondary_dark': '#b8a888',   # sable — accents
    'alert':          '#d06050',   # terre cuite — erreurs / anomalies
    'success':        '#70b890',   # vert sauge — normaux
    'warning':        '#c8983a',   # ambre warm — warnings
    'gray':           '#807868',   # gris warm — neutres
}

# Niveaux de log ordonnés
ORDRE_NIVEAUX = ['V', 'D', 'I', 'W', 'E', 'F']

# Couleurs par niveau (warm dark)
COULEURS_NIVEAUX = {
    'V': '#7aaec8',   # bleu-ardoise
    'D': '#8090c0',   # bleu-lavande désaturé
    'I': '#9888c0',   # lilas doux
    'W': '#c8983a',   # ambre warm
    'E': '#d06050',   # terre cuite
    'F': '#a03828',   # rouge brique
}

def _rgba(r, g, b, a=1.0):
    """Couleur Matplotlib (tuples 0–1) — rcParams n'accepte pas les chaînes CSS rgba()."""
    return (r / 255, g / 255, b / 255, a)


MPL_TEXT = _rgba(240, 235, 220, 0.75)
MPL_TEXT_MUTED = _rgba(240, 235, 220, 0.40)
MPL_TEXT_STRONG = _rgba(240, 235, 220, 0.90)
MPL_DARK = _rgba(26, 23, 20, 0.8)
MPL_DARK_SOFT = _rgba(26, 23, 20, 0.4)

# Configuration Matplotlib globale — fond transparent, tons warm
plt.rcParams.update({
    "axes.facecolor":   "none",
    "figure.facecolor": "none",
    "axes.edgecolor":   _rgba(245, 240, 232, 0.15),
    "axes.labelcolor":  _rgba(240, 235, 220, 0.55),
    "xtick.color":      _rgba(240, 235, 220, 0.40),
    "ytick.color":      _rgba(240, 235, 220, 0.40),
    "text.color":       _rgba(240, 235, 220, 0.70),
    "font.family":      "sans-serif",
    "grid.color":       _rgba(245, 240, 232, 0.06),
    "grid.linewidth":   0.5,
    "axes.grid":        True,
})


# ============================================================
# HELPER : CARTE KPI
# ============================================================
def render_kpi_card(label, value, subtext="", color=None):
    """Affiche une carte KPI avec le style glassmorphism warm."""
    value_color = color if color else "#f0ebe0"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{value_color};">{value}</div>
            <div class="kpi-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HELPER : STYLE LIGNES DATAFRAME
# ============================================================
def styler_lignes(row):
    """Applique un style CSS sur toute la ligne selon le niveau ou l'anomalie."""
    niveau   = row.get("Niveau",   "")
    anomalie = row.get("Anomalie", 0)

    if niveau in ["E", "F"]:
        return [
            "background-color: rgba(208, 96, 80, 0.14); color: #e08878;"
        ] * len(row)
    elif niveau == "W":
        return [
            "background-color: rgba(200, 152, 58, 0.14); color: #d4a860;"
        ] * len(row)
    elif anomalie == -1:
        return [
            "background-color: rgba(200, 180, 138, 0.10); color: #c8b48a;"
        ] * len(row)
    return [""] * len(row)


# ============================================================
# DASHBOARD PRINCIPAL
# ============================================================
def afficher_dashboard(df, df_filtre, anomalies, normaux):
    """
    Affiche le contenu du Dashboard (sans les onglets).
    Les autres onglets sont gérés par main.py.
    """

    total_filtre    = len(df_filtre)
    total_anomalies = len(anomalies)
    total_normaux   = len(normaux)
    taux = round((total_anomalies / total_filtre) * 100, 1) if total_filtre > 0 else 0

    # ---- Bannière d'alerte ----
    if taux >= 10:
        st.error(f"⚠ ALERTE CRITIQUE — taux d'anomalies élevé ({taux}%). Intervention recommandée.")
    elif taux > 0:
        st.warning(f"⚠ AVERTISSEMENT — {total_anomalies} anomalie(s) détectée(s) ({taux}%).")
    elif total_filtre > 0:
        st.success("✓ SYSTÈME STABLE — aucune anomalie critique détectée.")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ============================================================
    # CARTES KPI
    # ============================================================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_kpi_card("📊 TOTAL LOGS",  total_filtre,    "Périmètre filtré")
    with col2:
        render_kpi_card("✓ NORMAUX",      total_normaux,   "Activité saine",   color="#70b890")
    with col3:
        render_kpi_card("⚠ ANOMALIES",   total_anomalies, "Détections ML",    color="#d06050")
    with col4:
        render_kpi_card("TAUX",           f"{taux}%",      "Part des anomalies")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ============================================================
    # GRAPHIQUES — rangée 1
    # ============================================================
    col_g1, col_g2 = st.columns(2)

    # ---- Graphique 1 : Volume par source ----
    with col_g1:
        st.markdown(
            '<div class="section-titre">Volume par source</div>',
            unsafe_allow_html=True,
        )
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        fig1.patch.set_alpha(0.0)

        if not df_filtre.empty:
            sources_count = df_filtre["source"].value_counts()
            bar_colors = [COLORS["primary"], COLORS["secondary"],
                          COLORS["secondary_dark"], COLORS["gray"]]

            bars = ax1.bar(
                sources_count.index,
                sources_count.values,
                color=bar_colors[: len(sources_count)],
                width=0.5,
                zorder=3,
            )

            max_val = max(sources_count.values)
            ax1.set_ylim(0, max_val * 1.18)
            ax1.set_ylabel("Nombre de logs", fontsize=10)
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.spines["left"].set_alpha(0.15)
            ax1.spines["bottom"].set_alpha(0.15)

            for bar in bars:
                h = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + max_val * 0.025,
                    str(int(h)),
                    ha="center",
                    va="bottom",
                    color=MPL_TEXT,
                    fontsize=10,
                    fontweight="500",
                )
        else:
            ax1.text(0.5, 0.5, "Aucune donnée", ha="center", va="center",
                     color=MPL_TEXT_MUTED, transform=ax1.transAxes)
            ax1.axis("off")

        fig1.tight_layout(pad=1.2)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    # ---- Graphique 2 : Répartition normal / anomalie ----
    with col_g2:
        st.markdown(
            '<div class="section-titre">Normal / Anomalie</div>',
            unsafe_allow_html=True,
        )
        fig2, ax2 = plt.subplots(figsize=(5, 3.8))
        fig2.patch.set_alpha(0.0)

        sizes = [total_normaux, total_anomalies]

        if sum(sizes) > 0:
            pie_colors = [COLORS["success"], COLORS["alert"]]

            wedges, texts, autotexts = ax2.pie(
                sizes,
                labels=["Normal", "Anomalie"],
                colors=pie_colors,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"color": MPL_TEXT, "fontsize": 10},
                wedgeprops={
                    "edgecolor": MPL_DARK,
                    "linewidth": 1.5,
                    "alpha": 0.85,
                },
            )
            for at in autotexts:
                at.set_color(MPL_TEXT_STRONG)
                at.set_fontweight("500")

            # Cercle central donut
            centre = plt.Circle((0, 0), 0.65, fc=MPL_DARK_SOFT)
            ax2.add_artist(centre)
            ax2.axis("equal")
        else:
            ax2.text(0.5, 0.5, "Aucune donnée", ha="center", va="center",
                     color=MPL_TEXT_MUTED, transform=ax2.transAxes)
            ax2.axis("off")

        fig2.tight_layout(pad=1.2)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ============================================================
    # GRAPHIQUE 3 : Distribution par niveau
    # ============================================================
    st.markdown(
        '<div class="section-titre">Distribution par niveau</div>',
        unsafe_allow_html=True,
    )
    fig3, ax3 = plt.subplots(figsize=(11, 4))
    fig3.patch.set_alpha(0.0)

    if not df_filtre.empty:
        ordre  = [n for n in ORDRE_NIVEAUX if n in df_filtre["niveau"].values]
        counts = [len(df_filtre[df_filtre["niveau"] == n]) for n in ordre]
        colors = [COULEURS_NIVEAUX.get(n, COLORS["gray"]) for n in ordre]

        bars3 = ax3.bar(ordre, counts, color=colors, width=0.5, zorder=3, alpha=0.85)

        ax3.set_xlabel("Niveau", fontsize=10)
        ax3.set_ylabel("Nombre d'événements", fontsize=10)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.spines["left"].set_alpha(0.15)
        ax3.spines["bottom"].set_alpha(0.15)

        max_count = max(counts) if counts else 1
        ax3.set_ylim(0, max_count * 1.18)

        for bar, count in zip(bars3, counts):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_count * 0.025,
                str(count),
                ha="center",
                color=MPL_TEXT,
                fontsize=10,
                fontweight="500",
            )
    else:
        ax3.text(0.5, 0.5, "Aucune donnée", ha="center", va="center",
                 color=MPL_TEXT_MUTED, transform=ax3.transAxes)
        ax3.axis("off")

    fig3.tight_layout(pad=1.2)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ============================================================
    # DERNIERS LOGS
    # ============================================================
    st.markdown(
        '<div class="section-titre">📋 Derniers logs</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Afficher les 10 derniers logs"):
        if not df_filtre.empty:
            df_recent = df_filtre.tail(10)[
                ["source", "date", "heure", "niveau", "message", "anomalie"]
            ].rename(
                columns={
                    "source":   "Source",
                    "date":     "Date",
                    "heure":    "Heure",
                    "niveau":   "Niveau",
                    "message":  "Message",
                    "anomalie": "Anomalie",
                }
            )
            st.dataframe(
                df_recent.style.apply(styler_lignes, axis=1),
                use_container_width=True,
            )
        else:
            st.info("Aucun log à afficher.")