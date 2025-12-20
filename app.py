#streamlit run app.py


# ===============================================
# ⚽ Streamlit Dashboard – Expected Goals Explorer
# ===============================================
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="xG Dashboard", layout="wide")
st.title("⚽ Expected Goals (xG) Dashboard")

# ===============================================
# 🧠 Chargement et préparation des données
# ===============================================
DATA_PATH = Path("data/shots_with_xg.csv")
df = pd.read_csv(DATA_PATH)

# Si aucune colonne "shooter_name" n’existe, on en crée une avec 100 joueurs fictifs
if "shooter_name" not in df.columns:
    import numpy as np

    np.random.seed(42)
    players = [f"Joueur_{i:03d}" for i in range(1, 101)]
    df["shooter_name"] = np.random.choice(players, size=len(df))

    # Sauvegarde rapide dans un nouveau CSV pour réutilisation
    df.to_csv("data/shots_with_xg_players.csv", index=False)

else:
    st.info("✅ Colonne 'shooter_name' détectée — chargement du dataset existant.")

# ===============================================
# 🗂️ Création des onglets
# ===============================================
tabs = st.tabs(["📊 Vue générale","👥 Comparaison joueurs"])

# ===============================================
# 🟢 Onglet 1 – Vue générale
# ===============================================
with tabs[0]:
    st.subheader("📊 Vue générale par match")

    # --- Sélection du match
    matches = df["match_id"].unique()
    selected_match = st.selectbox("📅 Sélectionne un match :", matches)
    filtered = df[df["match_id"] == selected_match]

    # --- Statistiques agrégées
    col3, col4, col5 = st.columns(3)
    total_shots = len(filtered)
    total_goals = int(filtered["is_goal"].sum())
    total_xg = round(filtered["xG_pred"].sum(), 2)
    diff = round(total_goals - total_xg, 2)

    col3.metric("Nombre de tirs", total_shots)
    col4.metric("Buts réels", total_goals)
    col5.metric("xG cumulé", total_xg)

    st.write(f"🔍 **Différence buts - xG = {diff:+.2f}** "
             f"({'overperformance' if diff > 0 else 'underperformance' if diff < 0 else 'performance attendue'})")

    # --- Barplot Buts vs xG
    st.subheader("📈 Comparaison Buts réels vs xG cumulés")

    bar_data = pd.DataFrame({
        "Type": ["Buts réels", "xG cumulés"],
        "Valeur": [total_goals, total_xg]
    })

    fig_bar, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=bar_data, x="Type", y="Valeur", palette=["dodgerblue", "red"], ax=ax)
    for i, val in enumerate(bar_data["Valeur"]):
        ax.text(i, val + 0.05, f"{val:.2f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(bar_data["Valeur"]) * 1.2)
    ax.set_ylabel("Total")
    st.pyplot(fig_bar)

    # --- Carte des tirs
    st.subheader("🎯 Carte des tirs et xG")
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.gca().set_facecolor("#E6E6E6")
    plt.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color="black", linewidth=2)
    plt.plot([100, 100], [36.8, 63.2], color="red", linewidth=4)

    sns.scatterplot(
        data=filtered,
        x="x", y="y",
        hue="is_goal",
        size="xG_pred",
        sizes=(30, 400),
        alpha=0.6,
        palette={0: "dodgerblue", 1: "red"},
        ax=ax
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.xlabel("Position X (0 = notre but, 100 = but adverse)")
    plt.ylabel("Position Y (largeur du terrain)")
    st.pyplot(fig)


# ===============================================
# 🔵 Onglet 3 – Comparaison joueurs (tous matchs)
# ===============================================
with tabs[1]:
    st.subheader("👥 Comparaison de deux joueurs (tous matchs confondus)")

    players_all = sorted(df["shooter_name"].dropna().unique().tolist())
    if len(players_all) < 2:
        st.warning("Pas assez de joueurs pour la comparaison.")
        st.stop()

    colsel1, colsel2 = st.columns(2)
    player_a = colsel1.selectbox("Joueur A", players_all, index=0)
    player_b = colsel2.selectbox("Joueur B", players_all, index=1)

    df_a = df[df["shooter_name"] == player_a]
    df_b = df[df["shooter_name"] == player_b]

    # Agrégats
    def agg_stats(d):
        return {
            "shots": len(d),
            "goals": int(d["is_goal"].sum()),
            "xg": float(d["xG_pred"].sum())
        }

    stats_a = agg_stats(df_a)
    stats_b = agg_stats(df_b)

    # Statistiques principales
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric(f"{player_a} - Tirs", stats_a["shots"])
    col2.metric(f"{player_a} - Buts", stats_a["goals"])
    col3.metric(f"{player_a} - xG", round(stats_a["xg"], 2))
    col4.metric(f"{player_b} - Tirs", stats_b["shots"])
    col5.metric(f"{player_b} - Buts", stats_b["goals"])
    col6.metric(f"{player_b} - xG", round(stats_b["xg"], 2))

    # Barplot comparatif
    st.subheader("📈 Buts réels vs xG cumulés")
    comp_df = pd.DataFrame({
        "Joueur": [player_a, player_a, player_b, player_b],
        "Type": ["Buts", "xG", "Buts", "xG"],
        "Valeur": [stats_a["goals"], stats_a["xg"], stats_b["goals"], stats_b["xg"]],
    })

    figc, axc = plt.subplots(figsize=(6, 4))
    sns.barplot(data=comp_df, x="Joueur", y="Valeur", hue="Type", ax=axc, palette=["dodgerblue", "red"])
    for p in axc.patches:
        axc.annotate(f"{p.get_height():.2f}",
                     (p.get_x() + p.get_width()/2, p.get_height()),
                     ha="center", va="bottom", fontsize=9)
    axc.set_ylabel("Total")
    axc.set_xlabel("")
    axc.legend(title="")
    st.pyplot(figc)

    # Cartes de tirs côte à côte
    st.subheader("🎯 Cartes de tirs")
    colmap1, colmap2 = st.columns(2)
    with colmap1:
        st.caption(f"{player_a}")
        figA, axA = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=df_a, x="x", y="y",
            hue="is_goal", size="xG_pred",
            sizes=(30, 300), alpha=0.6,
            palette={0: "dodgerblue", 1: "red"}, ax=axA
        )
        axA.set_xlim(0, 100)
        axA.set_ylim(0, 100)
        axA.set_title(f"Tirs de {player_a}")
        st.pyplot(figA)
    with colmap2:
        st.caption(f"{player_b}")
        figB, axB = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=df_b, x="x", y="y",
            hue="is_goal", size="xG_pred",
            sizes=(30, 300), alpha=0.6,
            palette={0: "dodgerblue", 1: "red"}, ax=axB
        )
        axB.set_xlim(0, 100)
        axB.set_ylim(0, 100)
        axB.set_title(f"Tirs de {player_b}")
        st.pyplot(figB)

    # Tableau over/under-performance
    st.subheader("📊 Over / Under-performance")
    perf2 = pd.DataFrame({
        "Joueur": [player_a, player_b],
        "Buts": [stats_a["goals"], stats_b["goals"]],
        "xG": [round(stats_a["xg"], 2), round(stats_b["xg"], 2)],
    }).assign(Diff=lambda d: d["Buts"] - d["xG"])
    st.dataframe(perf2.style.background_gradient(cmap="coolwarm", subset=["Diff"]))
