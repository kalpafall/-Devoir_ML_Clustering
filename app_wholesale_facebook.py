"""Application Streamlit - Clustering de deux jeux de données UCI.

Page 1 : clients d'un grossiste (Wholesale customers) - modèle Kmeans (2 classes)
Page 2 : publications Facebook de vendeurs thaïlandais - modèle CAH (3 classes)

Chaque page reproduit la démarche du notebook correspondant : nettoyage,
normalisation, entraînement du meilleur modèle, puis affectation d'une nouvelle
observation à la classe la plus proche.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

DOSSIER = Path(__file__).parent

st.set_page_config(page_title="Clustering - Wholesale et Facebook", page_icon="🧩", layout="wide")


# ---------------------------------------------------------------- Modèles


@st.cache_resource
def modele_wholesale():
    """Kmeans à 2 classes sur les dépenses normalisées (notebook Wholesale)."""
    df = pd.read_csv(DOSSIER / "Wholesale customers data.csv")
    infos = df[["Channel", "Region"]]
    df = df.drop(columns=["Channel", "Region"])
    x = normalize(df.values)
    km = KMeans(n_clusters=2, init="random", max_iter=500, n_init=10, random_state=42).fit(x)
    partition = km.labels_
    profil_norm = pd.DataFrame(x, columns=df.columns).groupby(partition).mean()
    classe_frais = int(profil_norm["Fresh"].idxmax())
    noms = {
        c: (
            "client « produits frais » (profil hôtel / restaurant / café)"
            if c == classe_frais
            else "client « épicerie, lait et détergents » (profil commerce de détail)"
        )
        for c in range(2)
    }
    return {
        "df": df,
        "infos": infos,
        "partition": partition,
        "centroides": km.cluster_centers_,
        "classes": np.arange(2),
        "noms": noms,
        "silhouette": silhouette_score(x, partition),
        "pca": PCA(n_components=2).fit_transform(x),
        "methode": "Kmeans (2 classes)",
    }


@st.cache_resource
def modele_facebook():
    """CAH à 3 classes (lien moyen) sur l'engagement normalisé (notebook Facebook)."""
    df = pd.read_csv(DOSSIER / "Live_20210128.csv").dropna(axis=1, how="all")
    colonnes = [c for c in df.columns if c.startswith("num_")]
    infos = df[["status_type"]]
    df = df[colonnes]
    garder = df.sum(axis=1) > 0
    df, infos = df[garder].reset_index(drop=True), infos[garder].reset_index(drop=True)
    x = normalize(df.values)
    partition = AgglomerativeClustering(n_clusters=3, linkage="average").fit_predict(x)
    classes = np.unique(partition)
    centroides = np.array([x[partition == c].mean(axis=0) for c in classes])
    profil_norm = pd.DataFrame(x, columns=colonnes).groupby(partition).mean()
    noms = {
        int(profil_norm["num_comments"].idxmax()): "publication très commentée (type live vidéo)",
        int(profil_norm["num_likes"].idxmax()): "publication surtout aimée (type photo)",
        int(profil_norm["num_shares"].idxmax()): "publication surtout partagée",
    }
    return {
        "df": df,
        "infos": infos,
        "partition": partition,
        "centroides": centroides,
        "classes": classes,
        "noms": noms,
        "silhouette": silhouette_score(x, partition),
        "pca": PCA(n_components=2).fit_transform(x),
        "methode": "CAH, lien moyen (3 classes)",
    }


def predire_classe(valeurs, modele):
    """Normaliser l'observation puis prendre la classe du centroïde le plus proche."""
    obs = normalize(np.array(valeurs, dtype=float).reshape(1, -1))
    return int(modele["classes"][cdist(obs, modele["centroides"]).argmin()])


# ---------------------------------------------------------------- Affichage


def formulaire(modele, libelles, cle, unite=""):
    """Un champ par variable, pré-rempli avec la médiane ; renvoie les valeurs si envoyé."""
    df = modele["df"]
    with st.form(cle):
        colonnes_ecran = st.columns(3)
        valeurs = []
        for i, col in enumerate(df.columns):
            with colonnes_ecran[i % 3]:
                valeurs.append(
                    st.number_input(
                        libelles.get(col, col) + unite,
                        min_value=0,
                        value=int(df[col].median()),
                        step=1,
                        key=f"{cle}_{col}",
                    )
                )
        envoyer = st.form_submit_button("Prédire la classe", type="primary")
    return valeurs if envoyer else None


def afficher_resultat(valeurs, modele):
    if sum(valeurs) == 0:
        st.warning("Toutes les valeurs sont à zéro : saisissez au moins une valeur.")
        return
    classe = predire_classe(valeurs, modele)
    st.success(f"C'est une **{modele['noms'][classe]}** (classe {classe}).".replace("une **client", "un **client"))


def afficher_modele(modele, libelles, colonne_info, titre_info):
    df, partition, noms = modele["df"], modele["partition"], modele["noms"]
    etiquettes = [f"{c} - {noms[int(c)]}" for c in range(len(modele["classes"]))]

    c1, c2, c3 = st.columns(3)
    c1.metric("Observations", len(df))
    c2.metric("Modèle", modele["methode"])
    c3.metric("Indice de silhouette", f"{modele['silhouette']:.3f}")

    st.subheader("Effectif de chaque classe")
    effectifs = pd.Series(partition).value_counts().sort_index()
    st.dataframe(
        pd.DataFrame({"Classe": [etiquettes[c] for c in effectifs.index], "Effectif": effectifs.values}),
        hide_index=True,
    )

    st.subheader("Profil moyen de chaque classe")
    profil = df.groupby(partition).mean().round(1)
    profil.index = [etiquettes[c] for c in profil.index]
    profil.columns = [libelles.get(c, c) for c in profil.columns]
    st.dataframe(profil.T)

    st.subheader(titre_info)
    croise = pd.crosstab(
        pd.Series([etiquettes[c] for c in partition], name="Classe"),
        pd.Series(modele["infos"][colonne_info].values, name="Valeur réelle"),
    )
    st.dataframe(croise)

    st.subheader("Les classes sur les deux premières composantes principales")
    points = pd.DataFrame(modele["pca"], columns=["PC1", "PC2"])
    points["Classe"] = [noms[int(c)] for c in partition]
    st.scatter_chart(points, x="PC1", y="PC2", color="Classe")


# ---------------------------------------------------------------- Pages

page = st.sidebar.radio(
    "Jeu de données",
    ["Clients d'un grossiste", "Publications Facebook"],
)
st.sidebar.markdown(
    "Sources : UCI Machine Learning Repository  \n"
    "- [Wholesale customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers)  \n"
    "- [Facebook Live Sellers in Thailand](https://archive.ics.uci.edu/dataset/488/facebook+live+sellers+in+thailand)"
)

if page == "Clients d'un grossiste":
    modele = modele_wholesale()
    libelles = {
        "Fresh": "Produits frais",
        "Milk": "Produits laitiers",
        "Grocery": "Épicerie",
        "Frozen": "Surgelés",
        "Detergents_Paper": "Détergents et papier",
        "Delicassen": "Épicerie fine",
    }
    st.title("🛒 Segmentation des clients d'un grossiste")
    st.write(
        "Saisissez les **dépenses annuelles** d'un client (en unités monétaires) dans chaque catégorie "
        "pour connaître son segment. Le modèle compare la **répartition** des achats entre les "
        "catégories, pas leur montant total."
    )
    onglet_prediction, onglet_modele = st.tabs(["Prédire un segment", "À propos du modèle"])
    with onglet_prediction:
        valeurs = formulaire(modele, libelles, "wholesale")
        if valeurs is not None:
            afficher_resultat(valeurs, modele)
    with onglet_modele:
        modele_infos = dict(modele)
        modele_infos["infos"] = modele["infos"].replace({"Channel": {1: "Horeca", 2: "Commerce de détail"}})
        afficher_modele(modele_infos, libelles, "Channel", "Comparaison avec le canal de vente réel")

else:
    modele = modele_facebook()
    libelles = {
        "num_reactions": "Réactions (total)",
        "num_comments": "Commentaires",
        "num_shares": "Partages",
        "num_likes": "J'aime",
        "num_loves": "J'adore",
        "num_wows": "Wouah",
        "num_hahas": "Haha",
        "num_sads": "Triste",
        "num_angrys": "Grrr",
    }
    st.title("📱 Segmentation des publications Facebook")
    st.write(
        "Saisissez les **indicateurs d'engagement** d'une publication pour savoir à quel type "
        "d'engagement elle correspond. Le total des réactions est en principe la somme des "
        "j'aime, j'adore, wouah, haha, triste et grrr."
    )
    onglet_prediction, onglet_modele = st.tabs(["Prédire une classe", "À propos du modèle"])
    with onglet_prediction:
        valeurs = formulaire(modele, libelles, "facebook")
        if valeurs is not None:
            afficher_resultat(valeurs, modele)
    with onglet_modele:
        afficher_modele(modele, libelles, "status_type", "Comparaison avec le type de publication")
