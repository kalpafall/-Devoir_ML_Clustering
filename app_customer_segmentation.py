"""Application Streamlit - Segmentation des clients (modèle CAH du notebook ML5).

Même démarche que dans Customer_Segmentation.ipynb :
normalisation -> CAH (2 classes, lien moyen) -> centroïdes -> affectation
d'un nouveau client à la classe du centroïde le plus proche.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

FICHIER = Path(__file__).parent / "Customer_dataset.csv"

st.set_page_config(page_title="Segmentation des clients", page_icon="📊", layout="wide")


@st.cache_data
def charger_donnees():
    """Lire le jeu de données clients."""
    return pd.read_csv(FICHIER)


@st.cache_resource
def entrainer_modele(df):
    """Reproduire le modèle CAH du notebook et calculer les centroïdes."""
    x = normalize(df.values)
    partition = AgglomerativeClustering(n_clusters=2, linkage="average").fit_predict(x)
    classes = np.unique(partition)
    centroides = np.array([x[partition == c].mean(axis=0) for c in classes])

    # Nommer les classes d'après leur profil : la classe qui utilise le moins
    # le service est "peu fidèle", l'autre est "fidèle".
    usage = df.groupby(partition)["Seconds of Use"].mean()
    noms = {int(usage.idxmin()): "peu fidèle", int(usage.idxmax()): "fidèle"}

    composantes = PCA(n_components=2).fit_transform(x)
    return {
        "partition": partition,
        "classes": classes,
        "centroides": centroides,
        "noms": noms,
        "silhouette": silhouette_score(x, partition),
        "pca": composantes,
    }


def predire_classe(valeurs, modele):
    """Classe du centroïde le plus proche du nouveau client."""
    client = normalize(np.array(valeurs, dtype=float).reshape(1, -1))
    distances = cdist(client, modele["centroides"])
    return int(modele["classes"][distances.argmin()])


def libelle(colonne):
    """Nom de colonne sans les doubles espaces."""
    return " ".join(colonne.split())


df = charger_donnees()
modele = entrainer_modele(df)

st.title("📊 Segmentation des clients")
st.write(
    "Saisissez les caractéristiques d'un client d'une entreprise de télécommunications "
    "pour connaître son segment. Le modèle utilisé est une **CAH** "
    "(classification ascendante hiérarchique)."
)

onglet_prediction, onglet_modele = st.tabs(["Prédire un segment", "À propos du modèle"])

with onglet_prediction:
    with st.form("formulaire_client"):
        st.subheader("Caractéristiques du client")
        colonnes_ecran = st.columns(3)
        valeurs = []
        for i, col in enumerate(df.columns):
            entier = pd.api.types.is_integer_dtype(df[col])
            with colonnes_ecran[i % 3]:
                if entier:
                    v = st.number_input(
                        libelle(col), min_value=0, value=int(df[col].median()), step=1
                    )
                else:
                    v = st.number_input(
                        libelle(col), min_value=0.0, value=float(df[col].median()), step=1.0
                    )
            valeurs.append(v)
        envoyer = st.form_submit_button("Prédire le segment", type="primary")

    if envoyer:
        if sum(valeurs) == 0:
            st.warning("Toutes les valeurs sont à zéro : saisissez au moins une caractéristique.")
        else:
            classe = predire_classe(valeurs, modele)
            nom = modele["noms"][classe]
            if nom == "fidèle":
                st.success(f"C'est un client **{nom}** (classe {classe}).")
            else:
                st.error(f"C'est un client **{nom}** (classe {classe}).")

with onglet_modele:
    effectifs = pd.Series(modele["partition"]).value_counts().sort_index()
    c1, c2, c3 = st.columns(3)
    c1.metric("Clients dans le jeu de données", len(df))
    c2.metric("Nombre de classes", len(modele["classes"]))
    c3.metric("Indice de silhouette", f"{modele['silhouette']:.3f}")

    st.subheader("Effectif de chaque classe")
    tableau = pd.DataFrame(
        {
            "Classe": [f"{c} - {modele['noms'][int(c)]}" for c in effectifs.index],
            "Nombre de clients": effectifs.values,
        }
    )
    st.dataframe(tableau, hide_index=True)

    st.subheader("Profil moyen de chaque classe")
    profil = df.groupby(modele["partition"]).mean().round(2)
    profil.index = [f"{c} - {modele['noms'][int(c)]}" for c in profil.index]
    profil.columns = [libelle(c) for c in profil.columns]
    st.dataframe(profil.T)

    st.subheader("Les classes sur les deux premières composantes principales")
    points = pd.DataFrame(modele["pca"], columns=["PC1", "PC2"])
    points["Classe"] = [modele["noms"][int(c)] for c in modele["partition"]]
    st.scatter_chart(points, x="PC1", y="PC2", color="Classe")
