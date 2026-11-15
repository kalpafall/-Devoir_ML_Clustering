"""Modèles et interface communs aux quatre pages de l'application.

Chaque jeu de données suit la démarche de son notebook : nettoyage, normalisation
(sauf Wheat Seeds, comme dans le notebook), entraînement du meilleur modèle, puis
affectation d'une nouvelle observation à la classe du centroïde le plus proche.
"""
from io import StringIO
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

DOSSIER = Path(__file__).parent

# Une couleur par classe (les mêmes valeurs que dans .streamlit/config.toml)
PALETTE = ["#B4441C", "#2F5D8A", "#4F7F52", "#8A5A9E"]
COULEURS_BADGE = ["orange", "blue", "green", "violet"]
ENCRE = "#1C1917"


# ---------------------------------------------------------------- Description des jeux de données

JEUX = {
    "clients": {
        "titre": "Clients d'un opérateur télécom",
        "nouveau": False,
        "source": "Jeu de données du cours (notebook ML5)",
        "lien": None,
        "fichier": "Customer_dataset.csv",
        "resume": "3 150 abonnés décrits par leur usage du service : appels, SMS, ancienneté, facturation.",
        "question": "Ce client est-il fidèle ?",
        "unite": "client",
        "normaliser": True,
        "libelles": {
            "Call  Failure": "Appels échoués",
            "Complains": "Réclamation (0 ou 1)",
            "Subscription  Length": "Ancienneté (mois)",
            "Charge  Amount": "Niveau de facturation (0 à 9)",
            "Seconds of Use": "Durée d'appel (secondes)",
            "Frequency of use": "Nombre d'appels",
            "Frequency of SMS": "Nombre de SMS",
            "Distinct Called Numbers": "Numéros distincts appelés",
            "Age Group": "Tranche d'âge (1 à 5)",
            "Status": "Statut (1 actif, 2 inactif)",
            "Age": "Âge",
            "Customer Value": "Valeur client",
        },
    },
    "graines": {
        "titre": "Graines de blé",
        "nouveau": False,
        "source": "Jeu de données du cours (Wheat Seeds)",
        "lien": "https://archive.ics.uci.edu/dataset/236/seeds",
        "fichier": "wheat_seeds_dataset.csv",
        "resume": "199 graines de trois variétés de blé, mesurées par radiographie : surface, périmètre, forme.",
        "question": "Quelle taille de graine ?",
        "unite": "graine",
        "normaliser": False,
        "libelles": {
            "area A": "Surface",
            "perimeter": "Périmètre",
            "compactness": "Compacité",
            "length of kernel": "Longueur du grain",
            "width of kernel": "Largeur du grain",
            "asymmetry coefficient": "Coefficient d'asymétrie",
            "length of kernel groove": "Longueur du sillon",
        },
    },
    "grossiste": {
        "titre": "Clients d'un grossiste",
        "nouveau": True,
        "source": "UCI Machine Learning Repository · id 292",
        "lien": "https://archive.ics.uci.edu/dataset/292/wholesale+customers",
        "fichier": "Wholesale customers data.csv",
        "resume": "440 clients d'un grossiste portugais et leurs dépenses annuelles dans six catégories de produits.",
        "question": "Quel profil d'acheteur ?",
        "unite": "client",
        "normaliser": True,
        "libelles": {
            "Fresh": "Produits frais",
            "Milk": "Produits laitiers",
            "Grocery": "Épicerie",
            "Frozen": "Surgelés",
            "Detergents_Paper": "Détergents et papier",
            "Delicassen": "Épicerie fine",
        },
    },
    "facebook": {
        "titre": "Publications Facebook",
        "nouveau": True,
        "source": "UCI Machine Learning Repository · id 488",
        "lien": "https://archive.ics.uci.edu/dataset/488/facebook+live+sellers+in+thailand",
        "fichier": "Live_20210128.csv",
        "resume": "7 050 publications de dix vendeurs thaïlandais en direct, avec leurs réactions, commentaires et partages.",
        "question": "Quel type d'engagement ?",
        "unite": "publication",
        "normaliser": True,
        "libelles": {
            "num_reactions": "Réactions (total)",
            "num_comments": "Commentaires",
            "num_shares": "Partages",
            "num_likes": "J'aime",
            "num_loves": "J'adore",
            "num_wows": "Wouah",
            "num_hahas": "Haha",
            "num_sads": "Triste",
            "num_angrys": "Grrr",
        },
    },
}


# ---------------------------------------------------------------- Chargement et nettoyage


def _charger(cle):
    """Lire et nettoyer le jeu de données comme dans le notebook ; renvoie (variables, infos)."""
    df = pd.read_csv(DOSSIER / JEUX[cle]["fichier"])
    infos = None
    if cle == "grossiste":
        # Channel et Region sont des codes : retirés du clustering, gardés pour l'interprétation
        infos = df["Channel"].map({1: "Hôtel / restaurant / café", 2: "Commerce de détail"}).rename("Canal de vente")
        df = df.drop(columns=["Channel", "Region"])
    elif cle == "facebook":
        df = df.dropna(axis=1, how="all")
        infos = df["status_type"].rename("Type de publication")
        df = df[[c for c in df.columns if c.startswith("num_")]]
        garder = df.sum(axis=1) > 0  # publications sans aucun engagement retirées
        df, infos = df[garder].reset_index(drop=True), infos[garder].reset_index(drop=True)
    return df, infos


def _preparer(cle, valeurs):
    """Mettre les observations à l'échelle utilisée par le modèle."""
    x = np.asarray(valeurs, dtype=float)
    return normalize(x) if JEUX[cle]["normaliser"] else x


# ---------------------------------------------------------------- Entraînement


def _nommer(cle, df, x, partition):
    """Nom court et description de chaque classe, d'après son profil moyen."""
    if cle == "clients":
        usage = df.groupby(partition)["Seconds of Use"].mean()
        return {
            int(usage.idxmax()): ("Fidèle", "Utilise beaucoup le service : appels longs, nombreux numéros, forte valeur."),
            int(usage.idxmin()): ("Peu fidèle", "Utilise peu le service : faible durée d'appel et faible valeur client."),
        }
    if cle == "graines":
        ordre = df.groupby(partition)["area A"].mean().sort_values().index
        textes = [
            ("Petite graine", "Surface et longueur faibles, sillon court."),
            ("Graine moyenne", "Mesures intermédiaires entre les deux autres groupes."),
            ("Grosse graine", "Surface, périmètre et longueur les plus élevés."),
        ]
        return {int(c): t for c, t in zip(ordre, textes)}
    profil = pd.DataFrame(x, columns=df.columns).groupby(partition).mean()
    if cle == "grossiste":
        frais = int(profil["Fresh"].idxmax())
        return {
            frais: ("Produits frais", "Achats dominés par les produits frais : profil hôtel, restaurant, café."),
            1 - frais: ("Épicerie et entretien", "Épicerie, produits laitiers et détergents : profil commerce de détail."),
        }
    return {
        int(profil["num_comments"].idxmax()): ("Très commentée", "Beaucoup de commentaires : surtout des vidéos en direct, où l'on commente pour commander."),
        int(profil["num_likes"].idxmax()): ("Surtout aimée", "L'engagement passe par les j'aime : surtout des photos, statuts et liens."),
        int(profil["num_shares"].idxmax()): ("Surtout partagée", "Une part inhabituelle de partages."),
    }


@st.cache_resource(show_spinner="Entraînement du modèle…")
def modele(cle):
    """Entraîner le meilleur modèle du notebook et calculer ce dont l'interface a besoin."""
    df, infos = _charger(cle)
    x = _preparer(cle, df.values)
    if cle == "grossiste":
        km = KMeans(n_clusters=2, init="random", max_iter=500, n_init=10, random_state=42).fit(x)
        partition, centroides = km.labels_, km.cluster_centers_
        methode, detail = "Kmeans", "2 classes"
    else:
        n = 2 if cle == "clients" else 3
        partition = AgglomerativeClustering(n_clusters=n, linkage="average").fit_predict(x)
        centroides = np.array([x[partition == c].mean(axis=0) for c in range(n)])
        methode, detail = "CAH", f"lien moyen, {n} classes"
    acp = PCA(n_components=2).fit(x)
    return {
        "df": df,
        "infos": infos,
        "partition": partition,
        "centroides": centroides,
        "noms": _nommer(cle, df, x, partition),
        "silhouette": float(silhouette_score(x, partition)),
        "acp": acp,
        "plan": acp.transform(x),
        "methode": methode,
        "detail": detail,
    }


def predire(cle, m, valeurs):
    """Classe du centroïde le plus proche et proximité relative à chaque classe."""
    x = _preparer(cle, np.atleast_2d(valeurs))
    distances = cdist(x, m["centroides"])
    inverses = 1 / np.maximum(distances, 1e-12)
    proximites = inverses / inverses.sum(axis=1, keepdims=True)
    return distances.argmin(axis=1), proximites, m["acp"].transform(x)


# ---------------------------------------------------------------- Éléments d'interface


def _libelle(cle, col):
    return JEUX[cle]["libelles"].get(col, col)


def _nom(m, c):
    return m["noms"][int(c)][0]


def _carte_acp(m, nouveaux=None, hauteur=360):
    """Nuage de points de l'ACP, avec les nouvelles observations en surimpression."""
    noms = [m["noms"][c][0] for c in sorted(m["noms"])]
    points = pd.DataFrame(m["plan"], columns=["Axe 1", "Axe 2"])
    points["Classe"] = [_nom(m, c) for c in m["partition"]]
    couleur = alt.Color("Classe:N", scale=alt.Scale(domain=noms, range=PALETTE[: len(noms)]), legend=alt.Legend(orient="bottom", title=None, columns=2, labelLimit=220))
    fond = alt.Chart(points).mark_circle(size=22, opacity=0.45).encode(
        x=alt.X("Axe 1:Q", axis=alt.Axis(grid=False, ticks=False, labels=False)),
        y=alt.Y("Axe 2:Q", axis=alt.Axis(grid=False, ticks=False, labels=False)),
        color=couleur,
        tooltip=["Classe"],
    )
    graphique = fond
    if nouveaux is not None and len(nouveaux):
        repere = alt.Chart(nouveaux).mark_point(shape="diamond", size=280, filled=True, color=ENCRE, stroke="#FAF8F4", strokeWidth=1.5).encode(
            x="Axe 1:Q", y="Axe 2:Q", tooltip=["Observation", "Classe"]
        )
        graphique = fond + repere
    return graphique.properties(height=hauteur).configure_view(strokeWidth=0)


def en_tete(cle, m):
    """Badge d'origine, titre, résumé et chiffres clés du modèle."""
    jeu = JEUX[cle]
    if jeu["nouveau"]:
        st.badge("Nouveau jeu de données", icon=":material/new_releases:", color="orange")
    else:
        st.badge("Jeu de données du cours", icon=":material/school:", color="gray")
    st.title(jeu["titre"], anchor=False)
    source = f"[{jeu['source']}]({jeu['lien']})" if jeu["lien"] else jeu["source"]
    st.markdown(f"{jeu['resume']}  \n:gray[Source : {source}]")

    with st.container(horizontal=True, gap="small"):
        st.metric("Observations", f"{len(m['df']):,}".replace(",", " "), border=True)
        st.metric("Variables", m["df"].shape[1], border=True)
        st.metric("Modèle retenu", m["methode"], help=m["detail"], border=True)
        st.metric("Silhouette", f"{m['silhouette']:.3f}".replace(".", ","), help="Plus il est proche de 1, plus les classes sont nettes.", border=True)


def choix_interface(cle):
    """Sélecteur entre les trois vues de la page."""
    icones = {"Simple": ":material/person:", "Multiple": ":material/table_rows:", "Modèle": ":material/hub:"}
    return st.segmented_control(
        "Interface",
        list(icones),
        default="Simple",
        required=True,
        format_func=lambda v: f"{icones[v]} {v}",
        key=f"interface_{cle}",
        label_visibility="collapsed",
    )


def _champ(cle, df, col, prefixe):
    """Champ numérique pré-rempli avec la médiane, entier ou décimal selon la colonne."""
    if pd.api.types.is_integer_dtype(df[col]):
        return st.number_input(_libelle(cle, col), min_value=0, value=int(df[col].median()), step=1, key=f"{prefixe}_{col}")
    return st.number_input(_libelle(cle, col), min_value=0.0, value=float(df[col].median()), step=0.01, format="%.4f", key=f"{prefixe}_{col}")


def interface_simple(cle, m):
    """Une observation : formulaire à gauche, verdict et position sur le plan à droite."""
    df = m["df"]
    gauche, droite = st.columns([5, 6], gap="large")
    with gauche:
        with st.form(f"simple_{cle}", border=False):
            st.subheader(JEUX[cle]["question"], anchor=False)
            st.caption("Les champs sont pré-remplis avec la valeur médiane du jeu de données.")
            grille = st.columns(2)
            valeurs = []
            for i, col in enumerate(df.columns):
                with grille[i % 2]:
                    valeurs.append(_champ(cle, df, col, f"simple_{cle}"))
            envoyer = st.form_submit_button("Classer", icon=":material/arrow_forward:", type="primary", width="stretch")

    with droite:
        if not envoyer:
            with st.container(border=True, height="stretch", vertical_alignment="center", horizontal_alignment="center"):
                st.markdown(":material/radar:", text_alignment="center")
                st.caption(f"Renseignez les caractéristiques puis cliquez sur **Classer** pour situer ce {JEUX[cle]['unite']} parmi les classes.", text_alignment="center")
            return
        if sum(valeurs) == 0:
            st.warning("Toutes les valeurs sont nulles : impossible de calculer une répartition.", icon=":material/warning:")
            return

        classes, proximites, plan = predire(cle, m, [valeurs])
        c = int(classes[0])
        nom, description = m["noms"][c]
        with st.container(border=True):
            st.caption("Classe attribuée")
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.header(nom, anchor=False, width="content")
                st.badge(f"Classe {c}", color=COULEURS_BADGE[c])
            st.markdown(description)
            st.caption("Proximité avec chaque classe (part de l'inverse de la distance au centroïde)")
            for k in np.argsort(-proximites[0]):
                part = float(proximites[0][k])
                st.progress(part, text=f"{m['noms'][k][0]} · {part:.0%}")
        nouveaux = pd.DataFrame(plan, columns=["Axe 1", "Axe 2"]).assign(Observation="Votre saisie", Classe=nom)
        st.altair_chart(_carte_acp(m, nouveaux, hauteur=300), alt=f"Position de la saisie sur le plan de l'ACP, dans la classe {nom}")
        st.caption("Plan des deux premières composantes de l'ACP. Le losange noir représente votre saisie.")


def _lire_csv(fichier, colonnes):
    """Lire un CSV importé et retrouver les colonnes attendues (casse et espaces ignorés)."""
    brut = pd.read_csv(StringIO(fichier.getvalue().decode("utf-8-sig")), sep=None, engine="python")
    cle_col = lambda s: "".join(str(s).lower().split())  # noqa: E731
    correspondance = {cle_col(c): c for c in brut.columns}
    manquantes = [c for c in colonnes if cle_col(c) not in correspondance]
    if manquantes:
        return None, manquantes
    donnees = brut[[correspondance[cle_col(c)] for c in colonnes]]
    donnees.columns = colonnes
    return donnees.apply(pd.to_numeric, errors="coerce"), []


def interface_multiple(cle, m):
    """Plusieurs observations : tableau modifiable ou fichier CSV, puis résultats téléchargeables."""
    df = m["df"]
    colonnes = list(df.columns)
    st.subheader("Classer plusieurs observations", anchor=False)

    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        fichier = st.file_uploader("Importer un fichier CSV", type="csv", key=f"csv_{cle}", width=420)
        st.download_button(
            "Fichier modèle",
            df.head(5).to_csv(index=False).encode("utf-8"),
            file_name=f"modele_{cle}.csv",
            mime="text/csv",
            icon=":material/download:",
            help="Un CSV avec les bonnes colonnes, à remplir puis importer.",
        )

    if fichier is not None:
        donnees, manquantes = _lire_csv(fichier, colonnes)
        if donnees is None:
            st.error("Colonnes absentes du fichier : " + ", ".join(f"`{c}`" for c in manquantes), icon=":material/error:")
            return
        st.caption(f"{len(donnees)} lignes lues dans **{fichier.name}**.")
    else:
        st.caption("Ou modifiez directement le tableau : une ligne par observation, ajout et suppression de lignes possibles.")
        # Deux exemples par classe, choisis parmi les observations que le modèle range bien dans cette classe
        predites = predire(cle, m, df.values)[0]
        exemple = df.groupby(predites).head(2).sort_index().reset_index(drop=True)
        config = {col: st.column_config.NumberColumn(_libelle(cle, col), min_value=0) for col in colonnes}
        donnees = st.data_editor(exemple, num_rows="dynamic", column_config=config, hide_index=True, key=f"tableau_{cle}", alt="Observations à classer")

    donnees = donnees.reset_index(drop=True)
    valides = donnees.notna().all(axis=1) & (donnees.fillna(0).sum(axis=1) > 0)
    if not st.button(f"Classer {int(valides.sum())} observations", icon=":material/arrow_forward:", type="primary", disabled=not valides.any(), key=f"lancer_{cle}"):
        return
    if (~valides).any():
        st.warning(f"{int((~valides).sum())} ligne(s) ignorée(s) : valeur manquante ou toutes les valeurs nulles.", icon=":material/warning:")

    lignes = donnees[valides]
    classes, proximites, plan = predire(cle, m, lignes.values)
    resultats = lignes.copy()
    resultats.insert(0, "Classe", [_nom(m, c) for c in classes])
    resultats.insert(1, "Proximité", proximites.max(axis=1))

    gauche, droite = st.columns([3, 2], gap="large")
    with gauche:
        config = {col: st.column_config.NumberColumn(_libelle(cle, col)) for col in colonnes}
        config["Proximité"] = st.column_config.ProgressColumn("Proximité", min_value=0, max_value=1, format="percent")
        st.dataframe(resultats, column_config=config, hide_index=True, alt="Classe attribuée à chaque observation")
        st.download_button(
            "Télécharger les résultats",
            resultats.to_csv(index=False).encode("utf-8"),
            file_name=f"resultats_{cle}.csv",
            mime="text/csv",
            icon=":material/download:",
        )
    with droite:
        repartition = resultats["Classe"].value_counts()
        with st.container(horizontal=True, gap="small"):
            for k in sorted(m["noms"]):
                st.metric(m["noms"][k][0], int(repartition.get(m["noms"][k][0], 0)), border=True)
        nouveaux = pd.DataFrame(plan, columns=["Axe 1", "Axe 2"]).assign(
            Observation=[f"Ligne {i + 1}" for i in lignes.index], Classe=resultats["Classe"].values
        )
        st.altair_chart(_carte_acp(m, nouveaux, hauteur=280), alt="Position des observations classées sur le plan de l'ACP")
        st.caption("Les losanges noirs sont vos observations.")


def vue_modele(cle, m):
    """Effectifs, profil moyen des classes et comparaison avec une variable connue."""
    df, partition = m["df"], m["partition"]
    noms = {c: m["noms"][c][0] for c in sorted(m["noms"])}
    gauche, droite = st.columns([1, 1], gap="large")
    with gauche:
        st.subheader("Les classes", anchor=False)
        effectifs = pd.Series(partition).value_counts()
        for c, nom in noms.items():
            with st.container(border=True):
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(f"**{nom}**", width="content")
                    st.badge(f"{int(effectifs.get(c, 0))} {JEUX[cle]['unite']}s", color=COULEURS_BADGE[c])
                st.caption(m["noms"][c][1])
    with droite:
        st.subheader("Plan de l'ACP", anchor=False)
        st.altair_chart(_carte_acp(m, hauteur=330), alt="Les classes sur les deux premières composantes de l'ACP")

    st.subheader("Profil moyen de chaque classe", anchor=False)
    profil = df.groupby(partition).mean()
    profil.index = [noms[c] for c in profil.index]
    profil.columns = [_libelle(cle, c) for c in profil.columns]
    st.dataframe(profil.T.style.format(precision=2, decimal=",", thousands=" "), alt="Moyenne de chaque variable par classe")

    if m["infos"] is not None:
        st.subheader(f"Classes et {m['infos'].name.lower()} réel", anchor=False)
        st.caption("Cette variable n'a pas servi au clustering : elle permet de vérifier que les classes ont un sens.")
        croise = pd.crosstab(pd.Series([noms[c] for c in partition], name="Classe"), m["infos"].rename(m["infos"].name))
        st.dataframe(croise, alt="Tableau croisé des classes et de la variable de référence")


def page(cle):
    """Page complète d'un jeu de données."""
    m = modele(cle)
    en_tete(cle, m)
    st.space("small")
    vue = choix_interface(cle)
    if vue == "Simple":
        interface_simple(cle, m)
    elif vue == "Multiple":
        interface_multiple(cle, m)
    else:
        vue_modele(cle, m)
