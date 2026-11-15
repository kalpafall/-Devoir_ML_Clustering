"""Page d'accueil : présentation du devoir et des quatre jeux de données."""
import streamlit as st

from segmentation import JEUX

st.caption("Devoir de machine learning · DIT")
st.title("Segmentation par clustering", anchor=False)
st.markdown(
    "Quatre jeux de données, une même démarche : normalisation, analyse en composantes principales, "
    "**Kmeans** et **classification ascendante hiérarchique**, puis comparaison des indices de silhouette. "
    "Le meilleur modèle de chaque jeu est déployé ici. Il range une nouvelle observation dans la classe "
    "dont le centre est le plus proche."
)
st.space("medium")


def carte(cle):
    jeu = JEUX[cle]
    with st.container(border=True, height="stretch"):
        if jeu["nouveau"]:
            st.badge("Nouveau jeu de données", icon=":material/new_releases:", color="orange")
        else:
            st.badge("Jeu de données du cours", icon=":material/school:", color="gray")
        st.subheader(jeu["titre"], anchor=False)
        st.markdown(jeu["resume"])
        st.caption(jeu["source"])
        st.page_link(f"app_pages/{cle}.py", label=jeu["question"], icon=":material/arrow_forward:", icon_position="right")


st.subheader("Les nouveaux jeux de données", anchor=False)
st.caption("Choisis sur le UCI Machine Learning Repository pour ce devoir.")
gauche, droite = st.columns(2)
with gauche:
    carte("grossiste")
with droite:
    carte("facebook")

st.space("small")
st.subheader("Les jeux de données du cours", anchor=False)
gauche, droite = st.columns(2)
with gauche:
    carte("clients")
with droite:
    carte("graines")

st.space("medium")
st.caption(
    "Sur chaque page : **Simple** pour classer une observation, **Multiple** pour en classer plusieurs "
    "(tableau ou fichier CSV), **Modèle** pour comprendre les classes."
)
