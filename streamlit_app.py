"""Application Streamlit unique - Devoir ML Clustering (Kmeans et CAH).

Point d'entrée : navigation entre l'accueil et les quatre jeux de données.
Les modèles et l'interface commune sont dans segmentation.py, les pages dans app_pages/.
"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Segmentation par clustering", page_icon=":material/scatter_plot:", layout="wide")
st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")

page = st.navigation(
    {
        "Présentation": [
            st.Page("app_pages/accueil.py", title="Accueil", icon=":material/home:", default=True,
                    help="Le devoir, la démarche et les quatre jeux de données."),
        ],
        "Nouveaux jeux de données": [
            st.Page("app_pages/grossiste.py", title="Clients d'un grossiste", icon=":material/storefront:",
                    help="UCI 292. Profil d'acheteur selon la répartition des dépenses (Kmeans)."),
            st.Page("app_pages/facebook.py", title="Publications Facebook", icon=":material/forum:",
                    help="UCI 488. Type d'engagement d'une publication de vente en direct (CAH)."),
        ],
        "Jeux de données du cours": [
            st.Page("app_pages/clients.py", title="Clients télécom", icon=":material/call:",
                    help="Fidélité d'un abonné selon son usage du service (CAH)."),
            st.Page("app_pages/graines.py", title="Graines de blé", icon=":material/grass:",
                    help="Taille d'une graine selon ses mesures (CAH)."),
        ],
    }
)

with st.sidebar:
    st.caption("Kmeans et CAH · meilleur modèle déployé pour chaque jeu de données.")

page.run()
