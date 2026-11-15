"""Application Streamlit unique - Devoir ML Clustering (Kmeans et CAH).

Regroupe les trois modèles déployés dans une seule application à trois pages :
- Clients (télécom) : modèle CAH du notebook Customer_Segmentation.ipynb
- Grossiste : modèle Kmeans du notebook Wholesale_Customers_Clustering.ipynb
- Facebook : modèle CAH du notebook Facebook_Live_Sellers_Clustering.ipynb

Les applications app_customer_segmentation.py et app_wholesale_facebook.py
restent utilisables seules.
"""
import streamlit as st

st.set_page_config(page_title="Clustering - Kmeans et CAH", page_icon="🧩", layout="wide")

from app_wholesale_facebook import page_facebook, page_wholesale  # noqa: E402

page = st.navigation(
    {
        "Segmentation": [
            st.Page(
                "app_customer_segmentation.py",
                title="Clients (télécom)",
                icon=":material/group:",
                help="Classe d'un client d'opérateur télécom : fidèle ou peu fidèle (modèle CAH).",
                default=True,
            ),
            st.Page(
                page_wholesale,
                title="Grossiste",
                icon=":material/storefront:",
                help="Segment d'un client de grossiste selon la répartition de ses achats (modèle Kmeans).",
                url_path="grossiste",
            ),
            st.Page(
                page_facebook,
                title="Facebook",
                icon=":material/thumb_up:",
                help="Type d'engagement d'une publication Facebook de vendeur en direct (modèle CAH).",
                url_path="facebook",
            ),
        ]
    }
)

st.sidebar.caption(
    "Sources : [Customer dataset (cours ML5)](https://github.com/kalpafall/-Devoir_ML_Clustering) · "
    "[Wholesale customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers) · "
    "[Facebook Live Sellers](https://archive.ics.uci.edu/dataset/488/facebook+live+sellers+in+thailand) "
    "(UCI)"
)

page.run()
