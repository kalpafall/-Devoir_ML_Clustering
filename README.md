# Devoir ML — Clustering (Kmeans et CAH)

Projet local, à ouvrir dans VS Code. Tous les fichiers sont à la racine.

## Contenu

| Livrable | Fichier(s) | Données |
|---|---|---|
| 1. Application Streamlit customer segmentation | `app_customer_segmentation.py` (+ notebook `Customer_Segmentation.ipynb`) | `Customer_dataset.csv` |
| 2. Wheat Seeds : Kmeans + CAH + déploiement Gradio | `Kmeans_On_Wheat_Seeds_Dataset.ipynb` | `wheat_seeds_dataset.csv` |
| 3. Nouveau dataset 1 : Wholesale customers | `Wholesale_Customers_Clustering.ipynb` | `Wholesale customers data.csv` |
| 4. Nouveau dataset 2 : Facebook Live Sellers | `Facebook_Live_Sellers_Clustering.ipynb` | `Live_20210128.csv` |
| 5. Application Streamlit des deux nouveaux datasets | `app_wholesale_facebook.py` | les deux CSV ci-dessus |

Explications détaillées : `EXPLICATIONS.md`.

## Installation (une seule fois)

```bash
cd ~/Documents/Devoir_ML_Clustering
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-notebooks.txt
```

Dans VS Code, ouvrir un notebook puis choisir le noyau **venv** en haut à droite
(« Select Kernel » → « Python Environments » → `venv`).

## Lancer les applications en local

```bash
source venv/bin/activate
streamlit run streamlit_app.py                  # application unique : les trois jeux de données
streamlit run app_customer_segmentation.py      # livrable 1 seul
streamlit run app_wholesale_facebook.py         # livrable 5 seul
```

`streamlit_app.py` regroupe les trois modèles déployés dans une seule application à trois pages
(Clients télécom, Grossiste, Facebook). Les deux autres fichiers restent utilisables séparément.

L'application Gradio de Wheat Seeds se lance depuis la dernière cellule du notebook
(`interface.launch(share=True)` donne un lien public temporaire).

## Mettre en ligne sur Streamlit Community Cloud

Sur https://share.streamlit.io : « Create app », dépôt `kalpafall/-Devoir_ML_Clustering`, branche `main`,
fichier principal `streamlit_app.py`, Python 3.12 dans « Advanced settings » → « Deploy ».
Ce lien unique sert pour les livrables 1 et 5.

Pour deux liens séparés, créer plutôt deux applications avec les fichiers principaux
`app_customer_segmentation.py` (livrable 1) et `app_wholesale_facebook.py` (livrable 5).

Streamlit Cloud installe uniquement `requirements.txt` (les bibliothèques des notebooks n'y sont pas nécessaires).
