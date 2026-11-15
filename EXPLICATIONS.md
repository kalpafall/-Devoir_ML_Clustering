# Devoir ML - Clustering : explications

## Les livrables

| # | Livrable | Fichier |
|---|---|---|
| 1 | Application Streamlit `customer_segmentation` : https://devoir-ml-clustering.streamlit.app | `streamlit_app.py` (page « Clients télécom ») |
| 2 | Notebook Wheat Seeds : Kmeans + CAH + déploiement Gradio | `Kmeans_On_Wheat_Seeds_Dataset.ipynb` |
| 3 | Notebook Wholesale customers | `Wholesale_Customers_Clustering.ipynb` |
| 4 | Notebook Facebook Live Sellers | `Facebook_Live_Sellers_Clustering.ipynb` |
| 5 | Application Streamlit commune aux deux nouveaux datasets : https://devoir-ml-clustering.streamlit.app | `streamlit_app.py` (pages « Clients d'un grossiste » et « Publications Facebook ») |

## Les deux nouveaux jeux de données

### Wholesale customers
- **Lien :** https://archive.ics.uci.edu/dataset/292/wholesale+customers
- **Contenu :** 440 clients d'un grossiste portugais. Pour chacun : dépenses annuelles en produits frais,
  produits laitiers, épicerie, surgelés, détergents et papier, épicerie fine ; plus le canal de vente
  (Horeca ou commerce de détail) et la région.
- **Intérêt :** comme `customer_segmentation`, c'est une segmentation de clients, ici selon leurs habitudes
  d'achat.

### Facebook Live Sellers in Thailand
- **Lien :** https://archive.ics.uci.edu/dataset/488/facebook+live+sellers+in+thailand
- **Contenu :** 7 050 publications Facebook de 10 vendeurs thaïlandais de mode et cosmétiques (2012-2018).
  Pour chacune : type (photo, vidéo, statut, lien), date, et indicateurs d'engagement (commentaires,
  partages, j'aime, j'adore, wouah, haha, triste, grrr).
- **Intérêt :** un domaine différent (réseaux sociaux) : on regroupe les publications selon le type
  d'engagement qu'elles suscitent.

## Notebook Wholesale customers

1. **Nettoyage :** pas de valeur manquante ni de doublon. `Channel` et `Region` sont des codes, pas des
   quantités : on les retire du clustering et on les garde pour interpréter les classes.
2. **Préparation :** normalisation de chaque client (comme dans `customer_segmentation`), pour comparer la
   répartition des achats et non leur montant total.
3. **ACP :** 2 composantes, environ 83 % de l'information.
4. **Kmeans :** la méthode du coude donne k = 2. Silhouette : 0,500.
5. **CAH :** le lien moyen isole un seul client atypique, donc on utilise le critère de Ward. Le
   dendrogramme confirme 2 classes. Silhouette : 0,491.
6. **Meilleur modèle : Kmeans.** Les deux méthodes trouvent presque les mêmes groupes.
7. **Interprétation :** une classe achète surtout des produits frais (218 clients Horeca sur 250), l'autre
   surtout de l'épicerie, du lait et des détergents (majorité des commerces de détail). Le clustering
   retrouve le type de client sans l'avoir utilisé.

## Notebook Facebook Live Sellers

1. **Nettoyage :** suppression de 4 colonnes entièrement vides, de l'identifiant `status_id`, et de 116
   publications sans aucun engagement. Le type et la date sont gardés pour l'interprétation.
2. **Préparation :** normalisation de chaque publication, pour comparer la nature de l'engagement et non la
   popularité brute.
3. **ACP :** 2 composantes, près de 97 % de l'information.
4. **Kmeans :** la méthode du coude donne k = 3. Silhouette : 0,742.
5. **CAH :** lien moyen, dendrogramme tronqué (7 000 publications), 3 classes. Silhouette : 0,816.
6. **Meilleur modèle : CAH.** Une nouvelle publication est classée grâce au centroïde le plus proche
   (99,4 % d'accord avec la CAH).
7. **Interprétation :**
   - 1 172 publications **très commentées** : presque toutes des vidéos (les directs, où l'on commente pour
     commander) ;
   - 5 742 publications **surtout aimées** : presque toutes les photos, statuts et liens ;
   - 20 publications **surtout partagées**.

## L'application en ligne

Lien : https://devoir-ml-clustering.streamlit.app

Une seule application Streamlit (`streamlit_app.py`) regroupe les quatre jeux de données. Le menu de gauche
les range en deux groupes :
- **Nouveaux jeux de données** (UCI) : *Clients d'un grossiste* (Kmeans) et *Publications Facebook* (CAH),
  le livrable 5 ;
- **Jeux de données du cours** : *Clients télécom* (CAH, le livrable 1) et *Graines de blé* (CAH).

Chaque page affiche l'origine du jeu de données, ses chiffres clés (observations, variables, modèle retenu,
silhouette) et trois vues :
- **Simple** : un formulaire pour une observation. L'application donne sa classe, sa proximité avec
  chaque classe et sa position sur le plan de l'ACP.
- **Multiple** : un tableau modifiable ou un fichier CSV importé pour classer plusieurs observations d'un
  coup. Les résultats se téléchargent en CSV, et un fichier modèle est fourni.
- **Modèle** : description et effectif des classes, nuage de points de l'ACP, profil moyen de chaque
  classe, et comparaison avec une variable connue quand elle existe (canal de vente, type de publication).

Comme dans les notebooks, une nouvelle observation est rangée dans la classe dont le centroïde est le
plus proche.

## Liens à rendre

- Application Streamlit customer segmentation (livrable 1) : https://devoir-ml-clustering.streamlit.app
- Application Streamlit Wholesale customers et Facebook Live Sellers (livrable 5) : https://devoir-ml-clustering.streamlit.app
- Application Gradio Wheat Seeds : à compléter (lien donné par la dernière cellule du notebook)
