import streamlit as st
import requests
from PIL import Image
import io
import base64 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.title("Segmentation sémantique avec Segformer (PoC)")

st.write("### Nous avons avons comparée les performances de VGG-Unet et de Segformer pour la segmentation sémantique des images.")
st.write("### Voici un aperçu des résultats et un espace pour tester vous même les la performance des prédictions.")

# Charger le fichier CSV
data_file = "./results_POC.csv"  

# Lire le fichier CSV
df = pd.read_csv(data_file)

# Afficher le dataframe
st.write("Aperçu des données:", df)

# Préparation des données
models = df['Epoch 4']
epochs = ['Epoch 4'] * len(models)

# Mean Accuracy
fig_mean_accuracy = go.Figure()
fig_mean_accuracy.add_trace(go.Scatter(x=models, y=df['Mean Accuracy'], mode='lines+markers', name='Mean Accuracy'))
fig_mean_accuracy.update_layout(title='Accuracy moyenne momparée')

# Mean Loss
fig_mean_loss = go.Figure()
fig_mean_loss.add_trace(go.Scatter(x=models, y=df['Training Loss'], mode='lines+markers', name='Training Loss'))
fig_mean_loss.add_trace(go.Scatter(x=models, y=df['Validation Loss'], mode='lines+markers', name='Validation Loss'))
fig_mean_loss.update_layout(title='Perte moyenne comparée')

# Mean IoU
fig_mean_iou = go.Figure()
fig_mean_iou.add_trace(go.Scatter(x=models, y=df['Mean Iou'], mode='lines+markers', name='Mean IoU'))
fig_mean_iou.update_layout(title='Score IoU moyen comparé')

# Afficher les graphiques
st.plotly_chart(fig_mean_accuracy, use_container_width=True)
st.plotly_chart(fig_mean_loss, use_container_width=True)
st.plotly_chart(fig_mean_iou, use_container_width=True)

# Score IoU par catégorie
categories = ['Iou Void', 'Iou Flat', 'Iou Construction', 'Iou Object', 'Iou Nature', 'Iou Sky', 'Iou Human', 'Iou Vehicle']
fig_iou_per_category = go.Figure()

for category in categories:
    fig_iou_per_category.add_trace(go.Bar(x=models, y=df[category], name=category))

fig_iou_per_category.update_layout(barmode='group', title='Scores IoU par Catégorie')

# Afficher le graphique
st.plotly_chart(fig_iou_per_category, use_container_width=True)

# Indiquer que Segformer est le plus performant
st.subheader("Segformer est le plus performant!")
st.markdown("🥇 **Segformer_B5** a gagné avec les meilleures performances sur la plupart des catégories. Maintenant, nous pouvons faire des inférences sur des images.")

# Paths of images and annotated masks
images_paths = {
    "image1": "./dataset/images_prepped/val/0000FT_000294.png",
    "image2": "./dataset/images_prepped/val/0000FT_000576.png",
    "image3": "./dataset/images_prepped/val/0000FT_001016.png"
}

# Affichage des images en ligne
st.write("### Sélectionnez une image pour prédire les masques avec Segformer")
cols = st.columns(len(images_paths)) #st.column snous aide à afficher le simages en ligne pour que l'utilisateur ait une vue d'ensemble rapide. 
for idx, (label, path) in enumerate(images_paths.items()):
    with cols[idx]:
        image = Image.open(path)
        st.image(image, caption=label, use_column_width=True)

# Sélection d'image pour prédiction
image_id = st.selectbox("Choisissez une image pour l'inférence", list(images_paths.keys())) #selectbox permet à l'utilisateur de selectionner une image

"""# Menu to select an image
image_id = st.selectbox(
    "Choisissez une image",
    ("image1", "image2", "image3")
)
"""
if st.button('Prédire'):
    response = requests.post("https://segformer.azurewebsites.net/predict/", json={"image_id": image_id})
    if response.status_code == 200:
        result = response.json()
        annotated_mask = Image.open(io.BytesIO(base64.b64decode(result["annotated_mask"].split(",")[1])))
        predicted_mask = Image.open(io.BytesIO(base64.b64decode(result["predicted_mask"].split(",")[1])))

        st.image(annotated_mask, caption='Masque Annoté', use_column_width=True)
        st.image(predicted_mask, caption='Masque Prédit', use_column_width=True)

        st.session_state.annotated_mask = result["annotated_mask"]
        st.session_state.predicted_mask = result["predicted_mask"]
    else:
        st.write("Erreur lors de la prédiction")

if st.button('Évaluer'):
    if "annotated_mask" in st.session_state and "predicted_mask" in st.session_state:
        response = requests.post("https://segformer.azurewebsites.net/evaluate/", json={
            "annotated_mask": st.session_state.annotated_mask,
            "predicted_mask": st.session_state.predicted_mask
        })

        if response.status_code == 200:
            result = response.json()
            iou_score = result["iou_score"]
            annotated_mask = Image.open(io.BytesIO(base64.b64decode(result["annotated_mask"].split(",")[1])))
            predicted_mask = Image.open(io.BytesIO(base64.b64decode(result["predicted_mask"].split(",")[1])))

            st.image(annotated_mask, caption='Masque Annoté', use_column_width=True)
            st.image(predicted_mask, caption='Masque Prédit', use_column_width=True)
            st.write(f"Score IoU: {iou_score}")
        else:
            st.write("Erreur lors de l'évaluation")
    else:
        st.write("Veuillez d'abord prédire les masques.")
