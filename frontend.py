import streamlit as st
import requests
from PIL import Image
import io
import base64

st.title("Segmentation d'images")

# Menu to select an image
image_id = st.selectbox(
    "Choisissez une image",
    ("image1", "image2", "image3")
)

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
