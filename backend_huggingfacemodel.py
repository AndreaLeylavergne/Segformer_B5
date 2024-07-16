from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import numpy as np
from PIL import Image
import cv2
import base64
import io
import logging
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import rasterio
import aiohttp
import zipfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# URL du fichier modèle compressé sur Azure Blob Storage
url = "https://blobstorage224.blob.core.windows.net/segformerblob/segformer_b5_transformers.zip"

# Chemins des images et des masques annotés
images_paths = {
    "image1": "./dataset/images_prepped/val/0000FT_000294.png",
    "image2": "./dataset/images_prepped/val/0000FT_000576.png",
    "image3": "./dataset/images_prepped/val/0000FT_001016.png"
}

annotated_masks_paths = {
    "image1": "./dataset/annotations_prepped_grouped/val/0000FT_000294.png",
    "image2": "./dataset/annotations_prepped_grouped/val/0000FT_000576.png",
    "image3": "./dataset/annotations_prepped_grouped/val/0000FT_001016.png"
}

# Fonction pour lire les images
def read_img(raster_file: str) -> np.ndarray:
    print(f"Reading image from {raster_file}")
    with rasterio.open(raster_file) as src_img:
        rgb = src_img.read([1, 2, 3]).transpose(1, 2, 0)
        rgb = rgb.astype(np.float32)
        return rgb

# Fonction pour lire les masques
def read_msk(raster_file: str) -> np.ndarray:
    print(f"Reading mask from {raster_file}")
    with rasterio.open(raster_file) as src_msk:
        array = src_msk.read(1)
        array = np.squeeze(array)
        return array

# Fonction pour mapper les couleurs
def map_colors(mask):
    color_map = np.array([
        [0, 0, 0],       # Class 0 - background
        [128, 0, 0],     # Class 1 - red
        [0, 128, 0],     # Class 2 - green
        [128, 128, 0],   # Class 3 - yellow
        [0, 0, 128],     # Class 4 - blue
        [128, 0, 128],   # Class 5 - magenta
        [0, 128, 128],   # Class 6 - cyan
        [128, 128, 128], # Class 7 - white
    ])
    color_mask = color_map[mask]
    return color_mask

# Fonction pour faire une prédiction
def predict(model, processor, image_path):
    image = read_img(image_path)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.shape[:2],
            mode='bilinear',
            align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

    return pred_seg, image

# Fonction pour calculer l'IoU
def calculate_iou(array1, array2):
    assert array1.shape == array2.shape, "Arrays must have the same shape"
    array1_binary = (array1 > 0).astype(int)
    array2_binary = (array2 > 0).astype(int)
    intersection = np.sum(array1_binary * array2_binary)
    union = np.sum(array1_binary) + np.sum(array2_binary) - intersection
    return intersection / union if union > 0 else 0

# Classe pour représenter l'identifiant de l'image
class ImageID(BaseModel):
    image_id: str

# Fonction asynchrone pour télécharger et décompresser le modèle
async def download_and_extract_model(url, local_zip_path, extract_path):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception("Failed to download model file")
            with open(local_zip_path, "wb") as file:
                file.write(await response.read())
    
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Gestion du cycle de vie de l'application FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    local_zip_path = "segformer_b5_transformers.zip"
    extract_path = "./segformer_b5_transformers"

    await download_and_extract_model(url, local_zip_path, extract_path)

    # Vérifiez que le fichier preprocessor_config.json existe bien
    preprocessor_config_path = os.path.join(extract_path, "preprocessor_config.json")
    if not os.path.exists(preprocessor_config_path):
        raise FileNotFoundError(f"{preprocessor_config_path} not found")

    processor = SegformerImageProcessor.from_pretrained(extract_path)
    model = SegformerForSemanticSegmentation.from_pretrained(extract_path)
    logging.info("Model loaded successfully")

    yield

app = FastAPI(lifespan=lifespan)

# Route pour prédire le masque
@app.post("/predict/")
async def predict_mask(data: ImageID):
    image_id = data.image_id
    if image_id not in images_paths:
        raise HTTPException(status_code=404, detail="Image ID not found")
    
    image_path = images_paths[image_id]
    pred_seg, image = predict(model, processor, image_path)
    color_pred_mask = map_colors(pred_seg)
    color_pred_mask_image = Image.fromarray(color_pred_mask.astype(np.uint8))

    annotated_mask_path = annotated_masks_paths[image_id]
    annotated_mask_image = cv2.imread(annotated_mask_path, cv2.IMREAD_GRAYSCALE)
    annotated_mask_image = (annotated_mask_image / annotated_mask_image.max()) * 255
    annotated_mask_image = Image.fromarray(annotated_mask_image.astype(np.uint8)).resize((256, 128))

    color_pred_mask_stream = io.BytesIO()
    annotated_mask_stream = io.BytesIO()
        
    color_pred_mask_image.save(color_pred_mask_stream, format='PNG')
    annotated_mask_image.save(annotated_mask_stream, format='PNG')
    
    color_pred_mask_stream.seek(0)
    annotated_mask_stream.seek(0)
        
    color_pred_mask_data_url = base64.b64encode(color_pred_mask_stream.read()).decode('utf8')
    annotated_data_url = base64.b64encode(annotated_mask_stream.read()).decode('utf8')
        
    return JSONResponse(content={
        "annotated_mask": "data:image/png;base64," + annotated_data_url,
        "predicted_mask": "data:image/png;base64," + color_pred_mask_data_url
    })

# Route pour évaluer les masques
@app.post("/evaluate/")
async def evaluate_masks(data: dict):
    annotated_mask_data = data['annotated_mask']
    predicted_mask_data = data['predicted_mask']
    
    try:
        annotated_mask = Image.open(io.BytesIO(base64.b64decode(annotated_mask_data.split(',')[1]))).resize((256, 128))
        predicted_mask = Image.open(io.BytesIO(base64.b64decode(predicted_mask_data.split(',')[1]))).resize((256, 128))
        predicted_mask = predicted_mask.convert("L") # Convert "L" converts to grayscale the mask before evaluating. Important for obtaining IoU
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding images: {str(e)}")
    
    annotated_mask_array = np.array(annotated_mask)
    predicted_mask_array = np.array(predicted_mask)
    iou_score = calculate_iou(annotated_mask_array, predicted_mask_array)
    
    annotated_mask_stream = io.BytesIO()
    mask_image_stream = io.BytesIO()
    
    annotated_mask.save(annotated_mask_stream, format='PNG')
    predicted_mask.save(mask_image_stream, format='PNG')
    
    annotated_mask_stream.seek(0)
    mask_image_stream.seek(0)
    
    annotated_data_url = base64.b64encode(annotated_mask_stream.read()).decode('utf8')
    predicted_data_url = base64.b64encode(mask_image_stream.read()).decode('utf8')
    
    return JSONResponse(content={
        "iou_score": iou_score,
        "annotated_mask": "data:image/png;base64," + annotated_data_url,
        "predicted_mask": "data:image/png;base64," + predicted_data_url
    })

@app.get("/")
def root():
    return {"Greeting": "Hello, World!"}
