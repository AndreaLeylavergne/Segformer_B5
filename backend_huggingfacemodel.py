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
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)

# Chemin vers le dossier contenant les fichiers de votre modèle
MODEL_DIR = "AndreaLeylavergne/segformer_B0_B5"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    
    # Charger le processeur d'images
    processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)
    
    # Charger le modèle sur le bon appareil
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR).to(device)
    
    logging.info("Model loaded successfully")
       
    yield

app = FastAPI(lifespan=lifespan)

class ImageID(BaseModel):
    image_id: str

@app.get("/")
def root():
    return {"Greeting": "Hello, World!"}
