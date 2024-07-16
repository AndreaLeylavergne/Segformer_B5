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

app = FastAPI() #delete when charging the model 

# Paths of images and annotated masks
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

# Utility functions
def read_img(raster_file: str) -> np.ndarray:
    print(f"Reading image from {raster_file}")
    with rasterio.open(raster_file) as src_img:
        rgb = src_img.read([1, 2, 3]).transpose(1, 2, 0)
        rgb = rgb.astype(np.float32)
        return rgb

def read_msk(raster_file: str) -> np.ndarray:
    print(f"Reading mask from {raster_file}")
    with rasterio.open(raster_file) as src_msk:
        array = src_msk.read(1)
        array = np.squeeze(array)
        return array

def map_colors(mask):
    # Create a color map with high contrast colors
    color_map = np.array([
        [0, 0, 0],       # Class 0 - background
        [128, 0, 0],     # Class 1 - red
        [0, 128, 0],     # Class 2 - green
        [128, 128, 0],   # Class 3 - yellow
        [0, 0, 128],     # Class 4 - blue
        [128, 0, 128],   # Class 5 - magenta
        [0, 128, 128],   # Class 6 - cyan
        [128, 128, 128], # Class 7 - white
        # Add more colors if you have more classes
    ])
    
    # Apply the color map to the mask
    color_mask = color_map[mask]
    
    return color_mask

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

def calculate_iou(array1, array2):
    assert array1.shape == array2.shape, "Arrays must have the same shape"

    array1_binary = (array1 > 0).astype(int)
    array2_binary = (array2 > 0).astype(int)

    intersection = np.sum(array1_binary * array2_binary)
    union = np.sum(array1_binary) + np.sum(array2_binary) - intersection

    return intersection / union if union > 0 else 0

@app.get("/")
def root():
    return {"Greeting": "Hello, World!"}
