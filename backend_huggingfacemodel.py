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

@app.get("/")
def root():
    return {"Greeting": "Hello, World!"}
