import os
import pickle
import tensorflow as tf
from app.core.config import settings

MODEL_PATH = os.path.join(settings.MODELS_DIR, "best_model.h5")
ENCODER_PATH = os.path.join(settings.MODELS_DIR, "label_encoder.pkl")

model = tf.keras.models.load_model(MODEL_PATH)

label_encoder = None
if os.path.exists(ENCODER_PATH):
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)