import streamlit as st
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import keras

# Apply runtime compatibility patch for older Keras models containing quantization configs
try:
    original_dense_init = keras.layers.Dense.__init__
    def patched_dense_init(self, *args, **kwargs):
        if 'quantization_config' in kwargs:
            del kwargs['quantization_config']
        original_dense_init(self, *args, **kwargs)
    keras.layers.Dense.__init__ = patched_dense_init
except Exception as e:
    pass

@st.cache_resource
def load_tabular_model_bundle():
    """
    Loads and caches the tabular model bundle containing:
    - pipeline
    - threshold
    - features
    - held_out_test_metrics
    """
    return joblib.load("models/ovaris_final_full_model.joblib")

@st.cache_resource
def load_image_model():
    """
    Loads and caches the CNN USG classification model
    """
    return tf.keras.models.load_model("models/model.h5")

def predict_tabular_pmos(df):
    """
    Runs prediction for the tabular features DataFrame and returns:
    - prediction: binary (0 or 1)
    - probability: float (0.0 to 1.0)
    - threshold: float
    """
    bundle = load_tabular_model_bundle()
    pipeline = bundle["pipeline"]
    threshold = bundle.get("threshold", 0.5)
    
    probability = pipeline.predict_proba(df)[0][1]
    prediction = int(probability >= threshold)
    
    return prediction, probability, threshold

def predict_image_pmos(img_array):
    """
    Runs prediction on a preprocessed USG image array and returns:
    - prediction: string label
    - confidence: float probability (0.0 to 1.0)
    - raw_probs: array of shape (1, 2)
    """
    model = load_image_model()
    raw_probs = model.predict(img_array)[0]
    
    # Class index 0: Normal / Non-PMOS
    # Class index 1: PMOS
    pmos_prob = raw_probs[1]
    prediction = "PMOS" if pmos_prob >= 0.5 else "Normal"
    
    return prediction, pmos_prob, raw_probs
