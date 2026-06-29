import streamlit as st
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import os
import cv2
from PIL import Image

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
    Loads and caches the CNN USG classification model exactly as in CNN.ipynb.
    Uses compile=False to avoid compilation warning metrics.
    """
    return tf.keras.models.load_model("models/model.h5", compile=False)

@st.cache_resource
def load_image_validation_params():
    """
    Loads validation center and scaling parameters stored inside the model's HDF5 metadata.
    """
    import h5py
    try:
        with h5py.File("models/model.h5", "r") as f:
            if "usg_validation" in f:
                g = f["usg_validation"]
                threshold = float(g.attrs.get("threshold", 25.1597))
                scaler_mean = g["scaler_mean"][...]
                scaler_scale = g["scaler_scale"][...]
                center_0 = g["class_centers/0"][...]
                center_1 = g["class_centers/1"][...]
                
                # Check for custom logistic regression parameters
                logistic_weights = g["logistic_weights"][...] if "logistic_weights" in g else None
                logistic_bias = float(g.attrs.get("logistic_bias", 0.0)) if "logistic_bias" in g.attrs else None
                
                return {
                    "threshold": threshold,
                    "scaler_mean": scaler_mean,
                    "scaler_scale": scaler_scale,
                    "center_0": center_0,
                    "center_1": center_1,
                    "logistic_weights": logistic_weights,
                    "logistic_bias": logistic_bias
                }
    except Exception as e:
        print(f"Warning: Failed to load USG validation parameters: {e}")
    return None

@st.cache_resource
def get_feature_extractor():
    """
    Creates a Keras model that extracts features from the 'dense_1' layer of the loaded CNN.
    Identical to Model(inputs=model.inputs, outputs=model.layers[-2].output) in CNN.ipynb.
    """
    try:
        model = load_image_model()
        dense_layer = model.get_layer("dense_1")
        return tf.keras.Model(inputs=model.inputs, outputs=dense_layer.output)
    except Exception as e:
        print(f"Warning: Failed to build feature extractor: {e}")
    return None

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

# ============================================================
# AUDITED VALIDATION FUNCTIONS DIRECTLY FROM CNN.ipynb
# ============================================================

def is_valid_image_file(img_file):
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    if isinstance(img_file, str):
        name = img_file
    else:
        name = img_file.name
        
    ext = os.path.splitext(name)[1].lower()
    if ext not in allowed_extensions:
        return False, "Format file tidak didukung. Gunakan JPG, JPEG, PNG, atau BMP."

    try:
        if isinstance(img_file, str):
            with Image.open(img_file) as img:
                img.verify()
        else:
            img_file.seek(0)
            with Image.open(img_file) as img:
                img.verify()
        return True, "File gambar valid."
    except Exception:
        return False, "File tidak dapat dibaca sebagai gambar."

def basic_image_screening(img_file):
    try:
        if isinstance(img_file, str):
            img_pil = Image.open(img_file).convert("RGB")
        else:
            img_file.seek(0)
            img_pil = Image.open(img_file).convert("RGB")
            
        img_rgb = np.array(img_pil)
        height, width, _ = img_rgb.shape

        if height < 50 or width < 50:
            return False, "Gambar terlalu kecil dan tidak layak diproses sebagai citra USG."

        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]

        mean_saturation = np.mean(saturation)
        p90_saturation = np.percentile(saturation, 90)

        r = img_rgb[:, :, 0].astype(float)
        g = img_rgb[:, :, 1].astype(float)
        b = img_rgb[:, :, 2].astype(float)

        rgb_channel_diff = np.mean(
            np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
        )

        colorful_pixel_ratio = np.sum(saturation > 80) / saturation.size

        # Tolak jika warna terlalu mencolok (foto objek umum / screenshot HP)
        too_colorful = (
            (p90_saturation > 140 and colorful_pixel_ratio > 0.12) or
            (mean_saturation > 90 and colorful_pixel_ratio > 0.18) or
            (rgb_channel_diff > 65 and colorful_pixel_ratio > 0.15)
        )

        reason = f"sat_mean={mean_saturation:.2f}, sat_p90={p90_saturation:.2f}, rgb_diff={rgb_channel_diff:.2f}, color_ratio={colorful_pixel_ratio:.4f}"
        if too_colorful:
            return False, f"Gambar terlalu berwarna sehingga tidak sesuai dengan karakter citra USG dataset ({reason})."

        return True, f"Gambar lolos validasi awal ({reason})."
    except Exception as e:
        return False, f"Error pada validasi awal: {e}"

def is_line_art_or_scribble(img_file, img_size=128):
    try:
        if isinstance(img_file, str):
            img_pil = Image.open(img_file).convert("RGB")
        else:
            img_file.seek(0)
            img_pil = Image.open(img_file).convert("RGB")
            
        img_pil = img_pil.resize((img_size, img_size))
        img_rgb = np.array(img_pil)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        dark_ratio = np.sum(gray < 40) / gray.size
        mid_intensity_ratio = np.sum((gray >= 40) & (gray <= 220)) / gray.size
        bright_ratio = np.sum(gray > 230) / gray.size

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        gray_std = np.std(gray)

        mean = cv2.blur(gray.astype(np.float32), (5, 5))
        mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (5, 5))
        local_std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
        local_std_mean = np.mean(local_std)

        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(binary > 0) / binary.size

        is_scribble = (
            dark_ratio > 0.85 and
            mid_intensity_ratio < 0.16 and
            white_ratio < 0.10 and
            bright_ratio < 0.03 and
            edge_density < 0.16 and
            local_std_mean < 20
        )

        reason = f"dark={dark_ratio:.2f}, mid={mid_intensity_ratio:.2f}, white={white_ratio:.2f}, edge_dens={edge_density:.2f}, std_mean={local_std_mean:.2f}"
        if is_scribble:
            return True, f"Gambar terdeteksi sebagai coretan/line-art, bukan citra USG ({reason})."
        else:
            return False, f"Gambar bukan coretan/line-art ({reason})."
    except Exception as e:
        return False, f"Error pada deteksi coretan: {e}"

def is_smooth_repetitive_object_photo(img_file, img_size=128):
    try:
        if isinstance(img_file, str):
            img_pil = Image.open(img_file).convert("RGB")
        else:
            img_file.seek(0)
            img_pil = Image.open(img_file).convert("RGB")
            
        img_pil = img_pil.resize((img_size, img_size))
        img_rgb = np.array(img_pil)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        dark_ratio = np.sum(gray < 40) / gray.size
        mid_intensity_ratio = np.sum((gray >= 40) & (gray <= 220)) / gray.size
        bright_ratio = np.sum(gray > 230) / gray.size

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        mean = cv2.blur(gray.astype(np.float32), (5, 5))
        mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (5, 5))
        local_std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

        local_std_mean = np.mean(local_std)
        local_std_p90 = np.percentile(local_std, 90)

        gray_blur = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(6, img_size // 16),
            param1=50,
            param2=18,
            minRadius=max(4, img_size // 50),
            maxRadius=max(8, img_size // 5)
        )

        circle_count = 0 if circles is None else circles.shape[1]

        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(binary > 0) / binary.size

        smooth_repetitive_object = (
            dark_ratio < 0.35 and
            mid_intensity_ratio > 0.65 and
            edge_density < 0.16 and
            local_std_mean < 18 and
            circle_count >= 12
        )

        smooth_photo_like = (
            dark_ratio < 0.30 and
            mid_intensity_ratio > 0.70 and
            edge_density < 0.14 and
            local_std_mean < 14
        )

        repetitive_highlight_object = (
            dark_ratio < 0.40 and
            mid_intensity_ratio > 0.55 and
            circle_count >= 20 and
            edge_density < 0.18
        )

        is_non_usg_object = (
            smooth_repetitive_object or
            smooth_photo_like or
            repetitive_highlight_object
        )

        reason = f"dark={dark_ratio:.2f}, mid={mid_intensity_ratio:.2f}, circles={circle_count}, std_mean={local_std_mean:.2f}"
        if is_non_usg_object:
            return True, f"Gambar terdeteksi sebagai foto objek non-USG, bukan citra USG ({reason})."
        else:
            return False, f"Gambar tidak terdeteksi sebagai foto objek non-USG ({reason})."
    except Exception as e:
        return False, f"Error pada deteksi objek non-USG: {e}"

def validate_usg_image_with_model(img_file, img_array):
    """
    Performs USG image validation solely using CNN deep features (no manual rule-based filters).
    Returns:
        is_valid: bool (True if all stages pass)
        debug_info: dict containing raw metrics for auditing/debugging
        message: str explaining which stage failed or a success message
    """
    val_params = load_image_validation_params()
    debug_info = {
        "is_valid_file": False,
        "is_valid_file_msg": "",
        "basic_screening": True,
        "basic_screening_msg": "Skipped (CNN-only)",
        "is_scribble": False,
        "is_scribble_msg": "Skipped (CNN-only)",
        "is_non_usg_object": False,
        "is_non_usg_object_msg": "Skipped (CNN-only)",
        "dist_0": 0.0,
        "dist_1": 0.0,
        "min_dist": 0.0,
        "threshold": 25.1597,
        "feature_valid": False,
        "feature_valid_msg": "",
        "raw_probs": [0.0, 0.0],
        "error": None
    }
    
    if val_params is None:
        debug_info["error"] = "Validation parameters not loaded"
        return False, debug_info, "Validator parameter gagal dimuat."
        
    try:
        # Step 1: File readability and extension check (essential format validation)
        v1, m1 = is_valid_image_file(img_file)
        debug_info["is_valid_file"] = v1
        debug_info["is_valid_file_msg"] = m1
        if not v1:
            return False, debug_info, m1
            
        # Step 2: CNN Deep Feature Classifier validation (entirely CNN-based OOD classification)
        feature_extractor = get_feature_extractor()
        if feature_extractor is None:
            debug_info["error"] = "Feature extractor could not be built"
            return False, debug_info, "Feature extractor gagal dibuat."
            
        features = feature_extractor.predict(img_array)[0]
        
        # Check if we have our custom logistic regression model stored in the h5 file
        if val_params.get("logistic_weights") is not None:
            weights = val_params["logistic_weights"]
            bias = val_params["logistic_bias"]
            
            # Linear model decision score: score = features * weights + bias
            score = float(np.dot(features, weights) + bias)
            is_feature_valid = score > 0
            
            debug_info["min_dist"] = score  # store decision score
            debug_info["threshold"] = 0.0
            debug_info["feature_valid"] = is_feature_valid
            debug_info["feature_valid_msg"] = f"CNN score={score:.4f}"
        else:
            # Fallback to the original Euclidean distance OOD logic
            scaled_features = (features - val_params["scaler_mean"]) / val_params["scaler_scale"]
            dist_0 = np.linalg.norm(scaled_features - val_params["center_0"])
            dist_1 = np.linalg.norm(scaled_features - val_params["center_1"])
            min_dist = min(dist_0, dist_1)
            
            debug_info["dist_0"] = float(dist_0)
            debug_info["dist_1"] = float(dist_1)
            debug_info["min_dist"] = float(min_dist)
            debug_info["threshold"] = float(val_params["threshold"])
            
            is_feature_valid = min_dist <= val_params["threshold"]
            debug_info["feature_valid"] = is_feature_valid
            debug_info["feature_valid_msg"] = f"min_dist={min_dist:.4f}, th={val_params['threshold']:.4f}"
        
        # Add raw classification probs
        model = load_image_model()
        raw_probs = model.predict(img_array)[0]
        debug_info["raw_probs"] = [float(p) for p in raw_probs]
        
        if not is_feature_valid:
            return False, debug_info, "Gambar tidak dikenali sebagai citra USG ovarium."
            
        return True, debug_info, "Citra USG ovarium berhasil dikenali."
    except Exception as e:
        debug_info["error"] = str(e)
        # Conservative fallback: reject on validation error to minimize False Positives
        return False, debug_info, f"Terjadi kesalahan saat validasi: {e}"

def predict_image_pmos(img_array):
    """
    Runs prediction on a preprocessed USG image array and returns:
    - prediction: string label ("PMOS" or "Normal")
    - confidence: float probability (0.0 to 1.0)
    - raw_probs: array of shape (1, 2)
    """
    model = load_image_model()
    raw_probs = model.predict(img_array)[0]
    
    # Class index 0: Normal / Non-PMOS
    # Class index 1: PMOS
    pmos_prob = float(raw_probs[1])
    prediction = "PMOS" if pmos_prob >= 0.5 else "Normal"
    
    return prediction, pmos_prob, raw_probs
