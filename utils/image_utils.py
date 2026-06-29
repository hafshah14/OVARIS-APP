import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.cm as cm

def preprocess_image(uploaded_file, target_size=(128, 128)):
    """
    Reads an uploaded image, converts it to RGB, resizes, normalizes,
    and returns a preprocessed numpy array ready for the CNN model.
    """
    # Load image with PIL
    img = Image.open(uploaded_file).convert('RGB')
    original_img = img.copy()
    
    # Resize
    img_resized = img.resize(target_size)
    
    # Convert to array and normalize (0 to 1)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Add batch dimension
    img_tensor = np.expand_dims(img_array, axis=0)
    
    return img_tensor, original_img

def generate_gradcam(model, img_array, original_img, last_conv_layer_name=None):
    """
    Generates a Grad-CAM heatmap overlaid on the original image.
    Uses tf.GradientTape to compute gradients.
    """
    try:
        # Programmatically find the last conv layer name if not provided
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
                    last_conv_layer_name = layer.name
                    break
        
        if last_conv_layer_name is None:
            raise ValueError("No convolutional layer found in model.")
            
        # Reconstruct Functional graph to prevent graph tracing issues in Keras 3
        # Dynamically extract input shape from model inputs
        input_shape = model.input_shape[1:]  # e.g., (128, 128, 3)
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        target_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                target_output = x
        
        grad_model = tf.keras.models.Model(inputs=inputs, outputs=[target_output, x])
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # Find the top predicted class index
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
            
        # Get gradients of loss w.r.t. conv layer output
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool the gradients (mean over height and width)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply conv output by pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU to keep only positive contributions
        heatmap = tf.maximum(heatmap, 0.0)
        
        # Normalize between 0 and 1
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
            
        heatmap = heatmap.numpy()
        
        # Convert original image to numpy array
        orig_img_np = np.array(original_img)
        h, w = orig_img_np.shape[:2]
        
        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Scale heatmap to 0-255
        heatmap_scaled = np.uint8(255 * heatmap_resized)
        
        # Apply Jet colormap
        colormap = cm.get_cmap("jet")
        colormap_colors = colormap(np.arange(256))[:, :3]
        colormap_colors = np.uint8(colormap_colors * 255)
        heatmap_colored = colormap_colors[heatmap_scaled]
        
        # Superimpose the heatmap on the original image
        # Blend factor: 0.6 original, 0.4 heatmap
        superimposed = heatmap_colored * 0.4 + orig_img_np * 0.6
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        return Image.fromarray(superimposed), heatmap_resized
        
    except Exception as e:
        # Return fallback or None if Grad-CAM fails due to architecture differences
        print(f"Grad-CAM generation failed: {e}")
        return None, None
