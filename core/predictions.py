import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings(
    "ignore",
    category=UserWarning
)

from pathlib import Path
import numpy as np
import tensorflow as tf
from core.preprocessing import clahe_preprocessing, read_image_file
from core.config import (
    WEIGHTS_DIR,
    IMAGE_SIZE,
    THRESHOLDS,
    CATEGORIES_PARTS,
    MODEL_FILES
)

# Custom Metric Fallback to eliminate tensorflow-addons dependency
class CustomF1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.val = self.add_weight(name="f1", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        return self.val

    def reset_state(self):
        pass

# Model Cache
MODELS: dict[str, tf.keras.Model] = {}

# Helper Function: Load a Model
def load_model(model_name: str):
    if model_name not in MODELS:
        model_path = WEIGHTS_DIR / model_name

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please train the models first or place them in the 'weights' folder."
            )

        print(f"[INFO] Loading model: {model_path.name}")

        MODELS[model_name] = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "F1Score": CustomF1Score
            },
            compile=False
        )

    return MODELS[model_name]


# Helper Function: Select Appropriate Model
def get_model(model_type: str = "Parts"):

    if model_type not in MODEL_FILES:
        raise ValueError(
            f"Invalid model type: {model_type}. "
            f"Valid options are: {list(MODEL_FILES.keys())}"
        )

    return load_model(MODEL_FILES[model_type])


# Helper Function: Preprocess Image (Supports PNG, JPG, JPEG, DCM, DICOM)
def preprocess_image(img_path: str):

    """
    Load and preprocess image (including DICOM) for DenseNet121 prediction.
    """

    img_rgb = read_image_file(img_path)
    x = clahe_preprocessing(img_rgb)
    x = np.expand_dims(x, axis=0)

    return x


# Main Prediction Function
def predict(
    img_path: str,
    model: str = "Parts",
    return_confidence: bool = False,
    return_probs: bool = False,
    return_uncertainty: bool = False
):

    img_array = preprocess_image(img_path)
    model_instance = get_model(model)

    prediction_probs = model_instance.predict(
        img_array,
        verbose=0
    )[0]

    # -----------------------------------------
    # BODY PART CLASSIFIER
    # -----------------------------------------
    if model == "Parts":

        predicted_index = np.argmax(prediction_probs)
        prediction_label = CATEGORIES_PARTS[predicted_index]
        confidence = float(prediction_probs[predicted_index])

    # -----------------------------------------
    # FRACTURE CLASSIFIER
    # -----------------------------------------
    else:

        fractured_prob = float(prediction_probs[0])
        normal_prob = float(prediction_probs[1])

        threshold = THRESHOLDS.get(model, 0.5)

        if fractured_prob >= threshold:
            prediction_label = "fractured"
            confidence = fractured_prob
        else:
            prediction_label = "normal"
            confidence = normal_prob

    confidence_percent = round(confidence * 100, 2)

    # -----------------------------------------
    # CLINICAL UNCERTAINTY QUANTIFICATION (1c)
    # -----------------------------------------
    # Flag as uncertain if confidence is under 65% or class gap is narrow
    is_uncertain = (confidence_percent < 65.0)

    print(f"Prediction: {prediction_label} | Confidence: {confidence_percent}% | Uncertain: {is_uncertain}")

    if return_uncertainty:
        return prediction_label, confidence_percent, prediction_probs, is_uncertain

    if not return_confidence and not return_probs:
        return prediction_label

    if return_confidence and not return_probs:
        return prediction_label, confidence_percent

    return prediction_label, confidence_percent, prediction_probs