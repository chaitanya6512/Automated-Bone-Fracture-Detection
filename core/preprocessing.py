from PIL import Image
import cv2
import numpy as np
import tensorflow as tf

def read_image_file(img_path):
    """
    Read standard image files (PNG, JPG, JPEG) or DICOM (.dcm, .dicom) files.
    Returns RGB uint8 numpy array.
    """
    path_str = str(img_path).lower()
    if path_str.endswith(".dcm") or path_str.endswith(".dicom"):
        try:
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_voi_lut

            dicom = pydicom.dcmread(img_path)
            try:
                data = apply_voi_lut(dicom.pixel_array, dicom)
            except Exception:
                data = dicom.pixel_array

            if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
                data = np.max(data) - data

            data = data.astype(np.float32)
            if np.max(data) != np.min(data):
                data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255.0
            data = data.astype(np.uint8)

            if len(data.shape) == 2:
                img_np = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
            elif len(data.shape) == 3:
                if data.shape[2] == 1:
                    img_np = cv2.cvtColor(data[:, :, 0], cv2.COLOR_GRAY2RGB)
                else:
                    img_np = data
            return img_np
        except Exception as e:
            print(f"[WARNING] Failed to load DICOM file with pydicom: {e}. Falling back to PIL.")

    img = Image.open(img_path).convert("RGB")
    return np.array(img)


def crop_roi(img):

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Threshold image
    _, thresh = cv2.threshold(
        gray,
        15,
        255,
        cv2.THRESH_BINARY
    )

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return img

    # Largest contour
    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    # Add padding
    pad = 40

    x = max(x - pad, 0)
    y = max(y - pad, 0)

    w = min(w + 2 * pad, img.shape[1] - x)
    h = min(h + 2 * pad, img.shape[0] - y)

    cropped = img[y:y+h, x:x+w]

    return cropped

def clahe_preprocessing(img):

    img = np.clip(img, 0, 255).astype(np.uint8)
    img = crop_roi(img)
    img = cv2.resize(img, (320, 320))

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    enhanced = tf.keras.applications.densenet.preprocess_input(
        enhanced.astype(np.float32)
    )

    return enhanced

def preprocess_visual_image(img_path):

    img = read_image_file(img_path)
    img = crop_roi(img)

    return img