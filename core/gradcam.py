from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from core.preprocessing import preprocess_visual_image
from core.predictions import get_model, preprocess_image

# Base directories
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "plots" / "GradCAM"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output file used by the GUI
OUTPUT_FILE = OUTPUT_DIR / "latest_gradcam.png"


def find_last_conv_layer(model):
    """
    Dynamically locate the last 4D output convolutional or concatenation layer.
    """
    for layer in reversed(model.layers):
        try:
            output_shape = layer.output_shape
            if isinstance(output_shape, list):
                output_shape = output_shape[0]
            if len(output_shape) == 4:
                return layer.name
        except Exception:
            continue

    return "conv5_block16_concat"


# -----------------------------------------------------------------------------
# Grad-CAM++ Heatmap Generation
# -----------------------------------------------------------------------------
def make_gradcam_plus_plus_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generate a Grad-CAM++ heatmap using 1st, 2nd, and 3rd order gradients.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    try:
        conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        last_conv_layer_name = find_last_conv_layer(model)
        conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape3:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                conv_outputs, predictions = grad_model(img_array)
                if pred_index is None:
                    pred_index = 0
                class_channel = predictions[:, pred_index]

            grads_1 = tape1.gradient(class_channel, conv_outputs)
        grads_2 = tape2.gradient(grads_1, conv_outputs)
    grads_3 = tape3.gradient(grads_2, conv_outputs)

    if grads_1 is None:
        # Fallback to standard Grad-CAM if higher-order gradients are unsupported
        return make_standard_gradcam(img_array, model, last_conv_layer_name, pred_index)

    conv_outputs = conv_outputs[0]
    grads_1 = grads_1[0]
    grads_2 = grads_2[0]
    grads_3 = grads_3[0]

    # Compute alpha coefficients for Grad-CAM++
    sum_activation_grads = tf.reduce_sum(conv_outputs * grads_3, axis=(0, 1), keepdims=True)
    denom = 2.0 * grads_2 + sum_activation_grads
    denom = tf.where(denom != 0.0, denom, tf.ones_like(denom) * 1e-10)

    aij = grads_2 / denom

    positive_grads_1 = tf.maximum(grads_1, 0.0)
    weights = tf.reduce_sum(positive_grads_1 * aij, axis=(0, 1))

    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0.0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def make_standard_gradcam(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = 0
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)

    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


# Alias for backward compatibility
make_gradcam_heatmap = make_gradcam_plus_plus_heatmap


# -----------------------------------------------------------------------------
# Apply heatmap to original image
# -----------------------------------------------------------------------------
def overlay_heatmap(original_image_path, heatmap, alpha=0.35, output_file=None):
    from matplotlib import cm

    original_np = preprocess_visual_image(original_image_path).astype(np.float32)

    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(
        (original_np.shape[1], original_np.shape[0])
    )

    heatmap_np = np.array(heatmap_img).astype(np.float32) / 255.0

    colored_heatmap = cm.get_cmap("inferno")(heatmap_np)[..., :3]
    colored_heatmap = (colored_heatmap * 255).astype(np.float32)

    blended = original_np * (1 - alpha) + colored_heatmap * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    result = Image.fromarray(blended)
    target_out = output_file if output_file is not None else OUTPUT_FILE
    result.save(target_out)

    return target_out


def save_gradcam(model_type, img_path, output_path):
    """
    Generate and save Grad-CAM++ visualization.
    """
    model = get_model(model_type)
    img_array = preprocess_image(img_path)
    last_conv_layer_name = find_last_conv_layer(model)

    heatmap = make_gradcam_plus_plus_heatmap(
        img_array,
        model,
        last_conv_layer_name
    )

    original_np = preprocess_visual_image(img_path).astype(np.float32)

    heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(
        (original_np.shape[1], original_np.shape[0])
    )

    heatmap_np = np.array(heatmap_img).astype(np.float32) / 255.0

    from matplotlib import cm
    colored_heatmap = cm.get_cmap("inferno")(heatmap_np)[..., :3]
    colored_heatmap = (colored_heatmap * 255).astype(np.float32)

    blended = original_np * 0.6 + colored_heatmap * 0.4
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    result = Image.fromarray(blended)
    result.save(output_path)

    return str(output_path)


def generate_gradcam(image_path, model_type, output_path=None):
    """
    Generate Grad-CAM++ visualization for an image.
    """
    model = get_model(model_type)
    img_array = preprocess_image(image_path)
    last_conv_layer_name = find_last_conv_layer(model)

    heatmap = make_gradcam_plus_plus_heatmap(
        img_array,
        model,
        last_conv_layer_name,
    )

    if output_path is None:
        img_stem = Path(image_path).stem
        output_path = OUTPUT_DIR / f"gradcam_{img_stem}.png"

    output_path = overlay_heatmap(image_path, heatmap, output_file=output_path)

    return str(output_path)


if __name__ == "__main__":  # pragma: no cover
    test_image = input("Enter image path: ").strip()
    model_name = input("Enter model type (Parts/Elbow/Hand/Shoulder): ").strip()

    output = generate_gradcam(test_image, model_name)
    print(f"Grad-CAM++ saved to: {output}")