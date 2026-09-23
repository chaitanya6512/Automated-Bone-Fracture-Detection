from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import cv2
from tensorflow.keras.optimizers import Adam
from core.preprocessing import clahe_preprocessing


# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
WEIGHTS_DIR = BASE_DIR / "weights"
PLOTS_DIR = BASE_DIR / "plots"

LABELS = ["Elbow", "Hand", "Shoulder"]
IMAGE_SIZE = (320, 320)
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 3e-5
RANDOM_STATE = 42

# Create output folders if they do not exist
WEIGHTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Load Dataset
def load_dataset(dataset_path: Path):
    """
    Load the MURA dataset and prepare a list of image paths and labels.

    Expected structure:
        Dataset/
            train/
            valid/

    Returns:
        List[dict]
    """
    dataset = []

    for split_folder in dataset_path.iterdir():
        if not split_folder.is_dir():
            continue

        for body_part_folder in split_folder.iterdir():
            if not body_part_folder.is_dir():
                continue

            body_part = body_part_folder.name

            for patient_folder in body_part_folder.iterdir():
                if not patient_folder.is_dir():
                    continue

                for study_folder in patient_folder.iterdir():
                    if not study_folder.is_dir():
                        continue

                    study_name = study_folder.name

                    for image_file in study_folder.iterdir():
                        if image_file.is_file():
                            dataset.append({
                                "label": body_part,
                                "patient_id": patient_folder.name,
                                "image_path": str(image_file)
                            })

    return dataset


# Prepare DataFrame
def prepare_dataframe(dataset):
    """
    Convert dataset list to pandas DataFrame.
    """
    filepaths = [row["image_path"] for row in dataset]
    labels = [row["label"] for row in dataset]
    patient_ids = [row["patient_id"] for row in dataset]

    return pd.DataFrame({
        "Filepath": filepaths,
        "Label": labels,
        "patient_id": patient_ids
    })



# Create Data Generators
def create_generators(train_df, test_df):
    """
    Create training, validation, and test generators.
    """
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=clahe_preprocessing,
        validation_split=0.2,

        rotation_range=10,
        zoom_range=0.08,

        width_shift_range=0.05,
        height_shift_range=0.05,

        horizontal_flip=True,

        fill_mode="nearest"
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=clahe_preprocessing
    )

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col="Filepath",
        y_col="Label",
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_STATE,
        subset="training"
    )

    # ------------------------------------------------------------------
    # Compute class weights to handle class imbalance
    # ------------------------------------------------------------------
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_images.classes),
        y=train_images.classes
    )

    class_weights = dict(enumerate(class_weights))

    print("Class Weights:", class_weights)

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col="Filepath",
        y_col="Label",
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=RANDOM_STATE,
        subset="validation"
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col="Filepath",
        y_col="Label",
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_images, val_images, test_images, class_weights


# Build Model
def build_model():
    """
    Build the DenseNet transfer learning model.
    """
    base_model = tf.keras.applications.DenseNet121(
        input_shape=(320, 320, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )

    # Fine-tuning setup
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    for layer in base_model.layers[-50:]:
        layer.trainable = True

    inputs = base_model.input
    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(base_model.output)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.20)(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = tf.keras.layers.Dropout(0.15)(x)
    outputs = tf.keras.layers.Dense(len(LABELS), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model


# Save Training Plots
def save_plots(history):
    """
    Save accuracy and loss plots.
    """
    # Accuracy plot
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Body Part Classification Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"])
    plt.savefig(PLOTS_DIR / "body_parts_accuracy.png")
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Body Part Classification Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"])
    plt.savefig(PLOTS_DIR / "body_parts_loss.png")
    plt.close()

    # create recall plot and save it
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    figRecall = plt.gcf()
    figRecall.savefig(PLOTS_DIR / "Recall.jpeg")
    plt.clf()

    # create AUC plot and save it
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    figAUC = plt.gcf()
    figAUC.savefig(PLOTS_DIR / "AUC.jpeg")
    plt.clf()

# Main Training Pipeline
def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_DIR)

    if not dataset:
        raise ValueError(
            f"No images found in dataset folder: {DATASET_DIR}"
        )

    print(f"Total images found: {len(dataset)}")

    print("Preparing dataframe...")
    images_df = prepare_dataframe(dataset)

    print("Splitting dataset by patient level (zero data leakage)...")
    unique_patients = images_df["patient_id"].unique()
    train_patients, test_patients = train_test_split(
        unique_patients,
        train_size=0.9,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    train_df = images_df[images_df["patient_id"].isin(train_patients)]
    test_df = images_df[images_df["patient_id"].isin(test_patients)]


    print("Creating data generators...")
    train_images, val_images, test_images, class_weights = create_generators(
        train_df,
        test_df
    )

    print("Building model...")
    model = build_model()
    model.summary()

    print("Starting training...")
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=8,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        mode='max',
        factor=0.5,
        patience=3,
        verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=WEIGHTS_DIR / "DenseNet121_BodyParts_best.keras",
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    model = tf.keras.models.load_model(
        WEIGHTS_DIR / "DenseNet121_BodyParts_best.keras"
    )

    print("Evaluating model...")
    results = model.evaluate(test_images, verbose=0)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1] * 100:.2f}%")
    print(f"Test Recall: {results[2] * 100:.2f}%")
    print(f"Test Precision: {results[3] * 100:.2f}%")
    print(f"Test AUC: {results[4] * 100:.2f}%")

    print("Saving model...")
    model = tf.keras.models.load_model(
        WEIGHTS_DIR / "DenseNet121_BodyParts_best.keras"
    )


    print("Saving plots...")
    save_plots(history)

    print("Training completed successfully.")


# Script Entry Point
if __name__ == "__main__":
    main()