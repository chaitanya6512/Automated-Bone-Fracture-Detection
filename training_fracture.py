import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import cv2
from tensorflow.keras.optimizers import Adam
from training_parts import LEARNING_RATE
from core.preprocessing import clahe_preprocessing


class CategoricalFocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss to improve recall on challenging micro-fractures (1b).
    """
    def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        focal_weight = tf.pow(1.0 - y_pred, self.gamma)
        loss = -self.alpha * y_true * focal_weight * tf.math.log(y_pred)
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


def load_path(dataset_path: Path, part: str):
    """
    Load X-ray dataset using Pathlib
    """
    dataset = []
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        return dataset

    for folder in dataset_path.iterdir():
        if not folder.is_dir():
            continue
        for body in folder.iterdir():
            if not body.is_dir() or body.name != part:
                continue
            body_part = body.name
            for patient_dir in body.iterdir():
                if not patient_dir.is_dir():
                    continue
                patient_id = patient_dir.name
                for study_dir in patient_dir.iterdir():
                    if not study_dir.is_dir():
                        continue
                    study_name = study_dir.name.lower()
                    if study_name.endswith('positive'):
                        label = 'fractured'
                    elif study_name.endswith('negative'):
                        label = 'normal'
                    else:
                        continue
                    for img_file in study_dir.iterdir():
                        if img_file.is_file():
                            dataset.append({
                                'body_part': body_part,
                                'patient_id': patient_id,
                                'label': label,
                                'image_path': str(img_file)
                            })
    return dataset


# this function get part and know what kind of part to train, save model and save plots
def trainPart(part):
    THIS_FOLDER = Path(__file__).resolve().parent
    image_dir = THIS_FOLDER / 'Dataset'
    data = load_path(image_dir, part)
    labels = []
    filepaths = []

    for row in data:
        labels.append(row['label'])
        filepaths.append(row['image_path'])

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    images = pd.concat([filepaths, labels], axis=1)

    patient_df = pd.DataFrame(data)
    unique_patients = patient_df['patient_id'].unique()

    train_patients, test_patients = train_test_split(
        unique_patients,
        train_size=0.9,
        random_state=42,
        shuffle=True
    )

    train_df = patient_df[
        patient_df['patient_id'].isin(train_patients)
    ]

    test_df = patient_df[
        patient_df['patient_id'].isin(test_patients)
    ]

    train_df = train_df.rename(columns={
        'image_path': 'Filepath',
        'label': 'Label'
    })

    test_df = test_df.rename(columns={
        'image_path': 'Filepath',
        'label': 'Label'
    })

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=clahe_preprocessing,
        validation_split=0.2,
        rotation_range=10,
        zoom_range=0.08,
        width_shift_range=0.08,
        height_shift_range=0.08,
        brightness_range=[0.85, 1.15],
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=clahe_preprocessing
    )

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(320, 320),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=16,
        shuffle=True,
        seed=42,
        subset='training'
    )

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_images.classes),
        y=train_images.classes
    )
    class_weights = dict(enumerate(class_weights))
    print("Class Weights:", class_weights)

    val_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=clahe_preprocessing,
        validation_split=0.2
    )

    val_images = val_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(320, 320),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=16,
        shuffle=False,
        seed=42,
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(320, 320),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=16,
        shuffle=False
    )

    pretrained_model = tf.keras.applications.densenet.DenseNet121(
        input_shape=(320, 320, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    for layer in pretrained_model.layers[:-120]:
        layer.trainable = False

    for layer in pretrained_model.layers[-120:]:
        layer.trainable = True

    inputs = pretrained_model.input
    x = tf.keras.layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(pretrained_model.output)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.20)(x)

    x = tf.keras.layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.Dropout(0.15)(x)

    outputs = tf.keras.layers.Dense(
        2,
        activation='softmax'
    )(x)
    model = tf.keras.Model(inputs, outputs)

    print("-------Training " + part + "-------")

    plot_dir = THIS_FOLDER / "plots" / "FractureDetection" / part
    plot_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = THIS_FOLDER / "weights"
    weights_dir.mkdir(exist_ok=True)
    best_weight_path = weights_dir / f"DenseNet121_{part}_best.keras"

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=CategoricalFocalLoss(gamma=2.0, alpha=0.25),
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=7,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        mode='max',
        factor=0.5,
        patience=4,
        verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_weight_path),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=35,
        callbacks=[early_stop, reduce_lr, checkpoint],
        class_weight=class_weights
    )

    model = tf.keras.models.load_model(str(best_weight_path), compile=False)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=CategoricalFocalLoss(gamma=2.0, alpha=0.25),
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    results = model.evaluate(test_images, verbose=0)
    print(part + " Results:")
    print(results)
    print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")
    print(f"Test Recall: {np.round(results[2] * 100, 2)}%")
    print(f"Test Precision: {np.round(results[3] * 100, 2)}%")
    print(f"Test AUC: {np.round(results[4] * 100, 2)}%")

    # create plots for accuracy and save it
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    figAcc = plt.gcf()
    figAcc.savefig(plot_dir / "Accuracy.jpeg")
    plt.clf()

    # create plots for loss and save it
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    figAcc = plt.gcf()
    figAcc.savefig(plot_dir / "Loss.jpeg")
    plt.clf()

    # create recall plot and save it
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    figRecall = plt.gcf()
    figRecall.savefig(plot_dir / "Recall.jpeg")
    plt.clf()

    # create AUC plot and save it
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    figAUC = plt.gcf()
    figAUC.savefig(plot_dir / "AUC.jpeg")
    plt.clf()

if __name__ == "__main__":
    categories_parts = ["Elbow", "Hand", "Shoulder"]
    for category in categories_parts:
        trainPart(category)

