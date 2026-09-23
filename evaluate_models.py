from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc, 
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from core.predictions import predict
from core.gradcam import save_gradcam
import os
import shutil


# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset" / "valid"
PLOTS_DIR = BASE_DIR / "plots" / "Evaluation"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

BODY_PARTS = ["Elbow", "Hand", "Shoulder"]
FRACTURE_CLASSES = ["fractured", "normal"]


def collect_body_part_samples():
    """Collect one sample path for every image in the validation set with its true body-part label."""
    samples = []  # (image_path, true_label)

    for part in BODY_PARTS:
        part_dir = DATASET_DIR / part
        if not part_dir.exists():
            print(f"Skipping missing directory: {part_dir}")
            continue

        for patient_dir in part_dir.iterdir():
            if not patient_dir.is_dir():
                continue

            for study_dir in patient_dir.iterdir():
                if not study_dir.is_dir():
                    continue

                for img_path in study_dir.iterdir():
                    if img_path.is_file():
                        samples.append((img_path, part))

    return samples



def collect_fracture_samples(part):
    """Collect validation samples for one body part with true fracture labels."""
    samples = []  # (image_path, true_label)

    part_dir = DATASET_DIR / part
    if not part_dir.exists():
        print(f"Skipping missing directory: {part_dir}")
        return samples

    for patient_dir in part_dir.iterdir():
        if not patient_dir.is_dir():
            continue

        for study_dir in patient_dir.iterdir():
            if not study_dir.is_dir():
                continue

            study_name = study_dir.name.lower()
            if study_name.endswith("positive"):
                true_label = "fractured"
            elif study_name.endswith("negative"):
                true_label = "normal"
            else:
                continue

            for img_path in study_dir.iterdir():
                if img_path.is_file():
                    samples.append((img_path, true_label))

    return samples



def save_confusion_matrix(y_true, y_pred, labels, title, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def save_roc_curve(y_true, y_scores, positive_label, title, output_path):
    """
    Save ROC curve and print AUC score.

    Parameters:
        y_true : list[str]
            True class labels.
        y_scores : list[float]
            Probability scores for the positive class.
        positive_label : str
            Label considered as the positive class.
        title : str
            Plot title.
        output_path : Path
            File path to save the figure.
    """
    # Convert string labels to binary values
    y_true_binary = [1 if label == positive_label else 0 for label in y_true]

    # Compute ROC
    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"{title} AUC: {roc_auc:.4f}")

    return roc_auc


def predict_with_probability(img_path, model_type):
    """
    Returns:
        label, confidence_percent
    """
    return predict(img_path, model_type, return_confidence=True)

def evaluate_body_part_classifier():
    print("=" * 80)
    print("EVALUATING BODY PART CLASSIFIER")
    print("=" * 80)

    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    samples = collect_body_part_samples()
    y_true = []
    y_pred = []
    y_scores = []

    # Convert each predicted label + confidence into class probabilities
    for img_path, true_label in samples:
        pred_label, confidence, probs = predict(
            str(img_path),
            "Parts",
            return_confidence=True,
            return_probs=True
        )

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_scores.append(probs)

        

    # Classification report
    print(classification_report(
        y_true,
        y_pred,
        labels=BODY_PARTS,
        digits=4
    ))

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="macro"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    print("\nSUMMARY METRICS")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Confusion matrix
    save_confusion_matrix(
        y_true,
        y_pred,
        BODY_PARTS,
        "Body Part Classification",
        PLOTS_DIR / "BodyParts_ConfusionMatrix.png",
    )

    # ----- Multiclass ROC (One-vs-Rest) -----
    print("Total samples:", len(y_true))
    print("y_scores shape:", np.array(y_scores).shape)
    y_true_bin = label_binarize(y_true, classes=BODY_PARTS)
    y_scores = np.array(y_scores)
    assert len(y_scores.shape) == 2
    assert y_scores.shape[1] == 3

    fig, ax = plt.subplots(figsize=(7, 7))
    auc_values = []

    for i, class_name in enumerate(BODY_PARTS):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)

        ax.plot(fpr, tpr, linewidth=2,
                label=f"{class_name} (AUC = {roc_auc:.4f})")

    macro_auc = np.mean(auc_values)

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Body Parts ROC Curve (Macro AUC = {macro_auc:.4f})")
    ax.legend(loc="lower right")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "BodyParts_ROC_Curve.png", dpi=300)
    plt.close(fig)

    print(f"Body Parts Macro AUC: {macro_auc:.4f}")

    return {
    "Model": "BodyParts",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "AUC": macro_auc
}



def evaluate_fracture_classifier(part):
    print("=" * 80)
    print(f"EVALUATING {part.upper()} FRACTURE CLASSIFIER")
    print("=" * 80)

    samples = collect_fracture_samples(part)
    y_true = []
    y_pred = []
    y_scores = []   # Probability of 'fractured'

        # Error analysis folders
    ERROR_DIR = PLOTS_DIR / "Errors"

    false_negative_dir = ERROR_DIR / f"{part}_FalseNegative"
    false_positive_dir = ERROR_DIR / f"{part}_FalsePositive"

    false_negative_dir.mkdir(parents=True, exist_ok=True)
    false_positive_dir.mkdir(parents=True, exist_ok=True)

    for img_path, true_label in samples:
        # Get predicted label and confidence percentage
        pred_label, confidence = predict(
            str(img_path),
            part,
            return_confidence=True
        )

        y_true.append(true_label)
        y_pred.append(pred_label)

        # Get raw probabilities
        _, _, probs = predict(
            str(img_path),
            part,
            return_confidence=True,
            return_probs=True
        )

        # Probability of fractured class
        fractured_prob = float(probs[0])

        y_scores.append(fractured_prob)

                # -----------------------------
        # Save False Negatives
        # True = fractured
        # Predicted = normal
        # -----------------------------
        if true_label == "fractured" and pred_label == "normal":

            # Copy original image
            shutil.copy(
                img_path,
                false_negative_dir / img_path.name
            )

            # Save GradCAM
            save_gradcam(
                model_type=part,
                img_path=str(img_path),
                output_path=false_negative_dir / f"gradcam_{img_path.name}"
            )

        # -----------------------------
        # Save False Positives
        # True = normal
        # Predicted = fractured
        # -----------------------------
        elif true_label == "normal" and pred_label == "fractured":

            shutil.copy(
                img_path,
                false_positive_dir / img_path.name
            )

            save_gradcam(
                model_type=part,
                img_path=str(img_path),
                output_path=false_positive_dir / f"gradcam_{img_path.name}"
            )


    # ---------------------------------------------------
    # Threshold Optimization
    # ---------------------------------------------------

    y_true_binary = [
        1 if label == "fractured" else 0
        for label in y_true
    ]

    precisions, recalls, thresholds = precision_recall_curve(
        y_true_binary,
        y_scores
    )

    f1_scores = (
        2 * precisions * recalls
    ) / (
        precisions + recalls + 1e-8
    )

    best_index = np.argmax(f1_scores[:-1])

    best_threshold = thresholds[best_index]

    print(f"Best Threshold for {part}: {best_threshold:.4f}")
    print(f"Best F1 Score for {part}: {f1_scores[best_index]:.4f}")


    print(classification_report(
        y_true,
        y_pred,
        labels=FRACTURE_CLASSES,
        digits=4
    ))

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label="fractured"
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label="fractured"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label="fractured"
    )

    print("\nSUMMARY METRICS")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Save confusion matrix
    save_confusion_matrix(
        y_true,
        y_pred,
        FRACTURE_CLASSES,
        f"{part} Fracture Detection",
        PLOTS_DIR / f"{part}_ConfusionMatrix.png",
    )

    # Save ROC curve
    roc_auc = save_roc_curve(
        y_true,
        y_scores,
        positive_label="fractured",
        title=f"{part} ROC Curve",
        output_path=PLOTS_DIR / f"{part}_ROC_Curve.png",
    )

    return {
        "Model": part,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": roc_auc
    }


def main():

    results = []

    results.append(
        evaluate_body_part_classifier()
    )

    for part in BODY_PARTS:

        results.append(
            evaluate_fracture_classifier(part)
        )

    import pandas as pd

    pd.DataFrame(results).to_csv(
        PLOTS_DIR / "Evaluation_Summary.csv",
        index=False
    )

    print("\n")
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(pd.DataFrame(results))

    print("\nEvaluation complete.")
    print(f"Results saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()