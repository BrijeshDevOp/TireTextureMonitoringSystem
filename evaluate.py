import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "dataset")
TEST_DIR = os.path.join(DATA_DIR, "testing_data")
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
MODEL_PATH_KERAS = os.path.join(PROJECT_DIR, "model3.keras")
MODEL_PATH_HDF5 = os.path.join(PROJECT_DIR, "model3.hdf5")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")


def load_test_dataset():
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        shuffle=False,
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
    )
    def to_float(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y
    ds = ds.map(to_float)
    return ds


def main():
    # Load model (prefer Keras 3 format)
    if os.path.exists(MODEL_PATH_KERAS):
        model_path = MODEL_PATH_KERAS
    elif os.path.exists(MODEL_PATH_HDF5):
        model_path = MODEL_PATH_HDF5
    else:
        raise FileNotFoundError(
            f"Model file not found. Expected one of: {MODEL_PATH_KERAS} or {MODEL_PATH_HDF5}"
        )

    model = keras.models.load_model(model_path, compile=False)

    test_ds = load_test_dataset()

    y_true = []
    y_prob = []

    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0).reshape(-1)
        y_prob.extend(probs.tolist())
        y_true.extend(y_batch.numpy().reshape(-1).tolist())

    y_true = np.array(y_true).astype(int)
    y_pred = (np.array(y_prob) >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print("Model:", model_path)
    print("Test set size:", len(y_true))
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):\n", cm)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Cracked"], zero_division=0))

    # Create reports directory
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1) Confusion Matrix Heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Normal", "Cracked"], yticklabels=["Normal", "Cracked"]) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(REPORTS_DIR, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # 2) ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    roc_path = os.path.join(REPORTS_DIR, 'roc_curve.png')
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # 3) Precision-Recall Curve
    precs, recs, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recs, precs)
    plt.figure(figsize=(5, 4))
    plt.plot(recs, precs, label=f'AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    pr_path = os.path.join(REPORTS_DIR, 'precision_recall_curve.png')
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()

    # 4) Score Histogram by Class
    y_prob = np.array(y_prob)
    plt.figure(figsize=(6, 4))
    plt.hist(y_prob[np.array(y_true) == 0], bins=25, alpha=0.6, label='Normal', color='#1f77b4')
    plt.hist(y_prob[np.array(y_true) == 1], bins=25, alpha=0.6, label='Cracked', color='#d62728')
    plt.axvline(0.5, color='k', linestyle='--', alpha=0.7, label='Threshold 0.5')
    plt.xlabel('Predicted Probability (Cracked)')
    plt.ylabel('Frequency')
    plt.title('Score Distribution by Class')
    plt.legend()
    hist_path = os.path.join(REPORTS_DIR, 'score_histogram.png')
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()

    print("\nSaved charts:")
    print(" -", cm_path)
    print(" -", roc_path)
    print(" -", pr_path)
    print(" -", hist_path)


if __name__ == "__main__":
    main()
