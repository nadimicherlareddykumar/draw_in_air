"""
gesture_classifier.py
---------------------
ML-based gesture recognition using scikit-learn.

Workflow:
  1. COLLECT  – record labelled landmark samples while the app runs.
  2. TRAIN    – fit a Random Forest (or SVM) on the collected dataset.
  3. PREDICT  – classify live landmark vectors in real time.

Gesture labels (must match GESTURE_LABELS below):
  draw          – index finger up only
  erase         – index + middle up (peace sign)
  change_color  – three fingers up (index + middle + ring)
  save          – four fingers up (index–pinky)
  clear         – all five fingers up (open palm)
"""

import os
import csv
import pickle
import numpy as np

# ── scikit-learn imports (only needed for train / predict, not for collect)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[GestureClassifier] scikit-learn not found – ML features disabled.")


# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
GESTURE_LABELS = ["draw", "erase", "change_color", "save", "clear"]

DATASET_DIR  = os.path.join(os.path.dirname(__file__), "dataset")
DATASET_FILE = os.path.join(DATASET_DIR, "gesture_data.csv")
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "models")
MODEL_FILE   = os.path.join(MODEL_DIR,   "gesture_model.pkl")
SCALER_FILE  = os.path.join(MODEL_DIR,   "scaler.pkl")

# Minimum samples per class before training is allowed
MIN_SAMPLES_PER_CLASS = 20


class GestureClassifier:
    """
    Encapsulates data collection, model training, and real-time prediction.
    """

    def __init__(self, use_svm: bool = False):
        """
        Parameters
        ----------
        use_svm : bool
            True  → use SVM  (slower to train, sometimes more accurate)
            False → use Random Forest (fast, robust, recommended default)
        """
        os.makedirs(DATASET_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR,   exist_ok=True)

        self.use_svm = use_svm
        self.model   = None
        self.scaler  = None
        self.is_trained = False

        # Try loading a pre-trained model from disk
        self._load_model()

    # ──────────────────────────────────────────
    #  Data collection
    # ──────────────────────────────────────────
    def collect_sample(self, landmarks_flat: list, label: str):
        """
        Append one labelled sample to the CSV dataset.

        Parameters
        ----------
        landmarks_flat : list[float]
            63-element flattened landmark vector (21 points × x,y,z).
        label : str
            One of GESTURE_LABELS.
        """
        if label not in GESTURE_LABELS:
            print(f"[Collector] Unknown label '{label}'. Choose from {GESTURE_LABELS}")
            return

        with open(DATASET_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([label] + landmarks_flat)

        print(f"[Collector] Saved sample for '{label}'")

    def dataset_stats(self) -> dict:
        """Return {label: count} for the current dataset file."""
        stats = {lbl: 0 for lbl in GESTURE_LABELS}

        if not os.path.exists(DATASET_FILE):
            return stats

        with open(DATASET_FILE, "r") as f:
            for row in csv.reader(f):
                if row and row[0] in stats:
                    stats[row[0]] += 1

        return stats

    # ──────────────────────────────────────────
    #  Training
    # ──────────────────────────────────────────
    def train(self, verbose: bool = True) -> float:
        """
        Train (or retrain) the classifier on the current dataset.

        Returns
        -------
        float
            Test-set accuracy (0-1), or 0.0 if training failed.
        """
        if not SKLEARN_AVAILABLE:
            print("[Trainer] scikit-learn not available.")
            return 0.0

        if not os.path.exists(DATASET_FILE):
            print("[Trainer] No dataset file found. Collect samples first.")
            return 0.0

        # ── Load CSV ──────────────────────────
        X, y = [], []
        with open(DATASET_FILE, "r") as f:
            for row in csv.reader(f):
                if len(row) == 64:          # label + 63 features
                    y.append(row[0])
                    X.append([float(v) for v in row[1:]])

        X, y = np.array(X), np.array(y)

        if len(X) < len(GESTURE_LABELS) * MIN_SAMPLES_PER_CLASS:
            print(
                f"[Trainer] Need at least {MIN_SAMPLES_PER_CLASS} samples per class. "
                f"Current total: {len(X)}"
            )
            return 0.0

        # ── Scale features ────────────────────
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # ── Split ─────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # ── Choose model ──────────────────────
        if self.use_svm:
            self.model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
        else:
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )

        # ── Fit ───────────────────────────────
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # ── Evaluate ──────────────────────────
        y_pred = self.model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        if verbose:
            print(f"\n{'='*45}")
            print(f"  Model      : {'SVM' if self.use_svm else 'Random Forest'}")
            print(f"  Samples    : {len(X)} total")
            print(f"  Test acc.  : {acc*100:.1f}%")
            print(f"{'='*45}")
            print(classification_report(y_test, y_pred))

        # ── Persist ───────────────────────────
        self._save_model()
        return acc

    # ──────────────────────────────────────────
    #  Prediction
    # ──────────────────────────────────────────
    def predict(self, landmarks_flat: list) -> tuple:
        """
        Classify a single landmark vector.

        Returns
        -------
        (label: str, confidence: float)
            ("unknown", 0.0) when the model is not ready.
        """
        if not self.is_trained or self.model is None or self.scaler is None:
            return "unknown", 0.0

        X = np.array(landmarks_flat).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        label = self.model.predict(X_scaled)[0]

        # Confidence: use predict_proba if available
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_scaled)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 1.0   # SVM without probability=True falls here

        return label, confidence

    # ──────────────────────────────────────────
    #  Persistence helpers
    # ──────────────────────────────────────────
    def _save_model(self):
        with open(MODEL_FILE,  "wb") as f:
            pickle.dump(self.model,  f)
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"[Trainer] Model saved → {MODEL_FILE}")

    def _load_model(self):
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
            with open(MODEL_FILE,  "rb") as f:
                self.model  = pickle.load(f)
            with open(SCALER_FILE, "rb") as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
            print("[GestureClassifier] Pre-trained model loaded.")
        else:
            print("[GestureClassifier] No saved model found – rule-based mode active.")
