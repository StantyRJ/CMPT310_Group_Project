import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Sequence, Optional, Dict

from tqdm import tqdm
from .base import Classifier


class SVMClassifier(Classifier):
    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: float = "scale",
        coef0: float = 0.0,
        degree: int = 3,
        shrinking: bool = True,
        tol: float = 1e-3,
        cache_size: float = 200.0,
        probability: bool = True
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.shrinking = shrinking
        self.tol = tol
        self.cache_size = cache_size
        self.probability = probability
        
        self.model: Optional[SVC] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_map: Optional[Dict[int, int]] = None

    def fit(self, X: Sequence, y: Sequence):
        X = np.asarray(X, dtype=np.float32)

        # Flatten if image: (N, C, H, W) → (N, features)
        if X.ndim > 2:
            X = X.reshape(len(X), -1)

        # Map labels to 0..num_classes-1
        unique_labels = sorted(set(y))
        self.label_map = {old: new for new, old in enumerate(unique_labels)}
        mapped_y = np.array([self.label_map[val] for val in y], dtype=np.int32)

        # Normalize like CNN
        self.scaler = StandardScaler()
        X_norm = self.scaler.fit_transform(X)

        # Create SVM with probability support
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            coef0=self.coef0,
            degree=self.degree,
            shrinking=self.shrinking,
            tol=self.tol,
            cache_size=self.cache_size,
            probability=self.probability
        )

        self.model.fit(X_norm, mapped_y)

    def predict(self, X: Sequence):
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X, dtype=np.float32)

        if X.ndim > 2:
            X = X.reshape(len(X), -1)

        X_norm = self.scaler.transform(X)
        pred_idx = self.model.predict(X_norm)

        # Reverse map
        reverse_map = {v: k for k, v in self.label_map.items()}
        return np.array([reverse_map[i] for i in pred_idx])

    def predict_conf(self, X: Sequence) -> np.ndarray:
        """
        Returns probability matrix (N, num_classes) in ORIGINAL label order.
        Matches output of CNNClassifier.predict_conf().
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict_conf().")

        X = np.asarray(X, dtype=np.float32)

        if X.ndim > 2:
            X = X.reshape(len(X), -1)

        X_norm = self.scaler.transform(X)

        # Predict probs in mapped-label order
        mapped_probs = self.model.predict_proba(X_norm)

        # Reorder outputs to match ORIGINAL dataset label order
        reverse_map = {new: old for old, new in self.label_map.items()}
        ordered_labels = [reverse_map[i] for i in range(len(reverse_map))]

        # mapped_probs[:, i] corresponds to label index i
        # We just maintain the same ordering and return it
        return mapped_probs

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("No model to save.")

        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "label_map": self.label_map
        }, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.label_map = data["label_map"]

        tqdm.write(f"Model loaded from {path}")
