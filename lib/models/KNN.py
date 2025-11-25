import numpy as np
from sklearn.neighbors import BallTree
from tqdm import tqdm
from typing import Sequence
from .base import Classifier


class KNNClassifier(Classifier):
    """
    K-Nearest Neighbors classifier using BallTree acceleration.
    
    Attributes:
        K (int): Number of neighbors to consider in prediction.
        X_train (np.ndarray): Flattened training feature vectors.
        y_train (np.ndarray): Training labels.
        tree (BallTree): BallTree built on training data.
    """

    def __init__(self, K: int = 1):
        self.K = K
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.tree: BallTree | None = None

    def fit(self, X: Sequence, y: Sequence) -> None:
        """
        Fit the KNN model by storing training data and building a BallTree.

        Args:
            X: array-like of shape (N, ...) - training features
            y: array-like of shape (N,) - training labels
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Flatten if input has more than 2 dimensions (e.g., images)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        self.X_train = X
        self.y_train = y

        tqdm.write("Building BallTree…")
        self.tree = BallTree(self.X_train, metric="manhattan")
        tqdm.write("BallTree built.")

    def predict(self, X: Sequence) -> np.ndarray:
        """
        Predict labels for the input data using BallTree nearest neighbors.

        Args:
            X: array-like of shape (N, ...) - test features

        Returns:
            np.ndarray: predicted labels of shape (N,)
        """
        if self.tree is None or self.X_train is None or self.y_train is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        preds = []

        tqdm_bar = tqdm(range(len(X)), desc="KNN Predict", leave=False)
        for i in tqdm_bar:
            # Query K nearest neighbors
            dist, idx = self.tree.query([X[i]], k=self.K)
            neighbor_labels = self.y_train[idx[0]]

            # Majority vote
            uniq, counts = np.unique(neighbor_labels, return_counts=True)
            pred = uniq[np.argmax(counts)]
            preds.append(pred)

        return np.array(preds)

    def optimize_K(self, X: Sequence, y: Sequence, k_min: int = 1, k_max: int = 25, show_plot = False):
        """
        Sweep K values to find the one with the highest accuracy.

        Args:
            X: array-like of shape (N, ...) - training features
            y: array-like of shape (N,) - training labels
            k_min: minimum K value to test
            k_max: maximum K value to test

        Returns:
            Tuple[dict, int, float]: (results_dict, best_K, best_accuracy)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        tree = BallTree(X, metric="manhatten")

        # Precompute neighbor indices
        tqdm.write("Precomputing neighbor indices…")
        _, idx = tree.query(X, k=k_max + 1)
        idx = idx[:, 1:]  # remove self

        results = {}
        Ks = range(k_min, k_max + 1)

        tqdm.write("Sweeping K values…")
        for K in tqdm(Ks, desc="Optimize K"):
            correct = 0
            for i in range(len(X)):
                labels = y[idx[i, :K]]
                uniq, counts = np.unique(labels, return_counts=True)
                pred = uniq[np.argmax(counts)]
                if pred == y[i]:
                    correct += 1
            results[K] = correct / len(X)

        best_K = max(results, key=results.get)
        best_acc = results[best_K]

        return results, best_K, best_acc