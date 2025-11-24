# models/base.py
from typing import Any, Sequence


class Classifier:
    """Minimal classifier interface used by the pipeline.


    Implementations should provide fit(X, y) and predict(X).
    X can be a NumPy array, list of vectors, or torch.Tensor depending on the concrete impl.
    y should be a sequence of labels.
    """


    def fit(self, X: Any, y: Sequence) -> None:
        raise NotImplementedError


    def predict(self, X: Any) -> Sequence:
        raise NotImplementedError