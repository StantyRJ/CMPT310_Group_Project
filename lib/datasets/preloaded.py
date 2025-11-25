from typing import Tuple

class PreloadedDatasetProvider:
    """
    Wraps preloaded datasets and implements the same interface as PNGDataset.
    This is safe for multiprocessing because all workers use in-memory arrays.
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def load(self) -> Tuple:
        """
        Returns:
            X_train, y_train, X_test, y_test
        """
        return self.X_train, self.y_train, self.X_test, self.y_test
