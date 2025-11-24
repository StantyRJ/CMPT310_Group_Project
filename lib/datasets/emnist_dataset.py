from typing import Tuple, Sequence, Optional
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
import torch.nn.functional as F


def emnist_prepare_dataset(csv_path: str, test_fraction: float = 0.1, max_samples: Optional[int] = None, binarize: bool = False):
    """Load EMNIST CSV and return train/test TensorDatasets ready for CNN wrapper.

    CSV assumed to be: label,pixel1,...,pixel784
    """
    data = np.loadtxt(csv_path, delimiter=",")
    if max_samples is not None:
        data = data[:max_samples]

    y = data[:, 0].astype(int)
    X = data[:, 1:] / 255.0
    if binarize:
        X = (X > 0.5).astype(np.float32)

    X = X.reshape(-1, 1, 28, 28)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    # Resize to 64x64
    X_tensor = F.interpolate(X_tensor, size=(64, 64))
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    n_test = max(1, int(len(dataset) * test_fraction))
    n_train = len(dataset) - n_test
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])
    return train_dataset, test_dataset


class EMNISTCSVProvider:
    def __init__(self, csv_path: str, test_fraction: float = 0.1, max_samples: Optional[int] = None, binarize: bool = False):
        self.csv_path = csv_path
        self.test_fraction = test_fraction
        self.max_samples = max_samples
        self.binarize = binarize

    def load(self):
        train_dataset, test_dataset = emnist_prepare_dataset(
            self.csv_path, test_fraction=self.test_fraction, max_samples=self.max_samples, binarize=self.binarize
        )
        # Convert to arrays to make pipeline simple for non-torch models
        X_train = [x.numpy() for x, y in train_dataset]
        y_train = [int(y.item()) for x, y in train_dataset]
        X_test = [x.numpy() for x, y in test_dataset]
        y_test = [int(y.item()) for x, y in test_dataset]
        import numpy as np
        X_train = np.stack(X_train).astype('float32')
        X_test = np.stack(X_test).astype('float32')
        y_train = np.array(y_train, dtype=int)
        y_test = np.array(y_test, dtype=int)
        return X_train, y_train, X_test, y_test
