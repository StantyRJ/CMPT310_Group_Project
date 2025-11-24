from typing import Tuple, Sequence, List
import os
import random
from PIL import Image
import numpy as np
from torchvision import transforms

# Reuse the same character set mapping as original script
import string
characters = string.ascii_uppercase + string.ascii_lowercase + string.digits
char_to_idx = {c: i for i, c in enumerate(characters)}


train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def extract_label(filename: str) -> str:
    base = os.path.basename(filename)
    number = base.split("_")[0]
    return chr(int(number))


class PNGDataset:
    """Loads PNGs from training and optional test dirs and returns numpy arrays.

    Returns X_train, y_train, X_test, y_test
    X arrays are float32 tensors in shape (N, C, H, W) normalized to [-1,1]
    """

    def __init__(self, train_dir: str, test_dir: str = None, test_fraction: float = 0.1):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.test_fraction = test_fraction

    def _load_all(self):
        all_images = []  # list of (tensor, label)
        # load training folder
        if os.path.isdir(self.train_dir):
            for filename in os.listdir(self.train_dir):
                if not filename.lower().endswith(".png"):
                    continue
                label_char = extract_label(filename)
                if label_char not in char_to_idx:
                    continue
                label = char_to_idx[label_char]
                filepath = os.path.join(self.train_dir, filename)
                try:
                    img = Image.open(filepath)
                    tensor = train_transform(img)
                    all_images.append((tensor.numpy(), label))
                except Exception:
                    continue
        # optionally add test_dir images
        if self.test_dir and os.path.isdir(self.test_dir):
            for filename in os.listdir(self.test_dir):
                if not filename.lower().endswith(".png"):
                    continue
                label_char = extract_label(filename)
                if label_char not in char_to_idx:
                    continue
                label = char_to_idx[label_char]
                filepath = os.path.join(self.test_dir, filename)
                try:
                    img = Image.open(filepath)
                    tensor = test_transform(img)
                    all_images.append((tensor.numpy(), label))
                except Exception:
                    continue

        return all_images

    def load(self):
        all_images = self._load_all()
        random.shuffle(all_images)
        n_test = max(1, int(len(all_images) * self.test_fraction))
        test_data = all_images[:n_test]
        train_data = all_images[n_test:]

        if len(train_data) == 0:
            raise RuntimeError("No training images found. Check train_dir path and contents.")

        X_train = [t for t, y in train_data]
        y_train = [y for t, y in train_data]
        X_test = [t for t, y in test_data]
        y_test = [y for t, y in test_data]

        import numpy as np
        X_train = np.stack(X_train).astype('float32')
        X_test = np.stack(X_test).astype('float32')
        y_train = np.array(y_train, dtype=int)
        y_test = np.array(y_test, dtype=int)
        return X_train, y_train, X_test, y_test
