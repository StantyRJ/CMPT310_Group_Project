from typing import Tuple, Sequence, List
import os
import random
from PIL import Image
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Reuse the same character set mapping as original script
import string
characters = string.ascii_uppercase + string.ascii_lowercase + string.digits
char_to_idx = {c: i for i, c in enumerate(characters)}


train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    #transforms.RandomRotation(10),
    #transforms.RandomAffine(0, translate=(0.1, 0.1)),
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


def load_image(path, for_train=True):
    name = os.path.basename(path)
    label_c = extract_label(name)
    #
    if label_c not in char_to_idx:
        return None
    #
    label = char_to_idx[label_c]
    img = Image.open(path)
    tensor = (train_transform if for_train else test_transform)(img)
    #
    arrayed = tensor.numpy().astype("float32")
    return arrayed, label

class PNGDataset:
    """Loads PNGs from training and optional test dirs and returns numpy arrays.

    Returns X_train, y_train, X_test, y_test
    X arrays are float32 tensors in shape (N, C, H, W) normalized to [-1,1]
    """

    def __init__(self, train_dir: str, test_dir: str = None, test_fraction: float = 0.1):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.test_fraction = test_fraction

        # No on-disk cache: always build dataset from image folders

    def load(self):
        """
        Build dataset arrays from PNG folders and return X_train, y_train, X_test, y_test.
        """
        all_images = []

        # Load training folder
        if os.path.isdir(self.train_dir):
            for filename in os.listdir(self.train_dir):
                if not filename.lower().endswith(".png"):
                    continue
                img_data = load_image(os.path.join(self.train_dir, filename), True)
                if img_data:
                    all_images.append(img_data)

        # Load test folder (optional)
        if os.path.isdir(self.test_dir):
            for filename in os.listdir(self.test_dir):
                if not filename.lower().endswith(".png"):
                    continue
                img_data = load_image(os.path.join(self.test_dir, filename), False)
                if img_data:
                    all_images.append(img_data)

        if len(all_images) == 0:
            raise RuntimeError("No images found. Check train_dir/test_dir paths and contents.")

        # Separate images and labels into X and y
        X = np.stack([img for img, _ in all_images]).astype("float32")
        y = np.array([lbl for _, lbl in all_images], dtype=np.int64)

        # Split into train/test using configured fraction. Prefer stratified split but
        # fall back to a non-stratified split if stratification isn't possible.
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_fraction, stratify=y, random_state=42
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_fraction, random_state=42
            )

        return X_train, y_train, X_test, y_test



