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

        parent = os.path.abspath(os.path.join(train_dir, ".."))
        _save_file_dir = os.path.join(parent, "images_numpy_dataset.npz")
        self.save_file_dir = _save_file_dir

    def save_numpy_dataset(self):
        #
        # Create the saved dataset
        print("Creating .npz save file...")
        all_images = []
        #
        # Load training folder
        if os.path.isdir(self.train_dir):
            for filename in os.listdir(self.train_dir):
                if not filename.lower().endswith(".png"):
                    continue
                img_data = load_image(os.path.join(self.train_dir, filename), True)
                if img_data:
                    all_images.append(img_data)
        #
        # Load test folder
        if os.path.isdir(self.test_dir):
            for filename in os.listdir(self.test_dir):
                if not filename.lower().endswith(".png"):
                    continue
                img_data = load_image(os.path.join(self.test_dir, filename), False)
                if img_data:
                    all_images.append(img_data)

        random.shuffle(all_images)
        n_test = max(1, int(len(all_images) * self.test_fraction))
        test_data = all_images[:n_test]
        train_data = all_images[n_test:]

        if len(train_data) == 0:
            raise RuntimeError("No training images found. Check train_dir path and contents.")
        #
        # Stack the numpy arrays so they can be saved in blocks
        X_train = np.stack([t for t, _ in train_data]).astype("float32")
        y_train = np.array([t for _, t in train_data], dtype=int)
        X_test = np.stack([t for t, _ in test_data]).astype("float32")
        y_test = np.array([t for _, t in test_data], dtype=int)
        #
        # Save the numpy-ized data
        np.savez_compressed(
            self.save_file_dir,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

    def load(self):
        if not os.path.exists(self.save_file_dir):
            self.save_numpy_dataset()

        print(f"Loading saved dataset: data/images_numpy_dataset.npz")
        data = np.load(self.save_file_dir)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"]



