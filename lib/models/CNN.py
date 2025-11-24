import os
from typing import Sequence, Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
from .base import Classifier
from typing import Union, List, Dict, Optional

class CNNShape(nn.Module):
    def __init__(self, name="CNN", input_shape=(1, 32, 32), num_classes=62,
                 conv_channels=(64, 128, 256), fc_units=1024, dropout=0.4):
        super().__init__()
        self.name = name
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_ch = input_shape[0]
        H, W = input_shape[1], input_shape[2]

        # Build conv layers
        for ch in conv_channels:
            self.layers.append(nn.Conv2d(in_ch, ch, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm2d(ch))
            in_ch = ch
            # MaxPool reduces spatial size by factor of 2
            H = H // 2
            W = W // 2

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        flattened_size = conv_channels[-1] * H * W
        self.fc1 = nn.Linear(flattened_size, fc_units)
        self.fc2 = nn.Linear(fc_units, num_classes)

    def forward(self, x):
        for conv, bn in zip(self.layers, self.bns):
            x = self.pool(F.relu(bn(conv(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class CNNClassifier(Classifier):
    """
    Unified CNN classifier using a CNNShape definition.
    """

    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 64,
        device: Optional[str] = None,
        lr: float = 0.002,
        weight_decay: float = 0.01,
        cnn_shape: Optional[CNNShape] = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.cnn_shape = cnn_shape  # Allow custom CNN architectures
        self.model: Optional[CNNShape] = None

        self.norm_mean: Optional[torch.Tensor] = None
        self.norm_std: Optional[torch.Tensor] = None
        self.label_map: Optional[dict] = None

    def _prepare_dataset(self, X: Sequence, y: Sequence) -> TensorDataset:
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(np.asarray(X), dtype=torch.float32)
        y = torch.tensor(np.asarray(y), dtype=torch.long)
        return TensorDataset(X, y)

    def fit(self, X: Sequence, y: Sequence) -> None:
        # Map labels to 0..num_classes-1
        unique_labels = sorted(set(y))
        self.label_map = {old: new for new, old in enumerate(unique_labels)}
        mapped_labels = torch.tensor([self.label_map[label] for label in y], dtype=torch.long)

        X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.norm_mean = X_tensor.mean()
        self.norm_std = X_tensor.std()
        X_tensor = (X_tensor - self.norm_mean) / self.norm_std

        dataset = TensorDataset(X_tensor, mapped_labels)
        val_size = int(0.15 * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        num_classes = len(unique_labels)
        self.model = self.cnn_shape or CNNShape(num_classes=num_classes)
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=len(train_loader), pct_start=0.3
        )

        best_val_acc = 0.0
        best_state = None

        for epoch in tqdm(range(self.epochs), desc="Epochs", leave=False):
            self.model.train()
            train_correct, train_total = 0, 0
            for x_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()

                _, predicted = outputs.max(1)
                train_total += y_batch.size(0)
                train_correct += predicted.eq(y_batch).sum().item()

            train_acc = 100 * train_correct / train_total

            self.model.eval()
            val_correct, val_total = 0, 0
            for x_batch, y_batch in tqdm(val_loader, desc="Validation", leave=False):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                with torch.no_grad():
                    outputs = self.model(x_batch)
                    _, predicted = outputs.max(1)
                    val_total += y_batch.size(0)
                    val_correct += predicted.eq(y_batch).sum().item()

            val_acc = 100 * val_correct / val_total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = self.model.state_dict().copy()

            if (epoch + 1) % 10 == 0:
                tqdm.write(f"Epoch {epoch+1}/{self.epochs} - Train: {train_acc:.2f}%, Val: {val_acc:.2f}%")

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X: Sequence) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")

        X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
        X_tensor = (X_tensor - self.norm_mean) / self.norm_std
        X_tensor = X_tensor.to(self.device)

        self.model.eval()
        preds = []
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False)
        for batch in tqdm(loader, desc="Predicting", leave=False):
            x_batch = batch[0].to(self.device)
            with torch.no_grad():
                outputs = self.model(x_batch)
                pred = outputs.argmax(1).cpu().numpy()
                preds.extend(pred)

        return np.array(preds)

class CNNShapeTester:
    """
    Given a dataset builder (like PNGDataset) and multiple CNNShape configurations,
    trains and evaluates each shape. Reports holdout accuracy for comparison.
    """

    def __init__(self, dataset_builder, epochs: int = 20, batch_size: int = 64, device: Optional[str] = None):
        """
        dataset_builder: object with a .load() method that returns
            X_train, y_train, X_test, y_test
        """
        self.dataset_builder = dataset_builder
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.results: Dict[str, float] = {}  # maps shape name to holdout accuracy

        # Load the dataset arrays
        self.X_train, self.y_train, self.X_test, self.y_test = self.dataset_builder.load()

    def test_shapes(self, shapes: List[CNNShape]):
        """
        Accepts a list of CNNShape instances, trains a CNNClassifier for each, evaluates holdout accuracy.
        Returns a dict mapping shape names to accuracy.
        """
        for shape in shapes:
            print(f"\nTesting shape: {shape.name}")
            model = CNNClassifier(
                epochs=self.epochs,
                batch_size=self.batch_size,
                cnn_shape=shape,
                device=self.device
            )

            # Run training
            model.fit(self.X_train, self.y_train)

            # Evaluate holdout set
            holdout_preds = model.predict(self.X_test)
            correct = sum(p == t for p, t in zip(holdout_preds, self.y_test))
            total = len(self.y_test)
            accuracy = 100.0 * correct / total if total > 0 else 0.0
            print(f"Shape {shape.name} Holdout Accuracy: {accuracy:.2f}% ({correct}/{total})")

            self.results[shape.name] = accuracy

        return self.results

    def best_shape(self) -> Optional[str]:
        """Return the shape name with highest accuracy, or None if no results."""
        if not self.results:
            return None
        return max(self.results, key=self.results.get)