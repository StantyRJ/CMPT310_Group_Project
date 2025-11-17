import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def CNN(data, epochs=100, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tqdm.write(f"CNN using device=\"{device}\"")
    
    images = torch.stack([d[0] for d in data])
    labels = torch.tensor([d[1] for d in data], dtype=torch.long)
    
    unique_labels = sorted(set(labels.tolist()))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = torch.tensor([label_map[label.item()] for label in labels], dtype=torch.long)
    num_classes = len(unique_labels)
    
    mean = images.mean()
    std = images.std()
    images = (images - mean) / std
    
    dataset = TensorDataset(images, labels)
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = SimpleCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.002, epochs=epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.3
    )
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
        model.train()
        train_correct = 0
        train_total = 0
        
        # Batch loop with progress bar
        for x_batch, y_batch in tqdm(train_loader, desc="train_loader", leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        for x_batch, y_batch in tqdm(val_loader, desc="Validation Batches", leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                outputs = model(x_batch)
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch {epoch+1}/{epochs} - Train: {train_acc:.2f}%, Val: {val_acc:.2f}%")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.norm_mean = mean
    model.norm_std = std
    model.label_map = label_map

    # Save the model
    torch.save(
        {
            'modelState': model.state_dict(),
            'mean': model.norm_mean,
            'std': model.norm_std,
            'labelMap': model.label_map
        },
        "../CNN_model.pth")
    
    return model

def load_cnn_model():
    #
    # Load the saved model
    if not os.path.exists("../CNN_model.pth"):
        return ""
    #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load("../CNN_model.pt", map_location=device)
    #
    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(saved["modelState"])
    model.eval()    

    model.norm_mean = saved["mean"]
    model.norm_std  = saved["std"]
    model.label_map = saved["labelMap"]

    return model
    
