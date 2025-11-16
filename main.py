import os
from PIL import Image, ImageFilter
from torchvision import transforms
import torch
import random
import string
import numpy as np

from lib.KNN import KNN
from lib.KNN import KNNOpt
from lib.SVM import *
from lib.CNN import *


image_dir = os.path.join("data", "distorted")
other_dir = os.path.join("data", "characters")

def extract_label(filename):
    """
    filename: name of the file
    this function grabs numbers before first underscroll, returns the ascii output
    """
    base = os.path.basename(filename)
    number = base.split('_')[0]
    return chr(int(number))

def testKNN():
    if __name__=="__main__":
        data = []

        # Read all png files in the folders
        all_filepaths_filenames =   [(os.path.join(image_dir, filename),filename) for filename in os.listdir(image_dir) if filename.lower().endswith(".png")] \
                                + [(os.path.join(other_dir, filename),filename) for filename in os.listdir(other_dir) if filename.lower().endswith(".png")]

        for filepath,filename in all_filepaths_filenames:
            try:
                img = Image.open(filepath).convert("L").resize((64,64)) # Open it as greyscale img
                # store in data array as a binary vector
                arrayed = np.array(img)
                binarized = (arrayed > 128).astype(np.uint8).flatten()
                data.append((binarized,filename))
            except Exception as e:
                print(f"Skipping {filepath}: {e}")

        print(f"Loaded {len(data)} images") # Hopefully this > 0
    
    
        print(KNNOpt(data, 1, 1))
    """
    # Run KNN
    if len(data) > 1:
        # After confirming data
        # Test K's, 1 through 25
        results = []
        for K in range(1,26):
            print(f"Running KNN for K = {K}")
            results.append(KNN(data,K))
            print(f"Training for K = {K} has finished")
            print(f"K = {K}: Accuracy = {results[K-1]}")
        maxAccuracy = 0
        bestK = 0
        # Iterate through results to return the best K and its accuracy
        for i in range(1,25):
            if results[i] > maxAccuracy:
                maxAccuracy = results[i]
                bestK = i+1
        print(f"Best K is {K} with an accuracy of {maxAccuracy}")
        """

def testSVM(image_dir):

    all_filepaths_filenames =   [(os.path.join(image_dir, filename),filename) for filename in os.listdir(image_dir) if filename.lower().endswith(".png")] \
                              + [(os.path.join(other_dir, filename),filename) for filename in os.listdir(other_dir) if filename.lower().endswith(".png")]

    X = []
    y_labels = []
    # Create X and y_labels:
    for filepath,filename in all_filepaths_filenames:
        label = extract_label(filename)
        try:
            img = Image.open(filepath).convert("L").resize((64,64)) # Open it as greyscale img
            arr = np.array(img)
            binary = (arr > 128).astype(int).flatten()
            X.append(binary)
            y_labels.append(label)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
    X = np.array(X)
    y_labels = np.array(y_labels)
    # Train the model
    models = SVM_multiclass(X, y_labels, lr=0.1)
    # Run predictions
    predictions = predict_multiclass(X, models)

    # Test predictions against actual labels
    accuracy = np.mean(predictions == y_labels)
    print(f"Training accuracy: {accuracy:.2f}")


characters = string.ascii_uppercase + string.ascii_lowercase + string.digits  # 62 classes
char_to_idx = {c: i for i, c in enumerate(characters)}
idx_to_char = {i: c for i, c in enumerate(characters)}

# Set transforms
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64,64)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def prepare_dataset(training_dir, test_dir, test_fraction=0.1):
    """
    Combine training and test directories, randomly select a holdout test set,
    and return train_data and test_data lists.
    """
    all_images = []

    # Load training images
    for filename in os.listdir(training_dir):
        if filename.lower().endswith(".png"):
            label_char = extract_label(filename)
            if label_char not in char_to_idx:
                continue
            label = char_to_idx[label_char]
            filepath = os.path.join(training_dir, filename)
            try:
                img = Image.open(filepath)
                all_images.append((img, label, train_transform))
            except:
                continue

    # Load test images
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(".png"):
            label_char = extract_label(filename)
            if label_char not in char_to_idx:
                continue
            label = char_to_idx[label_char]
            filepath = os.path.join(test_dir, filename)
            try:
                img = Image.open(filepath)
                all_images.append((img, label, test_transform))
            except:
                continue

    # Shuffle and split
    random.shuffle(all_images)
    n_test = max(1, int(len(all_images) * test_fraction))
    test_data = all_images[:n_test]
    train_data = all_images[n_test:]

    # Apply transforms now
    train_data = [(transform(img), label) for img, label, transform in train_data]
    test_data = [(transform(img), label) for img, label, transform in test_data]

    return train_data, test_data

def train_and_test_CNN(training_dir, test_dir, test_fraction=0.1):
    train_data, test_data = prepare_dataset(training_dir, test_dir, test_fraction)

    print(f"Training on {len(train_data)} images, testing on {len(test_data)} images")
    model = CNN(train_data, epochs=100, batch_size=64)
    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total = 0
    skipped = 0
    
    for tensor, original_label in test_data:
        tensor = (tensor - model.norm_mean) / model.norm_std
        tensor = tensor.unsqueeze(0).to(device)
        
        if original_label in model.label_map:
            mapped_label = model.label_map[original_label]
        else:
            skipped += 1
            continue
        
        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(1).item()
        
        total += 1
        if pred == mapped_label:
            correct += 1
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Holdout Accuracy: {accuracy:.2f}% ({correct}/{total})")
    if skipped > 0:
        print(f"Skipped {skipped} samples with unseen labels")
    
    return model


train_and_test_CNN(training_dir=image_dir, test_dir=other_dir, test_fraction=0.1)
#testKNN()
#testKNN()