import os
from PIL import Image, ImageFilter
from lib.KNN import KNN
from lib.SVM import *
import numpy as np

image_dir = os.path.join("data", "distorted")

def testKNN(image_dir):
    data = []

    # Read all files in the folder
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(".png"): # Check if it's png
            label = filename[0] # Label with first letter
            filepath = os.path.join(image_dir,filename) # get the real path
            try:
                img = Image.open(filepath).convert("L") # Open it as greyscale img
                # store in data array as a binary vector
                arrayed = np.array(img)
                binarized = (arrayed > 128).astype(np.uint8).flatten()
                data.append((binarized,filename))
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    print(f"Loaded {len(data)} images") # Hopefully this > 0

    # Run KNN
    if len(data) > 1:
        print(KNN(data, 5))

def extract_label(filename):
    """
    filename: name of the file
    this function grabs numbers before first underscroll, returns the ascii output
    """
    base = os.path.basename(filename)
    number = base.split('_')[0]
    return chr(int(number))

def testSVM(image_dir):
    X = []
    y_labels = []
    # Create X and y_labels:
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(".png"): # Check if png
            label = extract_label(filename)
            filepath = os.path.join(image_dir,filename) # get the real path
            try:
                img = Image.open(filepath).convert("L").resize((64,64)) # Open it as img
                arr = np.array(img)
                binary = (arr > 128).astype(int)
                X.append(binary.flatten())
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

testKNN(image_dir)
#testSVM(image_dir)