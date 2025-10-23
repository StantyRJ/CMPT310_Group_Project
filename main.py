import os
from PIL import Image
from lib.KNN import KNN

image_dir = os.path.join("ImageDistorter", "Images", "Output")

data = []

# Read all files in the folder
for filename in os.listdir(image_dir):
    if filename.lower().endswith(".png"): # Check if it's png
        label = filename[0] # Label with first letter
        filepath = os.path.join(image_dir,filename) # get the real path
        try:
            img = Image.open(filepath).convert("L") # Open it as img
            data.append((img,label))
        except Exception as e:
            print(f"Skipping {filename}: {e}")

print(f"Loaded {len(data)} images") # Hopefully this > 0

# Run KNN
if len(data) > 1:
    print(KNN(data, 5))