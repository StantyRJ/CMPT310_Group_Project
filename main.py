import json

from tqdm import tqdm
from lib.models import KNNClassifier, SVMClassifier, CNNClassifier
from lib.datasets.png_dataset import PNGDataset
from lib.datasets.emnist_dataset import EMNISTCSVProvider
from lib.models.CNN import CNNShape, CNNShapeTester
from lib.pipeline import run_training
from datetime import datetime

def run_png_knn():
    dataset = PNGDataset("data/distorted", test_dir="data/characters", test_fraction=0.1)
    model = KNNClassifier(K=3)
    run_training(model, dataset)

def run_png_svm():
    dataset = PNGDataset("data/distorted", test_dir="data/characters", test_fraction=0.1)
    model = SVMClassifier()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_training(model, dataset, save_file=f"models/svm_png_{timestamp}.pt")

def run_emnist_svm():
    dataset = EMNISTCSVProvider(csv_path="emnist-balanced-train.csv", test_fraction=0.1, max_samples=5000)
    model = SVMClassifier()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_training(model, dataset, save_file=f"models/svm_emnist_{timestamp}.pt")

def run_png_cnn():
    dataset = PNGDataset("data/distorted", test_dir="data/characters", test_fraction=0.1)
    cnn_shape = CNNShape(name="CNN_png", input_shape=[1, 64, 64], conv_channels=[64, 128, 128, 128], fc_units=[512, 256])
    model = CNNClassifier(
        epochs=500,
        batch_size=128,
        cnn_shape=cnn_shape,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_training(model, dataset, save_path=f"models/cnn_png_{timestamp}.pt")

def run_png_cnn_shapes():
    # Load dataset
    dataset = PNGDataset(
        train_dir="data/distorted",
        test_dir="data/characters",
        test_fraction=0.1
    )

    # Define 10 different CNN shapes
    # Load JSON file
    with open("cnn_shapes.json", "r") as f:
        shapes_data = json.load(f)

    # Reconstruct CNNShape objects
    shapes = []
    for d in shapes_data:
        shape = CNNShape(
            name=d["name"],
            input_shape=tuple(d["input_shape"]),
            conv_channels=d["conv_channels"],
            fc_units=d["fc_units"]
        )
        shapes.append(shape)
    
    # Initialize tester
    tester = CNNShapeTester(dataset, epochs=100, batch_size=64)

    # Run tests
    results = tester.test_shapes(shapes)

    # Print results
    tqdm.write("\n=== Shape Performance ===")
    for shape_name, acc in results.items():
        tqdm.write(f"{shape_name}: {acc:.2f}%")

    tqdm.write(f"\nBest shape: {tester.best_shape()} with accuracy {results[tester.best_shape()]:.2f}%")

def run_emnist_cnn():
    dataset = EMNISTCSVProvider("emnist-balanced-train.csv", test_fraction=0.1, max_samples=20000)
    model = CNNClassifier(
        epochs=10,
        batch_size=128
    )
    run_training(model, dataset)

# NOTE: EMNIST database: https://www.kaggle.com/datasets/crawford/emnist?resource=download
# Please download the needed sets manually. MNIST files also need to be downloaded separately.

if __name__ == "__main__":
    # Choose which example to run
    run_png_svm()
    # run_png_knn()
    # run_emnist_svm()
    # run_png_cnn()
    # run_png_cnn_shapes()
    # run_emnist_cnn()
