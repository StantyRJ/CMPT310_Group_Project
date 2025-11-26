import json

from tqdm import tqdm
from lib.datasets.preloaded import PreloadedDatasetProvider
from lib.models import KNNClassifier, SVMClassifier, CNNClassifier
from lib.datasets.png_dataset import PNGDataset
from lib.datasets.emnist_dataset import EMNISTCSVProvider
from lib.models.CNN import CNNShape, CNNShapeTester
from lib.pipeline import run_training
from datetime import datetime
import matplotlib.pyplot as plot
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def run_png_knn():
    dataset = PNGDataset("data/distorted", test_dir="data/characters", test_fraction=0.1)
    model = KNNClassifier(K=3)
    run_training(model=model, dataset_provider=dataset)

def run_emnist_knn():
    dataset = EMNISTCSVProvider("emnist-balanced-train.csv", test_fraction=0.1, max_samples=20000)
    model = KNNClassifier(K=1)
    run_training(model=model, dataset_provider=dataset)

def run_png_svm():
    dataset = PNGDataset("data/distorted", test_dir="data/characters", test_fraction=0.8)
    model = SVMClassifier()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_training(model, dataset, save_file=f"models/svm_png_{timestamp}.pt")

def run_emnist_svm():
    dataset = EMNISTCSVProvider(csv_path="emnist-balanced-train.csv", test_fraction=0.1, max_samples=5000)
    model = SVMClassifier(kernel="rbf", C=10.0, gamma=0.001, coef0=0.0, degree=3, shrinking=True, tol=0.01, cache_size=200)

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
    run_training(model, dataset, save_file=f"models/cnn_png_{timestamp}.pt")

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
    shape = CNNShape("best_emnist", input_shape=(1, 64, 64), conv_channels = [
            32,
            64,
            128,
            128
        ],
        fc_units= [
            256,
            128
        ])
    model = CNNClassifier(
        epochs=100,
        batch_size=128,
        cnn_shape=shape
    )
    run_training(model, dataset)

def _evaluate_k(k, e_dataset, o_dataset):
    """Runs training for a single K and returns (k, (o_acc, e_acc))."""
    model = KNNClassifier(K=k)

    # run EMNIST
    e_output = run_training(model, e_dataset)
    if isinstance(e_output, dict):
        e_acc = e_output.get("accuracy") or e_output.get("test_accuracy")
        if e_acc is None:
            raise ValueError(f"Missing accuracy key in {e_output.keys()}")
    else:
        e_acc = float(e_output)

    # run PNG dataset
    o_output = run_training(model, o_dataset)
    if isinstance(o_output, dict):
        o_acc = o_output.get("accuracy") or o_output.get("test_accuracy")
        if o_acc is None:
            raise ValueError(f"Missing accuracy key in {o_output.keys()}")
    else:
        o_acc = float(o_output)

    return k, (o_acc, e_acc)


def run_png_knn_sweep():
    # --- Load datasets ---
    e_dataset = EMNISTCSVProvider("emnist-balanced-train.csv",
                                  test_fraction=0.1, max_samples=20000)
    o_dataset = PNGDataset("data/distorted",
                           test_dir="data/characters", test_fraction=0.1)

    a, b, c, d = e_dataset.load()
    e_dataset = PreloadedDatasetProvider(a, b, c, d)

    a, b, c, d = o_dataset.load()
    o_dataset = PreloadedDatasetProvider(a, b, c, d)

    # K values
    K_values = [1, 2, 3, 5, 7, 9, 11, 15]

    # --- Parallel sweep ---
    results = {}
    eval_fn = partial(_evaluate_k, e_dataset=e_dataset, o_dataset=o_dataset)

    with ProcessPoolExecutor(max_workers=10) as executor:
        for k, (o_acc, e_acc) in executor.map(eval_fn, K_values):
            tqdm.write(f"K={k} → o_acc={o_acc:.4f}, e_acc={e_acc:.4f}")
            results[k] = (o_acc, e_acc)

    # --- Best K ---
    best_k = max(results, key=results.get)
    best_acc = results[best_k]

    tqdm.write(f"\n=== Best K: {best_k} with accuracy {best_acc} ===\n")

    # --- Plotting ---
    Ks = list(results.keys())
    o_accs = [results[k][0] for k in Ks]
    e_accs = [results[k][1] for k in Ks]

    plot.plot(Ks, o_accs, marker='o', label="Our Dataset Accuracy")
    plot.plot(Ks, e_accs, marker='s', label="EMNIST Accuracy")

    plot.title("KNN Accuracy for Different K")
    plot.xlabel("K")
    plot.ylabel("Accuracy")
    plot.grid(True)
    plot.legend()
    plot.savefig("knn_sweep.png", dpi=300)
    plot.show()

# NOTE: EMNIST database: https://www.kaggle.com/datasets/crawford/emnist?resource=download
# Please download the needed sets manually. MNIST files also need to be downloaded separately.

if __name__ == "__main__":
    # Choose which example to run
    run_png_knn_sweep()
    # run_png_svm()
    # run_emnist_svm()
    # run_png_cnn()
    # run_png_cnn_shapes()
    # run_emnist_cnn()
    # run_png_knn()
    # run_emnist_knn()
