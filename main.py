from lib.models import KNNClassifier, SVMClassifier, CNNClassifier
from lib.datasets.png_dataset import PNGDataset
from lib.datasets.emnist_dataset import EMNISTCSVProvider
from lib.models.CNN import CNNShape, CNNShapeTester
from lib.pipeline import run_training

def run_png_knn():
    dataset = PNGDataset("data/distorted", test_dir="data/characters", test_fraction=0.1)
    model = KNNClassifier(K=3)
    run_training(model, dataset)

def run_emnist_svm():
    dataset = EMNISTCSVProvider("emnist-balanced-train.csv", test_fraction=0.1, max_samples=5000)
    model = SVMClassifier(lr=0.1)
    run_training(model, dataset)

def run_png_cnn():
    dataset = PNGDataset("data/distorted", test_dir="data/characters", test_fraction=0.1)
    model = CNNClassifier(
        epochs=10,
        batch_size=128
    )
    run_training(model, dataset)

def run_png_cnn_shapes():
    # Load dataset
    dataset = PNGDataset(
        train_dir="data/distorted",
        test_dir="data/characters",
        test_fraction=0.1
    )

    # Define 10 different CNN shapes
    shapes = [
        CNNShape(
            name="lightweight",
            input_shape=(1, 64, 64),
            conv_channels=[16, 32],
            fc_units=64
        ),
        CNNShape(
            name="medium_depth",
            input_shape=(1, 64, 64),
            conv_channels=[32, 64],
            fc_units=128
        ),
        CNNShape(
            name="deep_small_fc",
            input_shape=(1, 64, 64),
            conv_channels=[32, 64, 128],
            fc_units=64
        ),
        CNNShape(
            name="deep_medium_fc",
            input_shape=(1, 64, 64),
            conv_channels=[32, 64, 128],
            fc_units=128
        ),
        CNNShape(
            name="deep_large_fc",
            input_shape=(1, 64, 64),
            conv_channels=[32, 64, 128],
            fc_units=256
        ),
        CNNShape(
            name="wide_conv",
            input_shape=(1, 64, 64),
            conv_channels=[64, 128, 256],
            fc_units=128
        ),
        CNNShape(
            name="narrow_conv",
            input_shape=(1, 64, 64),
            conv_channels=[16, 32, 64],
            fc_units=128
        ),
        CNNShape(
            name="extra_deep",
            input_shape=(1, 64, 64),
            conv_channels=[16, 32, 64, 128],
            fc_units=128
        ),
        CNNShape(
            name="extra_wide",
            input_shape=(1, 64, 64),
            conv_channels=[64, 128, 256, 512],
            fc_units=256
        ),
        CNNShape(
            name="balanced",
            input_shape=(1, 64, 64),
            conv_channels=[32, 64, 128, 64],
            fc_units=128
        )
    ]
    
    # Initialize tester
    tester = CNNShapeTester(dataset, epochs=10, batch_size=64)

    # Run tests
    results = tester.test_shapes(shapes)

    # Print results
    print("\n=== Shape Performance ===")
    for shape_name, acc in results.items():
        print(f"{shape_name}: {acc:.2f}%")

    print(f"\nBest shape: {tester.best_shape()} with accuracy {results[tester.best_shape()]:.2f}%")

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
    # run_png_knn()
    # run_emnist_svm()
    # run_png_cnn()
    run_png_cnn_shapes()
    # run_emnist_cnn()
