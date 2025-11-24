from lib.models import KNNClassifier, SVMClassifier, CNNClassifier
from lib.datasets.png_dataset import PNGDataset
from lib.datasets.emnist_dataset import EMNISTCSVProvider
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
    model = CNNClassifier(epochs=10, batch_size=128)
    run_training(model, dataset)

def run_emnist_cnn():
    dataset = EMNISTCSVProvider("emnist-balanced-train.csv", test_fraction=0.1, max_samples=20000)
    model = CNNClassifier(epochs=10, batch_size=128)
    run_training(model, dataset)

# Note to users: EMNIST database: https://www.kaggle.com/datasets/crawford/emnist?resource=download please download the needed sets
# download mnist as well yourself as the files are too big

if __name__ == "__main__":
    # choose which example to run
    # run_png_knn()
    # run_emnist_svm()
    # run_png_cnn()
    run_emnist_cnn()
