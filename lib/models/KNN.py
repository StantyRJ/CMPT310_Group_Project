import numpy as np
from .l1normImg import l1norm, l1normVec
from sklearn.neighbors import KDTree
import os
import multiprocessing as mp
from typing import Sequence, Any, Tuple
import numpy as np
from .base import Classifier


def extract_label(filename):
    """
    filename: name of the file
    this function grabs numbers before first underscroll, returns the ascii output
    """
    base = os.path.basename(filename)
    number = base.split('_')[0]
    return chr(int(number))


# For the love of all that's good, please optimize runtime of this
def KNN(data, K):
    """
    data: list of tuples (image,label)
    K: Number of neighbors to consider
    """
    n = len(data)
    if n == 0:
        print("Empty dataset, please check input")
        return -1
    if K < 1:
        print("Invalid K given, check input")
        return K
    correct = 0 

    for i, (testVec, testFilename) in enumerate(data): # for each piece of data
        distances = []
        testLabel = extract_label(testFilename)
        for j, (trainVec, trainFilename) in enumerate(data): # Compare against other data
            if i == j: # Except itself
                continue
            trainLabel = extract_label(trainFilename)
            dist = l1normVec(testVec,trainVec) # Find the distance
            distances.append((dist,trainLabel)) # and append the distance with their respective letter
        distances.sort(key=lambda x: x[0]) # Sort it
        nearest = [label for _, label in distances[:K]]

        pred = max(set(nearest), key=nearest.count) # first past the post voting
        if pred == testLabel:
            correct += 1
    
    accuracy = correct/n
    return accuracy

def findAccuracy(args):
    print(args)
    K, labels, idx = args
    n = len(labels)
    correct = 0
    for i in range(n):
        neighbor_labels=labels[idx[i,:K]]
        uniq, counts = np.unique(neighbor_labels, return_counts=True)
        pred = uniq[np.argmax(counts)]
        if pred == labels[i]:
            correct += 1
    return K, correct/n


def KNNOpt(data, Kstart, Kend, numJob=None):
    # does KNN with K in range(Kstart,Kend+1)
    if numJob is None:
        numJob = mp.cpu_count()

    # Prepare the data
    vecs = np.array([vec for vec, _ in data]) # we don't want the filename, just the vectorized stuff
    labels = np.array([extract_label(fname) for _, fname in data])

    # let's grow a tree
    tree = KDTree(vecs, metric="manhattan")
    print("Tree made")
    maxK = Kend

    # Get closest neighbors
    _, idx = tree.query(vecs, k=maxK+1)
    print("query made")
    # remove thyself
    idx = idx[:, 1:]

    Ks = list(range(Kstart, Kend+1))
    tasks = [(K,labels,idx) for K in Ks]
    print("Ks Made")

    with mp.Pool(numJob) as pool:
        results = pool.map(findAccuracy, tasks)

    result_dict = dict(results)

    # --- Find best K and best accuracy ---
    best_K = max(result_dict, key=result_dict.get)
    best_acc = result_dict[best_K]

    return result_dict, best_K, best_acc

class KNNClassifier(Classifier):
    """Wrapper for KNN-based classification.

    Stores training data on fit; uses KNN to predict.
    """

    def __init__(self, K: int = 3):
        self.K = K
        self.X_train = None
        self.y_train = None

    def fit(self, X: Sequence, y: Sequence) -> None:
        # KNN is typically a lazy learner — storing training data is enough
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def predict(self, X: Sequence) -> Sequence:
        X = np.asarray(X)
        results = []
        # Use the KNN function provided in lib.KNN; if unavailable, fallback to KNNOpt behavior
        for i in range(len(X)):
            # KNN in your lib might accept arrays in various shapes: adapt as needed
            pred = KNN(list(zip(self.X_train, self.y_train)), self.K, query=X[i]) if hasattr(KNN, "__call__") else None
            # Many KNN helpers in student libs take the whole dataset and K and return accuracy —
            # Here we try a simpler fallback: brute-force majority voting
            if pred is None:
                # brute force
                dists = np.linalg.norm(self.X_train - X[i], axis=1)
                nearest = np.argsort(dists)[: self.K]
                votes = self.y_train[nearest]
                # choose most common
                vals, counts = np.unique(votes, return_counts=True)
                results.append(vals[np.argmax(counts)])
            else:
                results.append(pred)
        return results

    def optimize_K(self, data: Sequence[Tuple[Any, Any]], k_min: int = 1, k_max: int = 25):
        """Call your KNNOpt helper if you want to sweep K.

        Accepts same format as original script's data: list of (vector, filename/label) tuples.
        """
        return KNNOpt(data, k_min, k_max)
