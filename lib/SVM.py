import numpy as np
from PIL import Image
import os

"""
This program will perform SVM with SGD
SVM(letter) in the form that SVM(A) generates a line to seperate A from not A
Note: Please FLATTEN your images into a 1D array before entering (greyshift so only 1's and 0's)
"""

def train_svm(X, y, w_pix=None, C=1.0, lr=1e-3, epochs=100):
    """
    Train SVM using SGD
    Please make sure that your X,y,and w_pix are numpy arrays, not normal arrays
    X: (n_samples, n_features)
    y: (n_samples,) labels in {-1,+1} (-1 is not letter, +1 is letter)
    w_pix: weights for pixels (n_features,)
    C: regularization parameter
    lr: learning rate
    epochs: number of passes
    """
    n_samples, n_features = X.shape
    if w_pix is None:
        w_pix = np.ones(n_features)
    
    Xw = X * w_pix # Pixel weighting

    beta = np.zeros(n_features)
    b = 0.0

    for epoch in range(epochs): # epoch passes
        for i in range(n_samples): # for each sample
            xi, yi = Xw[i], y[i]
            condition = yi * (np.dot(beta, xi) + b) >= 1

            # Compute subgradient and update
            if condition:
                beta -= lr * (beta)
            else:
                beta -= lr * (beta - C * yi * xi)
                b += lr * (C * yi)
    return beta, b, w_pix


# Uses the model to try and predict
def predict_svm(X, beta, b, w_pix=None):
    """
    X: (n_samples, n_features)
    y: (n_samples,) (holds label for each x in X)
    beta: weight vector for each pixel
    b: bias/intercept
    """
    if w_pix is None:
        w_pix = np.ones(X.shape[1])
    Xw = X * w_pix
    scores = np.dot(Xw, beta) + b
    return scores


# Trains the model for all 62 characters
def SVM_multiclass(X, y_labels, w_pix, C=1.0, lr=1e-3,epochs=100):
    """
    For all 62 characters (A-Z,a-z,0-9), run SVM
    Returns a dictionary of models: {class_label: (beta,b,w_pix)}
    """

    classes = sorted(list(set(y_labels)))
    models = {}

    for cls in classes:
        print(f"Training SVM for class {cls}")
        y_binary = np.where(y_labels == cls, 1, -1)
        beta, b, w_used = train_svm(X, y_binary, w_pix=w_pix, C=C, lr=lr, epochs=epochs)
        models[cls] = (beta, b, w_used)
    return models

# Predicts for X using models from SVM_mutliclass
def predict_multiclass(X, models):
    """
    X: (n_samples, n_features)
    models: dictionary (see above)
    Returns predicted labels
    """

    scores = []
    for cls, (beta, b, w_pix) in models.items():
        score = predict_svm(X,beta,b,w_pix)
        scores.append(score)
    scores = np.array(scores)
    pred_indices = np.argmax(scores, axis=0)
    return np.array(list(models.keys()))[pred_indices]