import numpy as np
from PIL import Image
from lib.l1normImg import l1norm
import os

def extract_label(filename):
    """
    filename: name of the file
    this function grabs numbers before first underscroll, returns the ascii output
    """
    base = os.path.basename(filename)
    number = base.split('_')[0]
    return chr(int(number))

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
            dist = l1norm(testVec,trainVec) # Find the distance
            distances.append((dist,trainLabel)) # and append the distance with their respective letter
        distances.sort(key=lambda x: x[0]) # Sort it
        nearest = [label for _, label in distances[:K]]

        pred = max(set(nearest), key=nearest.count) # first past the post voting
        if pred == testLabel:
            correct += 1
    
    accuracy = correct/n
    return accuracy