import numpy as np
from PIL import Image
from lib.l1normImg import l1norm

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

    for i, (testImg, testLabel) in enumerate(data): # for each piece of data
        distances = []
        for j, (trainImg, trainLabel) in enumerate(data): # Compare against other data
            if i == j: # Except itself
                continue
            dist = l1norm(testImg,trainImg) # Find the distance
            distances.append((dist,trainLabel)) # and append the distance with their respective letter
        distances.sort(key=lambda x: x[0]) # Sort it
        nearest = [label for _, label in distances[:K]]

        pred = max(set(nearest), key=nearest.count) # first past the post voting
        if pred == testLabel:
            correct += 1
    
    accuracy = correct/n
    return accuracy
