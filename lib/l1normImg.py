import numpy as np

def l1norm(image1, image2):
    # find image dimensions of image1 and image 2
    img1 = np.array(image1)
    img2 = np.array(image2)
    # if dim(image1) != dim(image2): return -1 (dimensions have to be same)
    if img1.shape != img2.shape: return -1
    # Turn it into a 2D array of 1's and 0's (Either filled in or not)
    binImg1 = (img1 > 128).astype(int)
    binImg2 = (img2 > 128).astype(int)
    # take XOR of the two 2D arrays
    xor = np.bitwise_xor(binImg1, binImg2)
    # Sum it
    # return answer
    return np.sum(xor)