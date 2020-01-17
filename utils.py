import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


# function to load list of images from a specified directory
def load_images(folder, size):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 0)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img_flat = img.flatten()
        img_norm = img_flat/np.amax(img_flat)
        if img is not None:
            images.append(img_norm)
    return images


# function to load all data in train/test split for both face and nonface categories
def load_data(size):
    face_train_path = os.path.join("Extracted_Faces", "Train")
    face_test_path = os.path.join("Extracted_Faces", "Test")
    non_train_path = os.path.join("Extracted_Non", "Train")
    non_test_path = os.path.join("Extracted_Non", "Test")
    face_train = load_images(face_train_path, size)
    face_test = load_images(face_test_path, size)
    non_train = load_images(non_train_path, size)
    non_test = load_images(non_test_path, size)
    return face_train, face_test, non_train, non_test


def roc_plot(p1f, p1n):
    false_pos = np.zeros(1000)
    false_neg = np.zeros(1000)
    true_pos = np.zeros(1000)
    thresh = np.flip(np.arange(1000))/998
    for i in range(0, 1000):
        false_pos[i] = np.sum(p1n >= thresh[i]) / 100
        false_neg[i] = np.sum([p1f <= thresh[i]]) / 100
        true_pos[i] = 1 - false_neg[i]
    plt.figure(1)
    plt.plot(false_pos, true_pos)
    plt.title('ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()