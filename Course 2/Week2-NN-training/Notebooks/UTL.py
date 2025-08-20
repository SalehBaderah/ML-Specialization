#Generating data for week2 assignment

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from skimage.transform import resize
from sklearn.model_selection import train_test_split


def load_data(num_samples=5000):
    # Load MNIST (28x28 images, labels 0–9)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # Resize images from 28x28 -> 20x20 and flatten
    X_resized = np.array([
        resize(img, (20,20), anti_aliasing=True).flatten()
        for img in X_all
    ])

    # Normalize to [0,1]
    X_resized = X_resized.astype(np.float32)

    # Pick 5000 balanced samples
    X_small, _, y_small, _ = train_test_split(
        X_resized, y_all, train_size=num_samples,
        stratify=y_all, random_state=42
    )

    # Reshape labels to (5000,1)
    y_small = y_small.reshape(-1,1)

    return X_small, y_small



