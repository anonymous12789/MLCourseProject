import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

def extract_fourier_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    features = magnitude_spectrum.flatten()
    return features

def process_images(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in tqdm(os.listdir(data_dir)):
        if file.endswith('.png'):
            image_path = os.path.join(data_dir, file)
            features = extract_fourier_features(image_path)
            output_path = os.path.join(output_dir, file.replace('.png', '.npy'))
            np.save(output_path, features)