import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

def extract_fourier_features(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Fourier Transform
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Compute the magnitude spectrum
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

    # Flatten the magnitude spectrum to a 1D vector
    features = magnitude_spectrum
    # print(features.shape)
    return features

def process_images(data_dir, output_dir, num_images = 20):
    list_images = [imgdir for imgdir in os.listdir(data_dir)][:num_images]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in tqdm(list_images):
        # if file.endswith('.png'):
        image_path = os.path.join(data_dir, file)
        features = extract_fourier_features(image_path)
        plt.imshow(features, cmap='gray')
        plt.savefig(os.path.join(output_dir, file))
        
if __name__ == "__main__":
    print("Analyze images:")
    process_images("data_processed/real/", "output/real/", 20)
    process_images("data_processed/fake/", "output/fake/", 20)