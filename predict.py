from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import pickle
import numpy as np
from process_data import extract_fourier_features

def predict(image_path, pca_path, model_path):
    # Load the image
    features = extract_fourier_features(image_path)
    features = np.array(features).reshape(1, -1)
    print(features.shape)
    # Load PCA model
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    features = pca.transform(features)
    # Load SVM model
    with open(model_path, 'rb') as f:
        svm = pickle.load(f)
    # Predict
    prediction = svm.predict(features)
    return prediction

if __name__ == "__main__":
    image_path = "test.jpg"
    pca_path = "models/pca_model.pkl"
    model_path = "models/cls_model.pkl"
    prediction = predict(image_path, pca_path, model_path)
    print("Prediction:", prediction)