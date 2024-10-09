from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
import tqdm
import os
from process_data import extract_fourier_features
from tqdm import tqdm
from sklearn.decomposition import PCA
from utils import save_model
# from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from visualize import visualize_pca
from utils import save_data
from sklearn.metrics import f1_score
# list_images = os.listdir('data')

fake_path = "dataset/fake"
real_path = "dataset/real"

list_fake_data =  [name for name in os.listdir(fake_path)]
list_real_data =  [name for name in os.listdir(real_path)]

Xs = []
ys = []
for i in tqdm(range(len(list_fake_data))):
  Xs.append(extract_fourier_features(os.path.join(fake_path, list_fake_data[i])))
  ys.append(0)

for i in tqdm(range(len(list_real_data))):
  Xs.append(extract_fourier_features(os.path.join(real_path, list_real_data[i])))
  ys.append(1)

X_train, X_val_test, y_train, y_val_test = train_test_split(Xs, ys, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

print("Train size:", len(y_train))
print("Train real and fake number:", y_train.count(0), y_train.count(1))
print("Val size:", len(y_val))
print("Val real and fake number:", y_val.count(0), y_val.count(1))
print("Test size:", len(y_test))
print("Test real and fake number:", y_test.count(0), y_test.count(1))

pca = PCA(n_components=150)
pca.fit(X_train)
X_train = pca.transform(X_train)
print(X_train.shape)
visualize_pca(150, pca)

# svm = svm.SVC(kernel='rbf', verbose=True)
# svm.fit(X_train, y_train)

model = LogisticRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
accuracy = accuracy_score(y_train, y_train_pred)
print("Train Accuracy:", accuracy)
conf_matrix = confusion_matrix(y_train, y_train_pred)
# print("Train Confusion Matrix:", conf_matrix)
# print("Train Precision:", conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0]))
# print("Train Recall:", conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1]))
print("Train F1 Score:", f1_score(y_train, y_train_pred))
print("Train FAR:", conf_matrix[1][0] / (conf_matrix[1][0] + conf_matrix[0][0]))
print("Train FRR:", conf_matrix[0][1] / (conf_matrix[0][1] + conf_matrix[1][1]))
print("Train ERR:", (conf_matrix[1][0] + conf_matrix[0][1]) / (conf_matrix[0][0] + conf_matrix[1][1]))

X_val = pca.transform(X_val)
y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print("Val Accuracy:", accuracy)
conf_matrix = confusion_matrix(y_val, y_val_pred)
# print("Val Confusion Matrix:", conf_matrix)
# print("Val Precision:", conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0]))
# print("Val Recall:", conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1]))
print("Val F1 Score:", f1_score(y_val, y_val_pred))
print("Val FAR:", conf_matrix[1][0] / (conf_matrix[1][0] + conf_matrix[0][0]))
print("Val FRR:", conf_matrix[0][1] / (conf_matrix[0][1] + conf_matrix[1][1]))
print("Val ERR:", (conf_matrix[1][0] + conf_matrix[0][1]) / (conf_matrix[0][0] + conf_matrix[1][1]))

X_test = pca.transform(X_test)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
conf_matrix = confusion_matrix(y_test, y_pred)
# print("Test Confusion Matrix:", conf_matrix)
# print("Test Precision:", conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0]))
# print("Test Recall:", conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1]))
print("Test F1 Score:", f1_score(y_test, y_pred))
print("Test FAR:", conf_matrix[1][0] / (conf_matrix[1][0] + conf_matrix[0][0]))
print("Test FRR:", conf_matrix[0][1] / (conf_matrix[0][1] + conf_matrix[1][1]))
print("Test ERR:", (conf_matrix[1][0] + conf_matrix[0][1]) / (conf_matrix[0][0] + conf_matrix[1][1]))

save_model(
    cls_model_path='models/logistic_model.pkl',
    pca_model_path='models/pca_model.pkl',
    model=model,
    pca=pca
)