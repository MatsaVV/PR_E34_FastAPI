import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.keras
import os

# Définir le chemin de suivi MLflow
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment("model_comparison")

# Charger les données
data = pd.read_csv('data/train.csv')
labels = data['label'].values
pixels = data.drop('label', axis=1).values

# Normalisation
pixels = pixels / 255.0
pixels = pixels.reshape(-1, 28, 28, 1)

# Séparation en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(pixels, labels, test_size=0.2, random_state=42)

print(f"Taille de l'ensemble d'entraînement : {X_train.shape}")
print(f"Taille de l'ensemble de test : {X_test.shape}")

np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# Fonction pour entraîner et logger un modèle
def train_and_log_model(model, model_name, epochs=50):
    with mlflow.start_run(run_name=model_name):
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, callbacks=[early_stopping], verbose=2)

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f'Précision de {model_name}: {test_acc:.4f}')

        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("accuracy", test_acc)
        mlflow.keras.log_model(model, model_name)

        return test_acc

# Modèle KNN
from sklearn.neighbors import KNeighborsClassifier
with mlflow.start_run(run_name="KNN"):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train.reshape(-1, 28*28), y_train)
    y_pred_knn = knn.predict(X_test.reshape(-1, 28*28))
    acc_knn = accuracy_score(y_test, y_pred_knn)
    mlflow.log_metric("accuracy", acc_knn)
    print(f'Précision du KNN: {acc_knn:.4f}')

# Modèle SVM
from sklearn.svm import SVC
with mlflow.start_run(run_name="SVM"):
    svm = SVC(kernel='linear')
    svm.fit(X_train.reshape(-1, 28*28), y_train)
    y_pred_svm = svm.predict(X_test.reshape(-1, 28*28))
    acc_svm = accuracy_score(y_test, y_pred_svm)
    mlflow.log_metric("accuracy", acc_svm)
    print(f'Précision du SVM: {acc_svm:.4f}')

# Modèle MLP simple
mlp = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
acc_mlp = train_and_log_model(mlp, "MLP")

# Modèle CNN
cnn = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
acc_cnn = train_and_log_model(cnn, "CNN")

# Comparaison des modèles
models_accuracy = {
    'KNN': acc_knn,
    'SVM': acc_svm,
    'MLP': acc_mlp,
    'CNN': acc_cnn
}

best_model = max(models_accuracy, key=models_accuracy.get)
print(f'Le meilleur modèle est : {best_model} avec une précision de {models_accuracy[best_model]:.4f}')
