import pytest
from fastapi.testclient import TestClient
from main import app
import numpy as np

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 404  # Pas de route définie pour "/", c'est normal

# Test d'une prédiction avec une entrée correcte
def test_predict_valid():
    test_data = {
        "data": np.random.randint(0, 256, 784).tolist()  # Génère une image aléatoire 28x28
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "prediction" in response.json()  # Vérifie que la réponse contient bien la clé 'prediction'

# Test d'une prédiction avec une entrée invalide (taille incorrecte)
def test_predict_invalid_size():
    test_data = {
        "data": [0] * 100  # Mauvaise taille, seulement 100 pixels au lieu de 784
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 400  # Doit retourner une erreur HTTP 400
    assert "detail" in response.json()

# Test d'une prédiction avec des données non valides
def test_predict_invalid_data():
    test_data = {
        "data": ["invalid"] * 784  # Données invalides (strings au lieu de nombres)
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # FastAPI renvoie une erreur 422 pour format incorrect
