import pytest
from fastapi.testclient import TestClient
from main import app
import numpy as np
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()
API_KEY = os.getenv("API_KEY", "default_token")
INVALID_TOKEN = "wrong_token"

# Création du client de test
@pytest.fixture
def client():
    return TestClient(app)

# Test d'un accès à une route inexistante
def test_root(client):
    response = client.get("/")
    assert response.status_code == 404

# Test d'une prédiction avec une entrée invalide (taille incorrecte)
def test_predict_invalid_size(client):
    test_data = {
        "data": [0] * 100  # Mauvaise taille (doit être 784)
    }
    response = client.post(
        "/predict",
        json=test_data,
        headers={"x-token": API_KEY}
    )
    assert response.status_code == 400
    assert "detail" in response.json()

# Test d'une prédiction avec des données non valides
def test_predict_invalid_data(client):
    test_data = {
        "data": ["invalid"] * 784  # Valeurs non numériques
    }
    response = client.post(
        "/predict",
        json=test_data,
        headers={"x-token": API_KEY}
    )
    assert response.status_code == 422  # Erreur de validation Pydantic

# Test avec un token incorrect
def test_predict_invalid_token(client):
    test_data = {
        "data": np.random.randint(0, 256, 784).tolist()
    }
    response = client.post(
        "/predict",
        json=test_data,
        headers={"x-token": INVALID_TOKEN}  # Token invalide
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Token invalide ou manquant"

# Test d'une requête vide
def test_predict_empty_request(client):
    response = client.post(
        "/predict",
        json={},
        headers={"x-token": API_KEY}
    )
    assert response.status_code == 422  # Erreur de validation
