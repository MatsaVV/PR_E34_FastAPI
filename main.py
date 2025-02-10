from fastapi import FastAPI, HTTPException
import numpy as np
from tensorflow.keras.models import load_model
from pydantic import BaseModel
import os

# Désactiver le GPU pour éviter les erreurs CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialiser FastAPI
app = FastAPI()

# Charger le modèle avec gestion d'erreur
try:
    model = load_model("model/model.h5", compile=False)
except Exception as e:
    raise RuntimeError(f"Impossible de charger le modèle : {e}")

# Définition du format attendu pour la requête
class ImageRequest(BaseModel):
    data: list[float]

@app.post("/predict")
def predict(image: ImageRequest):
    if len(image.data) != 784:
        raise HTTPException(status_code=400, detail=f"Nombre de valeurs incorrect : {len(image.data)}. Attendu : 784.")

    try:
        # Conversion et normalisation
        img = np.array(image.data, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
        # Prédiction
        predicted_label = int(np.argmax(model.predict(img)))

        return {"prediction": predicted_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
