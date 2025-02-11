from fastapi import FastAPI, HTTPException, Depends, Header, Security
from fastapi.security import APIKeyHeader
import numpy as np
from tensorflow.keras.models import load_model
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Désactiver le GPU pour éviter les erreurs CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialiser FastAPI avec un titre et une description
app = FastAPI(
    title="Reconnaissance de chiffres manuscrits",
    description="API permettant de reconnaître des chiffres manuscrits à partir d'un modèle de Deep Learning.",
    version="1.0.0"
)

# Charger le modèle
try:
    model = load_model("model/model.h5", compile=False)
except Exception as e:
    raise RuntimeError(f"Impossible de charger le modèle : {e}")

load_dotenv()
API_KEY = os.getenv("API_KEY", "default_token")
api_key_header = APIKeyHeader(name="x-token", auto_error=True)

def verify_token(api_key: str = Security(api_key_header)):
    """ Vérification du token dans les headers """
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Token invalide ou manquant")
    return api_key

# Définition du format attendu pour la requête
class ImageRequest(BaseModel):
    data: list[float]

@app.post("/predict", dependencies=[Depends(verify_token)], summary="Prédire un chiffre manuscrit")
def predict(image: ImageRequest):
    """
    Analyse une image de chiffre manuscrit et renvoie la prédiction du modèle.

    - **data** : Une liste de 784 valeurs correspondant à une image 28x28 en niveaux de gris.
    - **Retourne** : Le chiffre prédit par le modèle.
    """
    if len(image.data) != 784:
        raise HTTPException(status_code=400, detail=f"Nombre de valeurs incorrect : {len(image.data)}. Attendu : 784.")

    try:
        img = np.array(image.data, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
        predicted_label = int(np.argmax(model.predict(img)))

        return {"prediction": predicted_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
