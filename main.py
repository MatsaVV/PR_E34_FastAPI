from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
import numpy as np
from tensorflow.keras.models import load_model
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Charger les variables d'environnement
load_dotenv()
API_KEY = os.getenv("API_KEY", "default_token")

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

# Connexion à la base de données MySQL sur Azure
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Erreur : La variable d'environnement DATABASE_URL n'est pas définie !")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modèle pour stocker les feedbacks
class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    image_data = Column(String(2048), nullable=False)  # Ajoute une longueur explicite pour VARCHAR
    prediction = Column(Integer, nullable=False)
    correct = Column(Integer, nullable=False)  # 1 = Correct, 0 = Incorrect
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Définition du format attendu pour la requête
class ImageRequest(BaseModel):
    data: list[float]

api_key_header = APIKeyHeader(name="x-token", auto_error=True)

def verify_token(api_key: str = Security(api_key_header)):
    """ Vérification du token dans les headers """
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Token invalide ou manquant")
    return api_key

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

# Route pour stocker le feedback
class FeedbackRequest(BaseModel):
    image_data: str
    prediction: int
    correct: int

@app.post("/feedback")
def store_feedback(feedback: FeedbackRequest):
    """ Stocke les retours des utilisateurs sur les prédictions """
    db = SessionLocal()
    new_feedback = Feedback(image_data=feedback.image_data, prediction=feedback.prediction, correct=feedback.correct)
    db.add(new_feedback)
    db.commit()
    db.refresh(new_feedback)
    db.close()
    return {"message": "Feedback enregistré avec succès !"}

# Route pour récupérer les stats des feedbacks
from sqlalchemy.sql import func

@app.get("/feedback_stats")
def get_feedback_stats():
    """ Récupère les statistiques des feedbacks """
    db = SessionLocal()
    correct_counts = db.query(Feedback.prediction, func.count()).filter(Feedback.correct == 1).group_by(Feedback.prediction).all()
    incorrect_counts = db.query(Feedback.prediction, func.count()).filter(Feedback.correct == 0).group_by(Feedback.prediction).all()
    db.close()
    return {
        "correct_counts": [{"prediction": c[0], "count": c[1]} for c in correct_counts],
        "incorrect_counts": [{"prediction": i[0], "count": i[1]} for i in incorrect_counts]
    }
