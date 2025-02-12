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
from sqlalchemy.sql import func

# Charger les variables d'environnement
load_dotenv()
API_KEY = os.getenv("API_KEY", "default_token")

# Désactiver le GPU pour éviter les erreurs CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI(title="Reconnaissance de chiffres manuscrits", version="1.0.0")

# Charger le modèle
try:
    model = load_model("model/model.h5", compile=False)
except Exception as e:
    raise RuntimeError(f"Impossible de charger le modèle : {e}")

# Connexion à la base de données MySQL
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Erreur : La variable DATABASE_URL n'est pas définie !")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modèle pour stocker les feedbacks
class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    image_data = Column(String(2048), nullable=False)
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

@app.post("/predict", dependencies=[Depends(verify_token)])
def predict(image: ImageRequest):
    """ Prédire un chiffre manuscrit """
    if len(image.data) != 784:
        raise HTTPException(status_code=400, detail="Nombre de valeurs incorrect")

    img = np.array(image.data, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
    predicted_label = int(np.argmax(model.predict(img)))

    return {"prediction": predicted_label}

# Définition du format attendu pour stocker un feedback
class FeedbackRequest(BaseModel):
    image_data: str
    prediction: int
    correct: int

@app.post("/feedback")
def store_feedback(feedback: FeedbackRequest):
    """ Stocker un feedback utilisateur """
    db = SessionLocal()
    new_feedback = Feedback(image_data=feedback.image_data, prediction=feedback.prediction, correct=feedback.correct)
    db.add(new_feedback)
    db.commit()
    db.close()
    return {"message": "Feedback enregistré avec succès !"}

@app.get("/feedback", dependencies=[Depends(verify_token)])
def get_feedback():
    """ Récupérer tous les feedbacks stockés dans la base de données """
    db = SessionLocal()
    feedbacks = db.query(Feedback).all()
    db.close()

    return [{"id": f.id, "image_data": f.image_data, "prediction": f.prediction, "correct": f.correct, "timestamp": f.timestamp} for f in feedbacks]

@app.get("/feedback_stats")
def get_feedback_stats():
    """ Récupérer les statistiques des feedbacks """
    db = SessionLocal()
    correct_counts = db.query(Feedback.prediction, func.count()).filter(Feedback.correct == 1).group_by(Feedback.prediction).all()
    incorrect_counts = db.query(Feedback.prediction, func.count()).filter(Feedback.correct == 0).group_by(Feedback.prediction).all()
    db.close()
    return {
        "correct_counts": [{"prediction": c[0], "count": c[1]} for c in correct_counts],
        "incorrect_counts": [{"prediction": i[0], "count": i[1]} for i in incorrect_counts]
    }
