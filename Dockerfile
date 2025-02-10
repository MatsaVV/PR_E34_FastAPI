# Utiliser une image Python officielle légère
FROM python:3.10

# Définir le répertoire de travail
WORKDIR /app

# Copier tous les fichiers de l'API dans le conteneur
COPY . /app

# Installer Pipenv et les dépendances
RUN pip install --no-cache-dir pipenv \
    && pipenv install --deploy --ignore-pipfile

# Exposer le port 8000 pour l'API
EXPOSE 8000

# Lancer l'API avec Uvicorn
CMD ["pipenv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
