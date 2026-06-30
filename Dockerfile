# 1. Image Python légère
FROM python:3.11-slim

# 2. Répertoire de travail
WORKDIR /app

# 3. Copie des dépendances
COPY requirements.txt .

# 4. Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copie du code source (app.py, train.py, model.joblib)
COPY . .

# 6. Exposition du port 8000
EXPOSE 8000

# 7. Lancement de l'application FastAPI avec Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
