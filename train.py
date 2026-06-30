"""
train.py
Entraîne un modèle de régression sur le dataset California Housing
et sauvegarde le modèle entraîné dans model.joblib
"""

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Chargement du dataset California Housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Séparation variables explicatives / variable cible
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# 3. Jeu d'entraînement / jeu de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Entraînement du modèle (Random Forest)
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train, y_train)

# Évaluation rapide
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")

# 5. Sauvegarde du modèle entraîné
joblib.dump(model, "model.joblib")
print("Modèle sauvegardé dans model.joblib")
