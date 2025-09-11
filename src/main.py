import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# ==========================================================
# 1. Carga de datos
# ==========================================================
data = pd.read_csv("data/day.csv")

# ==========================================================
# 2. Preprocesamiento
# ==========================================================
# Codificación one-hot usando pandas
data = pd.get_dummies(
    data,
    columns=['season', 'yr', 'mnth', 'weekday', 'weathersit'],
    drop_first=True
)

# Normalización opcional de variables continuas
for col in ['temp', 'atemp', 'hum', 'windspeed']:
    data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

# Separar features y target
X = data.drop(['instant', 'dteday', 'casual', 'registered', 'cnt'], axis=1)
y = data['cnt']

# ==========================================================
# 3. Separación Train / Validation / Test
# ==========================================================
# 70% Train, 15% Validation, 15% Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# ==========================================================
# 4. Entrenamiento con Random Forest
# ==========================================================
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42
)
rf.fit(X_train, y_train)

# ==========================================================
# 5. Evaluación
# ==========================================================
def evaluar_modelo(model, X, y, nombre):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"{nombre}: R²={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    return r2, mae, rmse, y_pred

print("\n=== Evaluación del modelo ===")
r2_train, mae_train, rmse_train, y_pred_train = evaluar_modelo(rf, X_train, y_train, "Train")
r2_val, mae_val, rmse_val, y_pred_val = evaluar_modelo(rf, X_val, y_val, "Validation")
r2_test, mae_test, rmse_test, y_pred_test = evaluar_modelo(rf, X_test, y_test, "Test")

# ==========================================================
# 6. Guardar resultados
# ==========================================================
os.makedirs("results", exist_ok=True)

# Guardar métricas en archivo .txt
with open("results/metrics_rf.txt", "w") as f:
    f.write("=== Random Forest - Resultados ===\n")
    f.write(f"Train -> R²={r2_train:.4f}, MAE={mae_train:.2f}, RMSE={rmse_train:.2f}\n")
    f.write(f"Validation -> R²={r2_val:.4f}, MAE={mae_val:.2f}, RMSE={rmse_val:.2f}\n")
    f.write(f"Test -> R²={r2_test:.4f}, MAE={mae_test:.2f}, RMSE={rmse_test:.2f}\n")

# Scatter real vs predicho (Test)
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.xlabel("Real")
plt.ylabel("Predicho")
plt.title("Random Forest - Valores reales vs predicciones (Test)")
plt.savefig("results/predictions_rf.png")
plt.close()

# Comparación de métricas (R²)
labels = ["Train", "Validation", "Test"]
r2_scores = [r2_train, r2_val, r2_test]

plt.figure(figsize=(6, 4))
plt.bar(labels, r2_scores, color=["#4C72B0", "#55A868", "#C44E52"])
plt.ylim(0, 1)
plt.ylabel("R²")
plt.title("Comparación de R² por conjunto")
plt.savefig("results/metrics_rf.png")
plt.close()

# Importancia de características
importances = rf.feature_importances_
features = X.columns
plt.figure(figsize=(8, 6))
plt.barh(features, importances)
plt.title("Importancia de características - Random Forest")
plt.tight_layout()
plt.savefig("results/feature_importances.png")
plt.close()
