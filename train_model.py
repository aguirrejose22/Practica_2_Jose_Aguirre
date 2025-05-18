# train_model.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import shutil  # Para eliminar carpetas existentes
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ================================
# CONFIGURACIÓN DE MLFLOW
# ================================
mlflow.set_tracking_uri("http://localhost:9090")  # Servidor de seguimiento local
mlflow.set_experiment("Prevencion_Accidentes_Maquinaria")  # Nombre del experimento

# ================================
# SIMULACIÓN DE DATOS HISTÓRICOS
# ================================
anios = np.arange(2020, 2025)  # Años disponibles con datos reales
accidentes = np.array([480, 420, 370, 320, 280])  # Cantidad de accidentes simulada

# Se agrega una columna binaria que indica desde qué año se implementó seguridad
datos = pd.DataFrame({
    "anio": anios,
    "accidentes": accidentes,
    "mejora_seguridad": [0, 1, 1, 1, 1]
})

# ================================
# PREPARACIÓN DE VARIABLES
# ================================
X = datos[["anio", "mejora_seguridad"]]  # Variables independientes
y = datos["accidentes"]  # Variable objetivo

# ================================
# ENTRENAMIENTO DEL MODELO
# ================================
modelo = LinearRegression()

with mlflow.start_run():
    modelo.fit(X, y)

    # ================================
    # GUARDAR MODELO EN MLFLOW
    # ================================
    mlflow.sklearn.log_model(modelo, "modelo_prevencion")

    # ================================
    # GUARDAR MODELO LOCALMENTE PARA FLASK
    # ================================
    ruta_modelo = "modelo/logistic_model"
    if os.path.exists(ruta_modelo):
        shutil.rmtree(ruta_modelo)  # Eliminar si ya existe
    os.makedirs(ruta_modelo, exist_ok=True)
    mlflow.sklearn.save_model(modelo, ruta_modelo)

    # ================================
    # GENERAR PROYECCIÓN 2025-2029
    # ================================
    pred_anios = np.arange(2025, 2030)
    X_pred = pd.DataFrame({
        "anio": pred_anios,
        "mejora_seguridad": [1] * len(pred_anios)
    })
    pred_acc = modelo.predict(X_pred)

    # ================================
    # GRAFICAR RESULTADO
    # ================================
    plt.figure(figsize=(10, 5))
    plt.plot(anios, accidentes, marker='o', label='Histórico')
    plt.plot(pred_anios, pred_acc, marker='x', linestyle='--', label='Proyección')
    plt.title("Proyección de Disminución de Accidentes por Seguridad de Maquinaria")
    plt.xlabel("Año")
    plt.ylabel("Cantidad de Accidentes")
    plt.grid(True)
    plt.legend()

    # Crear carpeta 'static' si no existe
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/curva_proyeccion.png")

    print("✅ Modelo entrenado, guardado y gráfico generado.")
