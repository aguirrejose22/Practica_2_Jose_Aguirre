from flask import Flask, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def mostrar_resultado():
    # Datos de proyección simulados
    anios = list(range(2025, 2030))
    predicciones = [260, 230, 210, 190, 160]

    # Cargar el dataset con descripciones de accidentes
    df = pd.read_csv("data/osha_riesgo.csv")

    # Obtener las 10 descripciones de riesgo más frecuentes
    top_accidentes = df["descripcion_riesgo"].value_counts().head(10).reset_index()
    top_accidentes.columns = ["descripcion", "frecuencia"]
    top_list = list(zip(top_accidentes["descripcion"], top_accidentes["frecuencia"]))

    return render_template('result.html', predicciones=zip(anios, predicciones), top=top_list)

if __name__ == '__main__':
    app.run(debug=True)
