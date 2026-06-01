import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def detectar_fraude_transacciones(ruta_csv: str, test_size: float, random_state: int) -> tuple:
    # 1. Cargar dataset
    df = pd.read_csv(ruta_csv)

    # 2. Limpiar datos: el generador espera que se eliminen los nulos
    df = df.dropna()

    # 3. Separar variables independientes y objetivo
    X = df.drop(columns=["fraude"])
    y = df["fraude"]

    # 4. Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    # 5. Entrenar modelo
    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
    )
    modelo.fit(X_train, y_train)

    # 6. Predecir sobre el conjunto de prueba
    predicciones = modelo.predict(X_test)

    # 7. Calcular accuracy
    accuracy = accuracy_score(y_test, predicciones)

    # El generador devuelve predicciones como lista y accuracy como float
    return predicciones.tolist(), float(accuracy)