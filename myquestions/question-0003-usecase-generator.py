import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_filtrar_y_escalar_datos_solares():
    """
    Genera un caso de uso aleatorio para la función
    filtrar_y_escalar_datos_solares(df, target_col, umbral_varianza).
    """

    seed = random.randint(1, 100000)
    np.random.seed(seed)
    random.seed(seed)

    n_filas = random.randint(8, 15)

    df = pd.DataFrame({
        "radiacion_solar": np.random.uniform(200, 1200, n_filas),
        "temperatura_panel": np.random.uniform(15, 80, n_filas),
        "voltaje": np.random.uniform(10, 60, n_filas),
        "corriente": np.random.uniform(1, 15, n_filas),
        "angulo_inclinacion": np.random.uniform(5, 45, n_filas),
        "sensor_estable": np.random.uniform(9.9, 10.1, n_filas)
    })

    # Introducir NaNs aleatorios
    n_nans = random.randint(1, max(2, n_filas // 2))
    for _ in range(n_nans):
        fila = random.randint(0, n_filas - 1)
        col = random.choice(df.columns.tolist())
        df.loc[fila, col] = np.nan

    target_col = "instalacion_eficiente"
    df[target_col] = np.random.randint(0, 2, size=n_filas)

    umbral_varianza = random.uniform(0.001, 1.0)

    input_data = {
        "df": df.copy(),
        "target_col": target_col,
        "umbral_varianza": umbral_varianza
    }

    # Output esperado
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    imputer = SimpleImputer(strategy="mean")
    X_imputada = imputer.fit_transform(X)

    varianzas = np.var(X_imputada, axis=0)
    X_filtrada = X_imputada[:, varianzas > umbral_varianza]

    scaler = StandardScaler()
    X_escalada = scaler.fit_transform(X_filtrada)

    output_data = (X_escalada, y)

    return input_data, output_data


# Ejemplo
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_filtrar_y_escalar_datos_solares()

    print("=== INPUT ===")
    print("Nombre de la función esperada: filtrar_y_escalar_datos_solares")
    print("target_col:", entrada["target_col"])
    print("umbral_varianza:", entrada["umbral_varianza"])
    print(entrada["df"])

    print("\n=== OUTPUT ESPERADO ===")
    X_res, y_res = salida_esperada
    print("X procesada:")
    print(X_res)
    print("\ny:")
    print(y_res)