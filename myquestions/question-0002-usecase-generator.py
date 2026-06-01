
import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def generar_caso_de_uso_transformar_datos_oceanicos():
    """
    Genera un caso de uso aleatorio para la función
    transformar_datos_oceanicos(df, target_col).
    """

    seed = random.randint(1, 100000)
    np.random.seed(seed)
    random.seed(seed)

    n_filas = random.randint(8, 15)

    df = pd.DataFrame({
        "velocidad_corriente": np.random.uniform(0, 20, n_filas),
        "temperatura_agua": np.random.uniform(1, 30, n_filas),
        "salinidad": np.random.uniform(30, 45, n_filas),
        "profundidad": np.random.uniform(10, 5000, n_filas),
        "oxigeno_disuelto": np.random.uniform(0.5, 12, n_filas)
    })

    # Introducir NaNs aleatorios
    n_nans = random.randint(1, max(2, n_filas // 2))
    for _ in range(n_nans):
        fila = random.randint(0, n_filas - 1)
        col = random.choice(df.columns.tolist())
        df.loc[fila, col] = np.nan

    target_col = "zona_estable"
    df[target_col] = np.random.randint(0, 2, size=n_filas)

    input_data = {
        "df": df.copy(),
        "target_col": target_col
    }

    # Output esperado
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    imputer = SimpleImputer(strategy="mean")
    X_imputada = imputer.fit_transform(X)

    X_log = np.log1p(X_imputada)

    scaler = MinMaxScaler()
    X_escalada = scaler.fit_transform(X_log)

    output_data = (X_escalada, y)

    return input_data, output_data


# # Ejemplo
# if __name__ == "__main__":
#     entrada, salida_esperada = generar_caso_de_uso_transformar_datos_oceanicos()

#     print("=== INPUT ===")
#     print("Nombre de la función esperada: transformar_datos_oceanicos")
#     print("target_col:", entrada["target_col"])
#     print(entrada["df"])

#     print("\n=== OUTPUT ESPERADO ===")
#     X_res, y_res = salida_esperada
#     print("X procesada:")
#     print(X_res)
#     print("\ny:")
#     print(y_res)