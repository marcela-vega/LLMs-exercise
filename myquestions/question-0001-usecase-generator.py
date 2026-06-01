
import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize

def generar_caso_de_uso_preparar_firmas_espectrales():
    """
    Genera un caso de uso aleatorio para la función
    preparar_firmas_espectrales(df, target_col, umbral_minimo).
    """

    seed = random.randint(1, 100000)
    np.random.seed(seed)
    random.seed(seed)

    n_filas = random.randint(8, 15)
    n_bandas = random.randint(4, 7)

    data = {
        f"banda_{i}": np.random.uniform(0, 1000 * (i + 1), n_filas)
        for i in range(n_bandas)
    }

    df = pd.DataFrame(data)

    # Introducir NaNs aleatorios
    n_nans = random.randint(1, max(2, n_filas // 2))
    for _ in range(n_nans):
        fila = random.randint(0, n_filas - 1)
        col = random.choice(df.columns.tolist())
        df.loc[fila, col] = np.nan

    target_col = "tipo_mineral"
    df[target_col] = np.random.randint(0, 3, size=n_filas)

    umbral_minimo = random.uniform(200, 1200)

    input_data = {
        "df": df.copy(),
        "target_col": target_col,
        "umbral_minimo": umbral_minimo
    }

    # Output esperado
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    imputer = SimpleImputer(strategy="mean")
    X_imputada = imputer.fit_transform(X)

    mask = X_imputada.sum(axis=1) >= umbral_minimo
    X_filtrada = X_imputada[mask]
    y_filtrado = y[mask]

    X_normalizada = normalize(X_filtrada, norm="l2")

    output_data = (X_normalizada, y_filtrado)

    return input_data, output_data


# # Ejemplo
# if __name__ == "__main__":
#     entrada, salida_esperada = generar_caso_de_uso_preparar_firmas_espectrales()

#     print("=== INPUT ===")
#     print("Nombre de la función esperada: preparar_firmas_espectrales")
#     print("target_col:", entrada["target_col"])
#     print("umbral_minimo:", entrada["umbral_minimo"])
#     print(entrada["df"])

#     print("\n=== OUTPUT ESPERADO ===")
#     X_res, y_res = salida_esperada
#     print("X procesada:")
#     print(X_res)
#     print("\ny:")
#     print(y_res)