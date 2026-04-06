import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def generar_caso_de_uso_preprocesar_datos_clinicos():
    """
    Genera un caso de uso aleatorio para la función
    preprocesar_datos_clinicos(df, target_col).
    """

    seed = random.randint(1, 100000)
    np.random.seed(seed)
    random.seed(seed)

    n_filas = random.randint(8, 15)

    df = pd.DataFrame({
        "edad": np.random.randint(20, 80, size=n_filas).astype(float),
        "proteina_c_reactiva": np.random.uniform(0.1, 15.0, size=n_filas),
        "presion_arterial": np.random.uniform(90, 180, size=n_filas),
        "colesterol": np.random.uniform(120, 320, size=n_filas),
        "sexo": np.random.choice(["F", "M"], size=n_filas),
        "grupo_sanguineo": np.random.choice(["A", "B", "AB", "O"], size=n_filas),
        "fumador": np.random.choice(["si", "no"], size=n_filas)
    })

    # Introducir NaNs en columnas numéricas
    columnas_numericas = ["edad", "proteina_c_reactiva", "presion_arterial", "colesterol"]
    n_nans_num = random.randint(1, max(2, n_filas // 2))
    for _ in range(n_nans_num):
        fila = random.randint(0, n_filas - 1)
        col = random.choice(columnas_numericas)
        df.loc[fila, col] = np.nan

    # Introducir NaNs en columnas categóricas
    columnas_categoricas = ["sexo", "grupo_sanguineo", "fumador"]
    n_nans_cat = random.randint(1, max(2, n_filas // 3))
    for _ in range(n_nans_cat):
        fila = random.randint(0, n_filas - 1)
        col = random.choice(columnas_categoricas)
        df.loc[fila, col] = np.nan

    target_col = "riesgo_inflamatorio"
    df[target_col] = np.random.randint(0, 2, size=n_filas)

    input_data = {
        "df": df.copy(),
        "target_col": target_col
    }

    # OUTPUT ESPERADO
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    X_num = X.select_dtypes(include=[np.number])
    X_cat = X.select_dtypes(exclude=[np.number])

    imputer_num = SimpleImputer(strategy="mean")
    X_num_imputada = imputer_num.fit_transform(X_num)

    scaler = StandardScaler()
    X_num_escalada = scaler.fit_transform(X_num_imputada)

    imputer_cat = SimpleImputer(strategy="most_frequent")
    X_cat_imputada = imputer_cat.fit_transform(X_cat)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_codificada = encoder.fit_transform(X_cat_imputada)

    X_final = np.hstack([X_num_escalada, X_cat_codificada])

    output_data = (X_final, y)

    return input_data, output_data


# Ejemplo
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_preprocesar_datos_clinicos()

    print("=== INPUT ===")
    print("Nombre de la función esperada: preprocesar_datos_clinicos")
    print("target_col:", entrada["target_col"])
    print(entrada["df"])

    print("\n=== OUTPUT ESPERADO ===")
    X_res, y_res = salida_esperada
    print("Shape de X procesada:", X_res.shape)
    print("y:", y_res)
