import pandas as pd


def matriz_transicion_estados(df):
    df_work = df.copy()

    # Ordenar eventos dentro de cada proceso
    df_work = df_work.sort_values(["id_proceso", "timestamp"])

    # Obtener el siguiente estado dentro del mismo proceso
    df_work["siguiente_estado"] = df_work.groupby("id_proceso")["estado"].shift(-1)

    # Quitar filas que no tienen transición siguiente
    transiciones = df_work.dropna(subset=["siguiente_estado"])

    # Estados ordenados alfabéticamente
    estados = sorted(df_work["estado"].dropna().unique())

    # Si no hay transiciones, devolver matriz cuadrada en ceros
    if transiciones.empty:
        return pd.DataFrame(0.0, index=estados, columns=estados)

    # Contar transiciones estado -> siguiente_estado
    matriz = pd.crosstab(
        transiciones["estado"],
        transiciones["siguiente_estado"],
        normalize="index"
    )

    # Asegurar que estén todos los estados como filas y columnas
    matriz = matriz.reindex(index=estados, columns=estados, fill_value=0.0)

    # Asegurar tipo float
    matriz = matriz.astype(float)

    return matriz