import pandas as pd


def segmentar_por_cuantiles(df, col_nombre):
    return pd.qcut(
        df[col_nombre],
        q=4,
        labels=["Bajo", "Medio", "Alto", "Top"]
    )