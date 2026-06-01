import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def evaluar_deriva_mensual(df, target_col, fecha_col):
    df_work = df.copy()

    df_work[fecha_col] = pd.to_datetime(df_work[fecha_col])
    df_work["periodo"] = df_work[fecha_col].dt.to_period("M")

    periodos_ordenados = sorted(df_work["periodo"].unique())

    feature_names = [
        col for col in df_work.columns
        if col not in [target_col, fecha_col, "periodo"]
    ]

    resultados = []

    for i in range(1, len(periodos_ordenados)):
        periodos_train = periodos_ordenados[:i]
        periodo_eval = periodos_ordenados[i]

        df_train = df_work[df_work["periodo"].isin(periodos_train)]
        df_eval = df_work[df_work["periodo"] == periodo_eval]

        if len(df_eval) < 2:
            continue

        X_train = df_train[feature_names].values
        y_train = df_train[target_col].values

        X_eval = df_eval[feature_names].values
        y_eval = df_eval[target_col].values

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_eval_imp = imputer.transform(X_eval)

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_imp)
        X_eval_sc = scaler.transform(X_eval_imp)

        model = Ridge(alpha=1.0)
        model.fit(X_train_sc, y_train)

        y_pred = model.predict(X_eval_sc)
        rmse = np.sqrt(mean_squared_error(y_eval, y_pred))

        resultados.append({
            "periodo_evaluacion": str(periodo_eval),
            "n_meses_entrenamiento": i,
            "rmse": rmse
        })

    return pd.DataFrame(resultados).reset_index(drop=True)