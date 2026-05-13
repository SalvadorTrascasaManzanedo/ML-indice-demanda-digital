# 07_pycaret_balance_clases.py

import pandas as pd
from pycaret.classification import *

# =========================
# 1. Cargar base
# =========================

df = pd.read_csv("base_modelado_limpia.csv")

df = df[
    [
        "nivel_demanda_digital",
        "genre_macro_12",
        "career_span",
        "years_since_last_release",
        "num_releases",
        "log_tracks_last_release",
    ]
].dropna()

df["nivel_demanda_digital"] = df["nivel_demanda_digital"].astype("category")
df["genre_macro_12"] = df["genre_macro_12"].astype("category")

print("\nDistribución original:")
print(df["nivel_demanda_digital"].value_counts())


# =========================
# 2. Igualar clases por submuestreo
# =========================

n_min = df["nivel_demanda_digital"].value_counts().min()

df_balanceada = (
    df
    .groupby("nivel_demanda_digital", group_keys=False)
    .apply(lambda x: x.sample(n=n_min, random_state=123))
    .sample(frac=1, random_state=123)
    .reset_index(drop=True)
)

print("\nDistribución balanceada:")
print(df_balanceada["nivel_demanda_digital"].value_counts())


# =========================
# 3. Configurar PyCaret
# =========================

clf = setup(
    data=df_balanceada,
    target="nivel_demanda_digital",

    categorical_features=["genre_macro_12"],
    numeric_features=[
        "career_span",
        "years_since_last_release",
        "num_releases",
        "log_tracks_last_release",
    ],

    train_size=0.70,
    data_split_stratify=True,

    fold_strategy="stratifiedkfold",
    fold=5,

    # Ya no hace falta fix_imbalance porque hemos balanceado manualmente
    fix_imbalance=False,

    normalize=True,
    session_id=123,
    html=False
)


# =========================
# 4. Comparar modelos
# =========================

best_model = compare_models(
    include=["rf", "rbfsvm", "svm", "gbc", "lightgbm", "lr"],
    sort="F1"
)

tabla = pull()
tabla.to_csv("comparacion_pycaret_base_balanceada.csv", index=False)

print("\nRanking con base balanceada:")
print(tabla)

print("\nMejor modelo:")
print(best_model)


# =========================
# 5. Ajustar mejor modelo
# =========================

tuned_model = tune_model(
    best_model,
    optimize="F1",
    n_iter=20
)

tabla_tuning = pull()
tabla_tuning.to_csv("tuning_pycaret_base_balanceada.csv", index=False)

print("\nModelo ajustado:")
print(tuned_model)


# =========================
# 6. Evaluación
# =========================

predicciones = predict_model(tuned_model)

metricas = pull()
metricas.to_csv("metricas_pycaret_base_balanceada.csv", index=False)

print("\nMétricas en test balanceado:")
print(metricas)

plot_model(tuned_model, plot="confusion_matrix")
plot_model(tuned_model, plot="class_report")