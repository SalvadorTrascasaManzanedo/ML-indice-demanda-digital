import pandas as pd
from pycaret.classification import *

# =========================
# 1. Cargar base de datos
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
]

df["nivel_demanda_digital"] = df["nivel_demanda_digital"].astype("category")
df["genre_macro_12"] = df["genre_macro_12"].astype("category")

df = df.dropna()

print("\nPrimeras filas:")
print(df.head())

print("\nDistribución total de clases:")
print(df["nivel_demanda_digital"].value_counts())
print(df["nivel_demanda_digital"].value_counts(normalize=True) * 100)


# =========================
# 2. Configurar PyCaret
# =========================

clf = setup(
    data=df,
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

    fix_imbalance=True,
    normalize=True,

    session_id=123,
    html=False
)


# =========================
# 3. Comprobar train/test
# =========================

y_train = get_config("y_train")
y_test = get_config("y_test")

print("\nDistribución en TRAIN:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True) * 100)

print("\nDistribución en TEST:")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True) * 100)


# =========================
# 4. Ver modelos disponibles
# =========================

print("\nModelos disponibles en PyCaret:")
print(models())


# =========================
# 5. Comparación general inicial
# =========================

print("\nComparación general inicial:")

best_model_general = compare_models(sort="F1")

tabla_general = pull()
tabla_general.to_csv("comparacion_general_pycaret.csv", index=False)

print("\nRanking general:")
print(tabla_general)

print("\nMejor modelo general:")
print(best_model_general)


# =========================
# 6. Comparación dirigida incluyendo SVM RBF
# =========================

print("\nComparación dirigida con modelos prometedores + SVM RBF:")

modelos_dirigidos = compare_models(
    include=[
        "lightgbm",  # Light Gradient Boosting Machine
        "gbc",       # Gradient Boosting Classifier
        "rf",        # Random Forest
        "ada",       # AdaBoost
        "et",        # Extra Trees
        "lr",        # Logistic Regression
        "svm",       # SVM lineal
        "rbfsvm"     # SVM con kernel radial/RBF
    ],
    sort="F1"
)

tabla_dirigida = pull()
tabla_dirigida.to_csv("comparacion_dirigida_con_rbfsvm.csv", index=False)

print("\nRanking dirigido:")
print(tabla_dirigida)

print("\nMejor modelo en comparación dirigida:")
print(modelos_dirigidos)


# =========================
# 7. Ajustar el mejor modelo dirigido
# =========================

print("\nAjustando hiperparámetros del mejor modelo dirigido:")

tuned_model = tune_model(
    modelos_dirigidos,
    optimize="F1",
    n_iter=20
)

tabla_tuning = pull()
tabla_tuning.to_csv("resultados_tuning_pycaret.csv", index=False)

print("\nResultados tuning:")
print(tabla_tuning)

print("\nModelo ajustado:")
print(tuned_model)

print("\nHiperparámetros:")
print(tuned_model.get_params())


# =========================
# 8. Evaluación en test interno
# =========================

print("\nEvaluación en test interno:")

predicciones = predict_model(tuned_model)

metricas_test = pull()
metricas_test.to_csv("metricas_test_pycaret.csv", index=False)
predicciones.to_csv("predicciones_test_pycaret.csv", index=False)

print(metricas_test)


# =========================
# 9. Gráficos principales
# =========================

plot_model(tuned_model, plot="confusion_matrix")
plot_model(tuned_model, plot="class_report")

# Puede fallar en algún modelo, especialmente si no genera probabilidades.
try:
    plot_model(tuned_model, plot="auc")
except Exception as e:
    print("No se pudo generar AUC:", e)

# Puede fallar si el modelo no permite importancia de variables.
try:
    plot_model(tuned_model, plot="feature")
except Exception as e:
    print("No se pudo generar importancia de variables:", e)


# =========================
# 10. Evaluación interactiva
# =========================

evaluate_model(tuned_model)


# =========================
# 11. Guardar modelo final
# =========================

final_model = finalize_model(tuned_model)

save_model(final_model, "modelo_final_pycaret_con_rbfsvm")

print("\nAnálisis terminado.")