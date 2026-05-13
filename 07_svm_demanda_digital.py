import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score
)

# --------------------------------------------------
# 1. CARGA DE DATOS
# --------------------------------------------------

df = pd.read_csv("base_modelado_limpia.csv")

target = "nivel_demanda_digital"

numeric_features = [
    "career_span",
    "years_since_last_release",
    "num_releases",
    "log_tracks_last_release"
]

categorical_features = [
    "genre_macro_12"
]

df = df[
    [target] + numeric_features + categorical_features
].dropna()

df[target] = pd.Categorical(
    df[target],
    categories=["demanda_baja", "demanda_media", "demanda_alta"],
    ordered=False
)

df["genre_macro_12"] = df["genre_macro_12"].astype("category")

print("\nDimensiones de la base:")
print(df.shape)

print("\nPrimeras filas:")
print(df.head())

print("\nDistribución total de la variable criterio:")
print(df[target].value_counts())
print(df[target].value_counts(normalize=True).round(3))


# --------------------------------------------------
# 2. DEFINICIÓN DE X E y
# --------------------------------------------------

X = df[numeric_features + categorical_features]
y = df[target]


# --------------------------------------------------
# 3. TRAIN / TEST ESTRATIFICADO 70/30
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=123,
    stratify=y
)

print("\nDistribución en entrenamiento:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True).round(3))

print("\nDistribución en prueba:")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True).round(3))


# --------------------------------------------------
# 4. PREPROCESAMIENTO
# --------------------------------------------------

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# --------------------------------------------------
# 5. VALIDACIÓN CRUZADA Y GRID SEARCH
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)

param_grid = [
    {
        "svm__kernel": ["linear"],
        "svm__C": [0.01, 0.1, 1, 10, 100]
    },
    {
        "svm__kernel": ["rbf"],
        "svm__C": [0.01, 0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.001, 0.01, 0.1, 1]
    }
]

pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("svm", SVC(
        class_weight="balanced",
        probability=True,
        decision_function_shape="ovr",
        random_state=123
    ))
])

ruta_modelo = Path("modelo_svm_grid.joblib")

if ruta_modelo.exists():
    print("\n==============================")
    print("CARGANDO MODELO SVM YA ENTRENADO")
    print("==============================")
    grid = joblib.load(ruta_modelo)
    print("Modelo cargado desde modelo_svm_grid.joblib")

else:
    print("\n==============================")
    print("ENTRENANDO MODELO SVM")
    print("==============================")

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    joblib.dump(grid, ruta_modelo)
    print("\nModelo SVM guardado en modelo_svm_grid.joblib")


print("\n==============================")
print("SVM CON KERNEL LINEAL / RBF")
print("==============================")
print("Mejores hiperparámetros:")
print(grid.best_params_)
print("Mejor balanced accuracy en CV:")
print(round(grid.best_score_, 4))


# --------------------------------------------------
# 5B. PARÁMETROS DEL MODELO FINAL
# --------------------------------------------------

mejor_pipeline = grid.best_estimator_
mejor_svm = mejor_pipeline.named_steps["svm"]

print("\n==============================")
print("PARÁMETROS DEL MODELO SVM FINAL")
print("==============================")
print("Kernel seleccionado:", mejor_svm.kernel)
print("C:", mejor_svm.C)
print("Gamma:", mejor_svm.gamma)
print("Clases:", mejor_svm.classes_)
print("Vectores soporte por clase:", mejor_svm.n_support_)
print("Total vectores soporte:", mejor_svm.support_vectors_.shape[0])
print("Interceptos b:", mejor_svm.intercept_)
print("Forma dual_coef_:", mejor_svm.dual_coef_.shape)
print("Forma support_vectors_:", mejor_svm.support_vectors_.shape)


# --------------------------------------------------
# 6. EVALUACIÓN EN TEST
# --------------------------------------------------

pred = grid.predict(X_test)
proba = grid.predict_proba(X_test)

clases = grid.classes_

predicciones_df = X_test.copy()
predicciones_df["real"] = y_test.values
predicciones_df["predicho"] = pred

for i, clase in enumerate(clases):
    predicciones_df[f"prob_{clase}"] = proba[:, i]

predicciones_df.to_csv("predicciones_svm_test.csv", index=False)

print("\nPredicciones guardadas en predicciones_svm_test.csv")


y_test_bin = label_binarize(
    y_test,
    classes=clases
)

auc_macro = roc_auc_score(
    y_test_bin,
    proba,
    average="macro",
    multi_class="ovr"
)

recall_alta = recall_score(
    y_test,
    pred,
    labels=["demanda_alta"],
    average=None,
    zero_division=0
)[0]

precision_alta = precision_score(
    y_test,
    pred,
    labels=["demanda_alta"],
    average=None,
    zero_division=0
)[0]

f1_alta = f1_score(
    y_test,
    pred,
    labels=["demanda_alta"],
    average=None,
    zero_division=0
)[0]

resultados = {
    "modelo": "SVM_kernel_lineal_rbf",
    "accuracy": accuracy_score(y_test, pred),
    "balanced_accuracy": balanced_accuracy_score(y_test, pred),
    "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
    "precision_demanda_alta": precision_alta,
    "recall_demanda_alta": recall_alta,
    "f1_demanda_alta": f1_alta,
    "kappa": cohen_kappa_score(y_test, pred),
    "mcc": matthews_corrcoef(y_test, pred),
    "auc_macro_ovr": auc_macro
}

print("\n==============================")
print("EVALUACIÓN FINAL EN TEST")
print("==============================")
print(classification_report(y_test, pred, zero_division=0))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, pred, labels=clases))

print("\nMétricas resumen:")
resultados_df = pd.DataFrame([resultados])
print(resultados_df.round(4))

resultados_df.to_csv(
    "resultados_svm_rendimiento.csv",
    index=False
)


# --------------------------------------------------
# 7. MATRIZ DE CONFUSIÓN
# --------------------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    pred,
    labels=clases,
    xticks_rotation=45,
    cmap="Blues"
)

plt.title("Matriz de confusión - SVM")
plt.tight_layout()
plt.savefig("matriz_confusion_svm.png", dpi=300)
plt.close()


# --------------------------------------------------
# 8. CURVAS ROC MULTICLASE
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

for i, clase in enumerate(clases):
    RocCurveDisplay.from_predictions(
        y_test_bin[:, i],
        proba[:, i],
        name=f"ROC {clase}",
        ax=ax
    )

ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_title(f"Curvas ROC - SVM\nAUC macro OvR = {auc_macro:.3f}")
ax.set_xlabel("1 - Especificidad")
ax.set_ylabel("Sensibilidad")

plt.tight_layout()
plt.savefig("roc_svm.png", dpi=300)
plt.close()

print("\nAUC macro OvR:", round(auc_macro, 4))


# --------------------------------------------------
# 9. GRÁFICO COMPARATIVO DE MÉTRICAS
# --------------------------------------------------

metricas_plot = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_demanda_alta",
    "recall_demanda_alta",
    "f1_demanda_alta",
    "kappa",
    "mcc",
    "auc_macro_ovr"
]

resultados_largo = resultados_df.melt(
    id_vars="modelo",
    value_vars=metricas_plot,
    var_name="metrica",
    value_name="valor"
)

plt.figure(figsize=(10, 6))

plt.plot(
    resultados_largo["metrica"],
    resultados_largo["valor"],
    marker="o",
    label="SVM"
)

plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.ylabel("Valor")
plt.title("Rendimiento del modelo SVM")
plt.legend()
plt.tight_layout()
plt.savefig("comparacion_metricas_svm.png", dpi=300)
plt.close()


# --------------------------------------------------
# 10. GUARDAR PARÁMETROS DEL MODELO
# --------------------------------------------------

parametros_modelo = {
    "kernel": mejor_svm.kernel,
    "C": mejor_svm.C,
    "gamma": mejor_svm.gamma,
    "n_vectores_soporte_total": mejor_svm.support_vectors_.shape[0],
    "n_vectores_soporte_demanda_baja": mejor_svm.n_support_[0],
    "n_vectores_soporte_demanda_media": mejor_svm.n_support_[1],
    "n_vectores_soporte_demanda_alta": mejor_svm.n_support_[2],
    "interceptos_b": mejor_svm.intercept_.tolist(),
    "shape_dual_coef": str(mejor_svm.dual_coef_.shape),
    "shape_support_vectors": str(mejor_svm.support_vectors_.shape)
}

parametros_df = pd.DataFrame([parametros_modelo])
parametros_df.to_csv("parametros_svm_final.csv", index=False)


print("\nProceso terminado.")
print("Archivos generados:")
print("- modelo_svm_grid.joblib")
print("- resultados_svm_rendimiento.csv")
print("- predicciones_svm_test.csv")
print("- parametros_svm_final.csv")
print("- matriz_confusion_svm.png")
print("- roc_svm.png")
print("- comparacion_metricas_svm.png")