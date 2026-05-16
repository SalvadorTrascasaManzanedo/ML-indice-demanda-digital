import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, label_binarize
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
# En Random Forest no es necesario estandarizar las variables numéricas.
# El género sí se transforma mediante one-hot encoding.

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
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

param_grid = {
    "rf__n_estimators": [300, 500],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_leaf": [1, 2, 5],
    "rf__max_features": ["sqrt", "log2"],
    "rf__class_weight": ["balanced", "balanced_subsample"]
}

pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("rf", RandomForestClassifier(
        random_state=123,
        n_jobs=-1
    ))
])

grid_rf = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_rf.fit(X_train, y_train)

print("\n==============================")
print("RANDOM FOREST")
print("==============================")
print("Mejores hiperparámetros:")
print(grid_rf.best_params_)
print("Mejor balanced accuracy en CV:")
print(round(grid_rf.best_score_, 4))

# Guardar modelo entrenado
joblib.dump(grid_rf, "modelo_random_forest_grid.joblib")
print("\nModelo Random Forest guardado en modelo_random_forest_grid.joblib")


# --------------------------------------------------
# 5B. PARÁMETROS DEL MODELO FINAL
# --------------------------------------------------

mejor_pipeline = grid_rf.best_estimator_
mejor_rf = mejor_pipeline.named_steps["rf"]
mejor_preprocess = mejor_pipeline.named_steps["preprocess"]

print("\n==============================")
print("PARÁMETROS DEL MODELO RANDOM FOREST FINAL")
print("==============================")
print("Número de árboles:", mejor_rf.n_estimators)
print("Profundidad máxima:", mejor_rf.max_depth)
print("Mínimo de casos por hoja:", mejor_rf.min_samples_leaf)
print("Máximo de variables por partición:", mejor_rf.max_features)
print("Ponderación de clases:", mejor_rf.class_weight)
print("Clases:", mejor_rf.classes_)


# --------------------------------------------------
# 6. EVALUACIÓN EN TEST
# --------------------------------------------------

pred = grid_rf.predict(X_test)
proba = grid_rf.predict_proba(X_test)

clases = grid_rf.classes_

predicciones_df = X_test.copy()
predicciones_df["real"] = y_test.values
predicciones_df["predicho"] = pred

for i, clase in enumerate(clases):
    predicciones_df[f"prob_{clase}"] = proba[:, i]

predicciones_df.to_csv("predicciones_random_forest_test.csv", index=False)

print("\nPredicciones guardadas en predicciones_random_forest_test.csv")


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
    "modelo": "RandomForest",
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
    "resultados_random_forest_rendimiento.csv",
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
    cmap="Greens"
)

plt.title("Matriz de confusión - Random Forest")
plt.tight_layout()
plt.savefig("matriz_confusion_random_forest.png", dpi=300)
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
ax.set_title(f"Curvas ROC - Random Forest\nAUC macro OvR = {auc_macro:.3f}")
ax.set_xlabel("1 - Especificidad")
ax.set_ylabel("Sensibilidad")

plt.tight_layout()
plt.savefig("roc_random_forest.png", dpi=300)
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
    label="Random Forest"
)

plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.ylabel("Valor")
plt.title("Rendimiento del modelo Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig("comparacion_metricas_random_forest.png", dpi=300)
plt.close()


# --------------------------------------------------
# 10. IMPORTANCIA DE VARIABLES
# --------------------------------------------------

feature_names = mejor_preprocess.get_feature_names_out()

importancias_df = pd.DataFrame({
    "variable": feature_names,
    "importancia": mejor_rf.feature_importances_
}).sort_values(
    by="importancia",
    ascending=False
)

importancias_df.to_csv(
    "importancia_variables_random_forest.csv",
    index=False
)

print("\nImportancia de variables:")
print(importancias_df.head(20))

top_importancias = importancias_df.head(20).sort_values(
    by="importancia",
    ascending=True
)

plt.figure(figsize=(8, 7))
plt.barh(
    top_importancias["variable"],
    top_importancias["importancia"]
)
plt.xlabel("Importancia")
plt.title("Importancia de variables - Random Forest")
plt.tight_layout()
plt.savefig("importancia_variables_random_forest.png", dpi=300)
plt.close()


# --------------------------------------------------
# 11. GUARDAR PARÁMETROS DEL MODELO
# --------------------------------------------------

parametros_modelo = {
    "n_estimators": mejor_rf.n_estimators,
    "max_depth": mejor_rf.max_depth,
    "min_samples_leaf": mejor_rf.min_samples_leaf,
    "max_features": mejor_rf.max_features,
    "class_weight": mejor_rf.class_weight,
    "best_score_cv_balanced_accuracy": grid_rf.best_score_,
    "best_params": str(grid_rf.best_params_)
}

parametros_df = pd.DataFrame([parametros_modelo])
parametros_df.to_csv("parametros_random_forest_final.csv", index=False)


print("\nProceso terminado.")
print("Archivos generados:")
print("- modelo_random_forest_grid.joblib")
print("- parametros_random_forest_final.csv")
print("- resultados_random_forest_rendimiento.csv")
print("- predicciones_random_forest_test.csv")
print("- matriz_confusion_random_forest.png")
print("- roc_random_forest.png")
print("- comparacion_metricas_random_forest.png")
print("- importancia_variables_random_forest.csv")
print("- importancia_variables_random_forest.png")