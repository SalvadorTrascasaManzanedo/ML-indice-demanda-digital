# ML - Índice de Demanda Digital

Proyecto de aprendizaje automático supervisado orientado a predecir el **nivel de demanda digital de artistas de Spotify** a partir de variables sobre trayectoria discográfica, actividad reciente y macro-género musical.

El objetivo principal es desarrollar un modelo de clasificación multiclase que permita distinguir entre artistas de **demanda baja**, **demanda media** y **demanda alta**.

---

## 1. Descripción del proyecto

Este proyecto desarrolla un flujo completo de modelado supervisado aplicado a datos de artistas de Spotify. La variable objetivo es un **Índice de Demanda Digital (IDD)** construido a partir de tres indicadores:

- Popularidad del artista en Spotify.
- Número de seguidores.
- Oyentes mensuales.

A partir de este índice, los artistas se clasificaron en tres niveles:

- `demanda_baja`
- `demanda_media`
- `demanda_alta`

Como predictoras se utilizaron variables asociadas al perfil musical del artista, su trayectoria, actividad reciente, volumen de lanzamientos y macro-género musical.

---

## 2. Fuente de datos

La base de datos procede de la combinación de dos datasets publicados en Kaggle por Sarah Jeffreson:

- **Featured Spotify artists/tracks with metadata**:  
  https://www.kaggle.com/datasets/sarahjeffreson/featured-spotify-artiststracks-with-metadata

- **Large random Spotify artist sample with metadata**:  
  https://www.kaggle.com/datasets/sarahjeffreson/large-random-spotify-artist-sample-with-metadata

La combinación de ambas fuentes permitió trabajar con una muestra amplia y con mayor representación de artistas de distintos niveles de popularidad.

---

## 3. Construcción del Índice de Demanda Digital

El **Índice de Demanda Digital (IDD)** se construyó a partir de:

- `popularity`
- `followers`
- `monthly_listeners`

Dado que `followers` y `monthly_listeners` presentaban una fuerte asimetría positiva, se aplicó una transformación logarítmica antes de estandarizar las variables. Posteriormente, los tres indicadores se combinaron para generar un índice continuo.

El índice fue recodificado en tres clases:

- **Demanda baja**: artistas por debajo del percentil 50 del IDD.
- **Demanda alta**: artistas por encima del percentil 85 del IDD y con popularidad igual o superior a 60.
- **Demanda media**: casos intermedios.

Esta construcción hizo interpretable la variable criterio, aunque generó un problema relevante de **desbalance entre clases**, especialmente por la menor representación de artistas de demanda alta.

---

## 4. Preprocesamiento y análisis descriptivo

El preprocesamiento incluyó:

- Eliminación de duplicados.
- Tratamiento de valores perdidos.
- Revisión de valores implausibles.
- Transformación logarítmica de variables asimétricas.
- Construcción de variables derivadas de trayectoria musical.
- Agrupación de géneros musicales en macro-géneros.
- Preparación de una base final ligera para modelado.

La variable `genres` presentaba una gran dispersión por la cantidad de etiquetas musicales específicas. Para hacerla operativa, se aplicó un procedimiento semiautomatizado con embeddings semánticos mediante **Ollama** y el modelo `bge-m3`, agrupando los géneros en 12 macro-géneros.

### Distribución de demanda digital por macro-género

![Distribución de demanda digital por género](<Gráficas/Distribución de demanda digital por genero.png>)

### Distribución de predictoras por clase

![Predictoras por clase](<Gráficas/Predictoras por clase.png>)

El análisis descriptivo mostró que las predictoras presentan distribuciones heterogéneas, valores extremos y concentración en determinados rangos. Las variables más relacionadas con el nivel de demanda fueron el **número de lanzamientos** y los **años desde el último lanzamiento**, mientras que el macro-género aportó información complementaria.

---

## 5. Desarrollo del modelo

De forma exploratoria, se empleó PyCaret para identificar algoritmos prometedores. Posteriormente, se desarrolló un modelo de **Random Forest**, seleccionado por su buen rendimiento preliminar y por su adecuación a variables mixtas, relaciones no lineales y distribuciones no normales.

El conjunto de datos se dividió en entrenamiento y prueba con una proporción **70/30**, manteniendo la proporción de clases mediante partición estratificada.

El preprocesamiento del modelo incluyó:

- Variables numéricas sin estandarizar, ya que Random Forest no requiere escalado.
- Codificación *one-hot* del macro-género musical.
- Integración del preprocesamiento y el modelo dentro de un `Pipeline`.

El ajuste del modelo se realizó mediante **GridSearchCV** con **validación cruzada estratificada de 5 particiones**, seleccionando la mejor configuración según **precisión balanceada**. Esta métrica se utilizó por el desbalance de clases detectado en la variable criterio.

La configuración final seleccionada fue:

- `n_estimators = 500`
- `max_depth = 10`
- `min_samples_leaf = 5`
- `max_features = sqrt`
- `class_weight = balanced`

La mejor precisión balanceada media en validación cruzada fue de **0.638**.

---

## 6. Resultados del Random Forest

En el conjunto de prueba, el modelo obtuvo un rendimiento moderado:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.627 |
| Balanced accuracy | 0.634 |
| F1 macro | 0.559 |
| AUC macro OvR | 0.821 |

### Comparación de métricas

![Comparación de métricas Random Forest](<Gráficas/comparacion_metricas_random_forest.png>)

### Curvas ROC

![ROC Random Forest](<Gráficas/roc_random_forest.png>)

El modelo discriminó mejor los extremos de demanda: demanda baja y demanda alta. La clase de demanda media fue la más difícil de separar.

---

## 7. Matriz de confusión

![Matriz de confusión Random Forest](<Gráficas/matriz_confusion_random_forest.png>)

La clase mejor clasificada fue la **demanda baja**, con 2626 aciertos de 3331 casos. La **demanda alta** presentó un recall elevado, detectando 403 de 581 casos reales, aunque con baja precisión por la presencia de falsos positivos. La mayor dificultad apareció en la **demanda media**, que se confundió tanto con demanda baja como con demanda alta.

---

## 8. Importancia de variables

![Importancia de variables Random Forest](<Gráficas/importancia_variables_random_forest.png>)

La importancia de variables mostró que el **número de lanzamientos** fue la predictora más influyente del modelo. Le siguieron los **años desde el último lanzamiento**, la **duración de la carrera** y el **tamaño del último lanzamiento**. Los macro-géneros tuvieron una contribución individual menor, aunque aportaron información complementaria al modelo.

---

## 9. Archivos principales del repositorio

| Archivo | Descripción |
|---|---|
| `01_generar_embeddings.py` | Generación de embeddings semánticos de géneros |
| `03_clusterizar_generos.py` | Agrupación de géneros mediante clustering |
| `04_interpretar_clusters.py` | Interpretación de clusters de género |
| `05_crear_genero_macro_12.py` | Creación de la variable de macro-género |
| `06_analisis_preliminar_pycaret.py` | Comparación exploratoria de modelos con PyCaret |
| `07_random_forest.py` | Entrenamiento, ajuste y evaluación del Random Forest |
| `base_modelado_limpia.csv` | Base final utilizada para el modelado |
| `comparacion_modelos_pycaret.csv` | Resultados exploratorios de PyCaret |
| `importancia_variables_random_forest.csv` | Importancia de variables del modelo final |

---

## 10. Limitaciones

La variable criterio se construyó artificialmente a partir de métricas de Spotify, por lo que depende de decisiones de corte específicas y de la confiabilidad de la métrica de popularidad. Además, la clase de demanda alta está poco representada, lo que dificulta su clasificación precisa.

También pueden existir sesgos derivados de la combinación de datasets con procedimientos de muestreo distintos y de la recodificación semiautomatizada del género musical mediante Ollama.

Para mejorar el rendimiento sería necesario enriquecer tanto la variable criterio como las predictoras con información adicional procedente de otras fuentes, como métricas de consumo, visibilidad e interacción en Spotify, YouTube, Last.fm, MusicBrainz, Instagram o TikTok.
