###########---------------------------------------#############################.
###########--- ANALISIS GENERO PASADO POR OLLAMA--#############################.
###########---------------------------------------#############################.
install.packages("psych")
library(psych)

# Cargar base
base_pred <- read.csv(
  "base_con_genero_macro_12.csv",
  stringsAsFactors = FALSE
)

# Asegurar formato de la variable criterio
base_pred$nivel_demanda_digital <- factor(
  base_pred$nivel_demanda_digital,
  levels = c("demanda_baja", "demanda_media", "demanda_alta")
)

# Asegurar formato de la predictora de género
base_pred$genre_macro_12 <- factor(base_pred$genre_macro_12)

# Comprobaciones iniciales
dim(base_pred)
str(base_pred$genre_macro_12)

table(base_pred$genre_macro_12)
round(prop.table(table(base_pred$genre_macro_12)) * 100, 2)

length(levels(base_pred$genre_macro_12))
sum(is.na(base_pred$genre_macro_12))

###########---------------------------------------#############################.
###########------ GRAFICO CIRCULAR GENERO --------#############################.
###########---------------------------------------#############################.

tabla_macro_genero <- table(base_pred$genre_macro_12)

porcentaje_macro_genero <- round(
  prop.table(tabla_macro_genero) * 100,
  1
)

etiquetas_macro_genero <- paste0(
  names(tabla_macro_genero),
  " (",
  porcentaje_macro_genero,
  "%)"
)

pie(
  tabla_macro_genero,
  labels = etiquetas_macro_genero,
  main = "Distribución de artistas por macro-género"
)

# Guardar gráfico
png(
  "grafico_circular_genero_macro_12.png",
  width = 1200,
  height = 900
)

pie(
  tabla_macro_genero,
  labels = etiquetas_macro_genero,
  main = "Distribución de artistas por macro-género"
)

dev.off()

#### Gráfico coherente con colores previos

datos_plot <- as.data.frame(
  prop.table(tabla_genero_demanda, margin = 1) * 100
)

names(datos_plot) <- c(
  "genre_macro_12",
  "nivel_demanda_digital",
  "porcentaje"
)

# Asegurar orden de la demanda
datos_plot$nivel_demanda_digital <- factor(
  datos_plot$nivel_demanda_digital,
  levels = c("demanda_baja", "demanda_media", "demanda_alta")
)

ggplot(datos_plot, aes(
  x = genre_macro_12,
  y = porcentaje,
  fill = nivel_demanda_digital
)) +
  geom_col(color = "black", linewidth = 0.2) +
  coord_flip() +
  scale_fill_manual(
    values = c(
      "demanda_baja" = "lightyellow",
      "demanda_media" = "lightblue",
      "demanda_alta" = "plum"
    ),
    labels = c(
      "Demanda baja",
      "Demanda media",
      "Demanda alta"
    )
  ) +
  labs(
    title = "Distribución de demanda digital por macro-género",
    subtitle = "V de Cramer = 0.234",
    x = "Macro-género",
    y = "% dentro de cada género",
    fill = "Demanda digital"
  ) +
  ylim(0, 100) +
  theme_minimal()

###########---------------------------------------#############################.
###########------ V DE CRAMER --------------------#############################.
###########---------------------------------------#############################.

chi2 <- as.numeric(chisq_genero$statistic)
n_total <- sum(tabla_genero_demanda)
n_filas <- nrow(tabla_genero_demanda)
n_columnas <- ncol(tabla_genero_demanda)

cramers_v <- sqrt(
  chi2 / (n_total * min(n_filas - 1, n_columnas - 1))
)

cramers_v

###########---------------------------------------#############################.
###########------ GRÁFICO GÉNERO VS DEMANDA -------#############################.
###########---------------------------------------#############################.

library(ggplot2)

# Tabla género x demanda
tabla_genero_demanda <- table(
  base_pred$genre_macro_12,
  base_pred$nivel_demanda_digital
)

# Porcentajes de demanda dentro de cada género
datos_plot <- as.data.frame(
  prop.table(tabla_genero_demanda, margin = 1) * 100
)

names(datos_plot) <- c(
  "genre_macro_12",
  "nivel_demanda_digital",
  "porcentaje"
)

# Asegurar orden de demanda
datos_plot$nivel_demanda_digital <- factor(
  datos_plot$nivel_demanda_digital,
  levels = c("demanda_baja", "demanda_media", "demanda_alta")
)

# Porcentaje total de cada género en la muestra
porcentaje_genero <- round(
  prop.table(table(base_pred$genre_macro_12)) * 100,
  1
)

# Crear etiquetas del eje Y con porcentaje total
etiquetas_genero <- paste0(
  names(porcentaje_genero),
  " (",
  porcentaje_genero,
  "%)"
)

names(etiquetas_genero) <- names(porcentaje_genero)

# Gráfico
ggplot(datos_plot, aes(
  x = genre_macro_12,
  y = porcentaje,
  fill = nivel_demanda_digital
)) +
  geom_col(color = "white") +
  coord_flip() +
  scale_x_discrete(
    labels = etiquetas_genero
  ) +
  scale_fill_manual(
    values = c(
      "demanda_baja" = "lightyellow",
      "demanda_media" = "lightblue",
      "demanda_alta" = "plum"
    ),
    labels = c(
      "Demanda baja",
      "Demanda media",
      "Demanda alta"
    )
  ) +
  labs(
    title = "Distribución de demanda digital por macro-género",
    subtitle = "Entre paréntesis se muestra el peso de cada macro-género en la muestra",
    x = "Macro-género",
    y = "% dentro de cada género",
    fill = "Demanda digital"
  ) +
  theme_minimal()

head(base_pred)

###########---------------------------------------#############################.
###########------ DEPURAR BASE PARA MODELADO -----#############################.
###########---------------------------------------#############################.

# Si la base ya está cargada como base_pred, usar directamente.
# Si no, cargarla:
# base_pred <- read.csv("cruda_ollama_con_embeddings.csv", stringsAsFactors = FALSE)

# Identificar columnas de embeddings
cols_embeddings <- grep("^genre_emb_", names(base_pred), value = TRUE)

# Comprobar cuántas columnas de embeddings hay
length(cols_embeddings)

# Crear base ligera para modelado
base_modelado_limpia <- base_pred[, c(
  "nivel_demanda_digital",
  "career_span",
  "years_since_last_release",
  "num_releases",
  "log_tracks_last_release",
  "genre_macro_12"
)]

# Asegurar formatos
base_modelado_limpia$nivel_demanda_digital <- factor(
  base_modelado_limpia$nivel_demanda_digital,
  levels = c("demanda_baja", "demanda_media", "demanda_alta")
)

base_modelado_limpia$genre_macro_12 <- as.factor(
  base_modelado_limpia$genre_macro_12
)

# Comprobaciones
dim(base_pred)
dim(base_modelado_limpia)

names(base_modelado_limpia)


# Guardar base final ligera
write.csv(
  base_modelado_limpia,
  "../base_modelado_limpia.csv",
  row.names = FALSE
)