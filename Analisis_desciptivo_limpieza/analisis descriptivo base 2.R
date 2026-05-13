###########---------------------------------------#############################.
###########------ ANALISIS DESCRIPTIVO BASE 2 ----#############################.  
###########---------------------------------------#############################.

# Carga de base de datos
data2 <- read.csv("CLEANED_featured_Spotify_artist_info2.csv",
                  stringsAsFactors = FALSE)

head(data2)

# Dimensiones de la base original
nrow(data2); ncol(data2)

# Duplicados por artista
sum(duplicated(data2$ids))
sum(duplicated(data2$names))

# Decisión:
# La segunda base puede tener varias filas por artista, por lo que se crea una
# versión a nivel artista.
#
# Justificación:
# El modelo trabaja a nivel artista, no a nivel aparición en playlist o fecha.
# Por tanto, para evitar duplicar artificialmente artistas, se conserva una
# única fila por artista. Si existe la variable dates, se conserva la observación
# más reciente.

if ("dates" %in% names(data2)) {
  data2$dates <- as.Date(data2$dates)
  data2 <- data2[order(data2$ids, data2$dates), ]
  data2_artist <- data2[!duplicated(data2$ids, fromLast = TRUE), ]
} else {
  data2_artist <- data2[!duplicated(data2$ids), ]
}


# Dimensiones de la base a nivel artista
nrow(data2_artist); ncol(data2_artist)

# Descriptivos generales
psych::describe(data2_artist)

# Variables numéricas
vars_numeric <- c(
  "popularity",
  "followers",
  "num_releases",
  "num_tracks",
  "monthly_listeners"
)

# Base numérica
datos_num2 <- data2_artist[, vars_numeric]

# Cuartiles para conocer la distribución
sapply(
  datos_num2,
  quantile,
  probs = c(0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1),
  na.rm = TRUE
)

#####################       VARIABLE CRITERIO   ################################

#####--- VARIABLE FOLLOWERS

# Datos crudos
hist(
  data2_artist$followers,
  breaks = 50,
  main = "Distribución de followers - base 2",
  xlab = "Followers",
  col = "orange"
)

# Transformación logarítmica
data2_artist$log_followers <- log10(data2_artist$followers + 1)

hist(
  data2_artist$log_followers,
  breaks = 50,
  main = "Distribución logarítmica de followers - base 2",
  xlab = "log10(followers + 1)",
  col = "orange"
)

# Comparación antes y después
psych::describe(
  data2_artist[, c("followers", "log_followers")]
)

####--- VARIABLE MONTHLY LISTENERS

par(mfrow = c(1, 2))

# Datos crudos
hist(
  data2_artist$monthly_listeners,
  breaks = 50,
  main = "Distribución de monthly_listeners - base 2",
  xlab = "Oyentes mensuales",
  col = "orange"
)

# Transformación logarítmica
data2_artist$log_monthly_listeners <- log10(data2_artist$monthly_listeners + 1)

hist(
  data2_artist$log_monthly_listeners,
  breaks = 50,
  main = "Distribución logarítmica de monthly_listeners - base 2",
  xlab = "log10(monthly_listeners + 1)",
  col = "orange"
)

par(mfrow = c(1, 1))

# Comparación antes y después
psych::describe(
  data2_artist[, c("monthly_listeners", "log_monthly_listeners")]
)

####--- VARIABLE POPULARITY

# Descriptivos
psych::describe(data2_artist$popularity)

# Histograma
hist(
  data2_artist$popularity,
  breaks = 50,
  main = "Distribución de popularidad - base 2",
  xlab = "Popularidad",
  col = "orange"
)

############### CONSTRUCCIÓN DEL ÍNDICE DE DEMANDA DIGITAL #####################

# Decisión:
# Se construye el índice con monthly_listeners, followers y popularity.
#
# Justificación:
# monthly_listeners aproxima audiencia reciente, followers comunidad estable
# y popularity popularidad relativa en Spotify. Las dos primeras variables se
# transforman con logaritmo por su fuerte asimetría. Después, las tres se
# estandarizan para poder combinarlas en una escala común.

data2_artist$z_monthly_listeners <- as.numeric(scale(data2_artist$log_monthly_listeners))
data2_artist$z_followers <- as.numeric(scale(data2_artist$log_followers))
data2_artist$z_popularity <- as.numeric(scale(data2_artist$popularity))

data2_artist$indice_demanda_digital <- rowMeans(
  data2_artist[, c("z_monthly_listeners", "z_followers", "z_popularity")],
  na.rm = TRUE
)

head(
  data2_artist[, c(
    "z_monthly_listeners",
    "z_followers",
    "z_popularity",
    "indice_demanda_digital"
  )]
)

# Descriptivos del índice
psych::describe(data2_artist$indice_demanda_digital)

# Distribución del índice
hist(
  data2_artist$indice_demanda_digital,
  breaks = 50,
  main = "Distribución del índice de demanda digital - base 2",
  xlab = "Índice de demanda digital",
  col = "orange"
)

############### COHERENCIA DEL ÍNDICE #########################################

# Componentes del índice
componentes_indice2 <- data2_artist[, c(
  "z_monthly_listeners",
  "z_followers",
  "z_popularity"
)]

# Correlaciones entre componentes
round(
  cor(componentes_indice2, use = "complete.obs"),
  2
)

# ACP
pca_demanda2 <- prcomp(
  componentes_indice2,
  center = FALSE,
  scale. = FALSE
)

summary(pca_demanda2)
pca_demanda2$rotation
############### CONSTRUCCIÓN VARIABLE CRITERIO ################################

# Decisión:
# Se construye nivel_cartel usando el índice de demanda digital y un filtro
# adicional de popularity >= 60 para la categoría de demanda alta.
#
# Justificación:
# El análisis previo mostró que exigir popularity >= 60 mejora la validez
# conceptual de la categoría alta, evitando que artistas con popularidad moderada
# entren como demanda alta solo por compensación en el índice.

p50_indice2 <- quantile(data2_artist$indice_demanda_digital, 0.50, na.rm = TRUE)
p85_indice2 <- quantile(data2_artist$indice_demanda_digital, 0.85, na.rm = TRUE)

data2_artist$nivel_cartel <- ifelse(
  data2_artist$indice_demanda_digital >= p85_indice2 &
    data2_artist$popularity >= 60,
  "demanda_alta",
  ifelse(
    data2_artist$indice_demanda_digital >= p50_indice2,
    "demanda_media",
    "demanda_baja"
  )
)

data2_artist$nivel_cartel <- factor(
  data2_artist$nivel_cartel,
  levels = c("demanda_baja", "demanda_media", "demanda_alta")
)

# Proporciones variable criterio
table(data2_artist$nivel_cartel)
prop.table(table(data2_artist$nivel_cartel)) * 100

############### COMPROBACIÓN DE COHERENCIA ####################################

# Decisión:
# Comprobar si las clases de nivel_cartel están ordenadas de forma coherente
# en las métricas originales.
#
# Justificación:
# Si la variable criterio es adecuada, demanda_alta debería mostrar mayores
# valores de monthly_listeners, followers, popularity e índice de demanda digital.

psych::describeBy(
  data2_artist[, c(
    "monthly_listeners",
    "followers",
    "popularity",
    "log_monthly_listeners",
    "log_followers",
    "indice_demanda_digital"
  )],
  group = data2_artist$nivel_cartel
)

############### GRÁFICOS POR NIVEL DE CARTEL ##################################

par(mfrow = c(1, 3))

boxplot(
  log_monthly_listeners ~ nivel_cartel,
  data = data2_artist,
  main = "Monthly listeners por nivel - base 2",
  xlab = "Nivel de cartel",
  ylab = "log10(monthly_listeners + 1)",
  col = "orange"
)

boxplot(
  log_followers ~ nivel_cartel,
  data = data2_artist,
  main = "Followers por nivel - base 2",
  xlab = "Nivel de cartel",
  ylab = "log10(followers + 1)",
  col = "orange"
)

boxplot(
  popularity ~ nivel_cartel,
  data = data2_artist,
  main = "Popularity por nivel - base 2",
  xlab = "Nivel de cartel",
  ylab = "Popularity",
  col = "orange"
)

par(mfrow = c(1, 1))