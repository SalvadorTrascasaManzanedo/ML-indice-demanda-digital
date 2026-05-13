###########---------------------------------------#############################.
###########---------ANALISIS PREDICTORES ---------#############################.
###########---------------------------------------#############################.
#- install.packages("psych")
#- install.packages("ggpplot")
#- install.packages("caret")

library(psych)
library(ggplot2)
library(caret)

base <- read.csv(
  "base_procesada.csv",
  stringsAsFactors = FALSE
)

# Asegurar formato de la variable criterio
base$nivel_demanda_digital <- factor(
  base$nivel_demanda_digital,
  levels = c("demanda_baja", "demanda_media", "demanda_alta")
)

dim(base)
str(base)

table(base$nivel_demanda_digital)
round(prop.table(table(base$nivel_demanda_digital)) * 100, 2)

base_pred <- base[, c(
  "nivel_demanda_digital",
  "genres",
  "first_release",
  "last_release",
  "num_releases",
  "num_tracks",
  "fuente"
)]

###########---------------------------------------#############################.
###########------ PREDICTORAS DERIVADAS ----------#############################.
###########---------------------------------------#############################.


# La base está referida a 2024
anio_referencia <- 2024

# Antigüedad discográfica del artista en 2024
base_pred$artist_age_2024 <- anio_referencia - base_pred$first_release

# Años entre primer y último lanzamiento registrado
base_pred$career_span <- base_pred$last_release - base_pred$first_release

# Años desde el último lanzamiento hasta 2024
base_pred$years_since_last_release <- anio_referencia - base_pred$last_release

# Indicador de artistas con 20 o más lanzamientos
# num_releases está capada en 20
base_pred$releases_cap20 <- ifelse(base_pred$num_releases == 20, 1, 0)

# Tracks del lanzamiento más reciente transformados en log
base_pred$log_tracks_last_release <- log10(base_pred$num_tracks + 1)

summary(base_pred[, c(
  "artist_age_2024",
  "career_span",
  "years_since_last_release",
  "num_releases",
  "releases_cap20",
  "log_tracks_last_release"
)])

# Se excluye artist_age_2024 de la base final de modelado.
# es combinación exacta de career_span y years_since_last_release:

base_modelado <- base_pred[, c(
  "nivel_demanda_digital",
  "career_span",
  "years_since_last_release",
  "num_releases",
  "releases_cap20",
  "log_tracks_last_release"
)]

###########---------------------------------------#############################.
###########------ PREPARAR VARIABLE GENERO -------#############################.
###########---------------------------------------#############################.
# Decisión:
# Se transforma genres en una variable categórica simplificada para el modelado.
# La variable genres puede contener varios géneros por artista y demasiadas categorías distintas.
# Se extrae el primer género y se agrupan los géneros poco frecuentes.

extraer_primer_genero <- function(x) {
  
  x <- as.character(x)
  x <- trimws(x)
  
  # Tratar valores vacíos como NA
  x[x %in% c("", "NA", "[]", "nan", "NaN")] <- NA
  
  # Eliminar corchetes y comillas si existieran
  x <- gsub("\\[|\\]|'|\"", "", x)
  
  # Extraer primer género antes de la primera coma
  primer_genero <- sapply(x, function(valor) {
    
    if (is.na(valor) || trimws(valor) == "") {
      return(NA_character_)
    }
    
    partes <- unlist(strsplit(valor, ","))
    trimws(partes[1])
  })
  
  return(tolower(primer_genero))
}

# Crear género principal
base_pred$genero_principal <- extraer_primer_genero(base_pred$genres)

# Revisar géneros más frecuentes
sort(table(base_pred$genero_principal), decreasing = TRUE)[1:30]

# Número de géneros distintos
length(unique(base_pred$genero_principal))

frecuencia_generos <- table(base_pred$genero_principal)

generos_frecuentes <- names(frecuencia_generos[frecuencia_generos >= 50])

base_pred$genero_modelo <- ifelse(
  base_pred$genero_principal %in% generos_frecuentes,
  base_pred$genero_principal,
  "otros"
)

# Los artistas sin género se codifican como categoría propia
base_pred$genero_modelo[is.na(base_pred$genero_principal)] <- "sin_genero"

base_pred$genero_modelo <- as.factor(base_pred$genero_modelo)

# Comprobación
table(base_pred$genero_modelo)

length(levels(base_pred$genero_modelo))

#---- RELACIÓN GENERO VS DEMANDA DIGITAL 
tabla_genero_demanda <- table(
  base_pred$genero_modelo,
  base_pred$nivel_demanda_digital
)

tabla_genero_demanda

round(
  prop.table(tabla_genero_demanda, margin = 1) * 100,
  2
)

chisq.test(tabla_genero_demanda)

chisq_genero <- chisq.test(tabla_genero_demanda)

# Frecuencias esperadas
chisq_genero$expected

# Número de celdas con frecuencia esperada menor que 5
sum(chisq_genero$expected < 5)

# Porcentaje de celdas con frecuencia esperada menor que 5
mean(chisq_genero$expected < 5) * 100

# El test chi-cuadrado muestra una asociación significativa entre genero_modelo
# y nivel_demanda_digital.
#
# Aunque R genera un aviso sobre la aproximación del chi-cuadrado, solo 3 celdas
# presentan frecuencias esperadas menores que 5, lo que equivale al 2.5% del total.
#
# Dado que este porcentaje es bajo, se considera que la prueba es aceptable.
# Por tanto, genero_modelo se mantiene como predictora viable para el modelo.

###########---------------------------------------#############################.
###########------ V DE CRAMER --------------------#############################.
###########---------------------------------------#############################.

# Se calcula V de Cramer para valorar la magnitud de la asociación entre género y nivel de demanda digital.

# Justificación:
# Con muestras grandes, el chi-cuadrado puede ser significativo incluso cuando la asociación es débil. V de Cramer permite evaluar el tamaño del efecto.

chi2 <- as.numeric(chisq_genero$statistic)
n <- sum(tabla_genero_demanda)
k <- min(nrow(tabla_genero_demanda) - 1, ncol(tabla_genero_demanda) - 1)

v_cramer <- sqrt(chi2 / (n * k))

v_cramer

###########---------------------------------------#############################.
###########------ NUMERICAS VS VARIABLE CRITERIO -#############################.
###########---------------------------------------#############################.

vars_num_modelo <- c(
  "career_span",
  "years_since_last_release",
  "num_releases",
  "releases_cap20",
  "log_tracks_last_release"
)

# Medias por nivel de demanda
aggregate(
  base_pred[, vars_num_modelo],
  by = list(nivel_demanda_digital = base_pred$nivel_demanda_digital),
  FUN = mean
)

# Medianas por nivel de demanda
aggregate(
  base_pred[, vars_num_modelo],
  by = list(nivel_demanda_digital = base_pred$nivel_demanda_digital),
  FUN = median
)

###########---------------------------------------#############################.
###########------ TESTS KRUSKAL-WALLIS -----------#############################.
###########---------------------------------------#############################.

# Se aplican pruebas de Kruskal-Wallis para comprobar diferencias entre los tres niveles de demanda digital.

# Es un contraste no paramétrico adecuado cuando no se asume normalidad y la  variable criterio tiene más de dos grupos.

tests_kruskal <- lapply(vars_num_modelo, function(var) {
  
  test <- kruskal.test(
    base_pred[[var]] ~ base_pred$nivel_demanda_digital
  )
  
  data.frame(
    variable = var,
    estadistico = as.numeric(test$statistic),
    p_value = test$p.value
  )
})

tests_kruskal <- do.call(rbind, tests_kruskal)

tests_kruskal

###########---------------------------------------#############################.
###########------ TAMANO DEL EFECTO KRUSKAL ------#############################.
###########---------------------------------------#############################.

# Se calcula una medida de tamaño del efecto para las pruebas de Kruskal-Wallis.

# Con muestras grandes, los p-valores pueden ser significativos incluso con diferencias pequeñas. El tamaño del efecto ayuda a valorar la relevancia real de cada predictora.

n_total <- nrow(base_pred)
k_grupos <- length(levels(base_pred$nivel_demanda_digital))

tests_kruskal$epsilon_squared <- (
  tests_kruskal$estadistico - k_grupos + 1
) / (
  n_total - k_grupos
)

tests_kruskal


###########---------------------------------------#############################.
###########------ COLINEALIDAD ENTRE PREDICTORAS -#############################.
###########---------------------------------------#############################.

# Se revisa la correlación entre predictoras numéricas para detectar redundancias.

# Algunas variables pueden medir información muy similar. Esto no impide entrenar modelos basados en árboles, pero puede afectar a modelos lineales o dificultar la interpretación.

matriz_cor <- cor(
  base_pred[, vars_num_modelo],
  use = "complete.obs"
)

round(matriz_cor, 2)

# Pares de variables con correlación absoluta superior a 0.80
cor_altas <- which(
  abs(matriz_cor) > 0.80 & abs(matriz_cor) < 1,
  arr.ind = TRUE
)

cor_altas <- data.frame(
  var1 = rownames(matriz_cor)[cor_altas[, 1]],
  var2 = colnames(matriz_cor)[cor_altas[, 2]],
  correlacion = matriz_cor[cor_altas]
)

cor_altas

# Se excluye releases_cap20 de la base final de modelado.

# releases_cap20 presenta una correlación alta con num_releases (r = 0.81), ya que se deriva directamente de esta variable. Para evitar redundancia entre predictoras, se conserva num_releases y se excluye releases_cap20.


###########---------------------------------------#############################.
###########------ BASE FINAL DE MODELADO ---------#############################.
###########---------------------------------------#############################.

base_modelado <- base_pred[, c(
  "nivel_demanda_digital",
  "career_span",
  "years_since_last_release",
  "num_releases",
  "log_tracks_last_release",
  "genero_modelo"
)]

# Asegurar formatos
base_modelado$nivel_demanda_digital <- factor(
  base_modelado$nivel_demanda_digital,
  levels = c("demanda_baja", "demanda_media", "demanda_alta")
)

base_modelado$genero_modelo <- as.factor(base_modelado$genero_modelo)



#-- Revisión de las cuantitativas: 

vars_numericas_finales <- c(
  "career_span",
  "years_since_last_release",
  "num_releases",
  "log_tracks_last_release"
)

psych::describe(base_modelado[, vars_numericas_finales])

summary(base_modelado[, vars_numericas_finales])

#-- Revision de la categorica:

# Variable criterio
table(base_modelado$nivel_demanda_digital)

round(
  prop.table(table(base_modelado$nivel_demanda_digital)) * 100,
  2
)
par(mfrow = c(2, 2))

hist(
  base_modelado$career_span,
  main = "Distribución de career_span",
  xlab = "Años entre primer y último lanzamiento",
  ylab = "Frecuencia",
  col = "coral"
)

hist(
  base_modelado$years_since_last_release,
  main = "Distribución de years_since_last_release",
  xlab = "Años desde el último lanzamiento",
  ylab = "Frecuencia",
  col = "coral"
)

hist(
  base_modelado$num_releases,
  main = "Distribución de num_releases",
  xlab = "Número de lanzamientos",
  ylab = "Frecuencia",
  col = "coral"
)

hist(
  base_modelado$log_tracks_last_release,
  main = "Distribución de log_tracks_last_release",
  xlab = "Log tracks último lanzamiento",
  ylab = "Frecuencia",
  col = "coral"
)

par(mfrow = c(1, 1))

# Género preparado para modelado
table(base_modelado$genero_modelo)

round(
  prop.table(table(base_modelado$genero_modelo)) * 100,
  2
)

