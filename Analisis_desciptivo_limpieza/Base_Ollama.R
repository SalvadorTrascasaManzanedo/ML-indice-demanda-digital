###########---------------------------------------#############################.
###########------ CREAR BASE CRUDA OLLAMA --------#############################.
###########---------------------------------------#############################.

# Cargar base procesada
base <- read.csv(
  "base_procesada.csv",
  stringsAsFactors = FALSE
)

# Asegurar formato de la variable criterio
base$nivel_demanda_digital <- factor(
  base$nivel_demanda_digital,
  levels = c("demanda_baja", "demanda_media", "demanda_alta")
)

# Seleccionar variables necesarias
base_pred <- base[, c(
  "nivel_demanda_digital",
  "genres",
  "first_release",
  "last_release",
  "num_releases",
  "num_tracks"
)]

# Asegurar formato numérico
vars_num <- c(
  "first_release",
  "last_release",
  "num_releases",
  "num_tracks"
)

base_pred[, vars_num] <- lapply(
  base_pred[, vars_num],
  as.numeric
)

# La base está referida a 2024
anio_referencia <- 2024

# Crear variables derivadas usadas después como predictoras
base_pred$career_span <- 
  base_pred$last_release - base_pred$first_release

base_pred$years_since_last_release <- 
  anio_referencia - base_pred$last_release

base_pred$releases_cap20 <- ifelse(
  base_pred$num_releases == 20,
  1,
  0
)

base_pred$log_tracks_last_release <- 
  log10(base_pred$num_tracks + 1)

# Limpiar texto de géneros para Ollama
limpiar_generos_ollama <- function(x) {
  
  x <- as.character(x)
  x <- trimws(x)
  
  # Tratar valores vacíos como NA
  x[x %in% c("", "NA", "[]", "nan", "NaN")] <- NA
  
  # Eliminar corchetes y comillas
  x <- gsub("\\[|\\]|'|\"", "", x)
  
  # Pasar a minúsculas
  x <- tolower(trimws(x))
  
  return(x)
}

# Crear base cruda para Ollama
cruda_ollama <- data.frame(
  row_id = 1:nrow(base_pred),
  nivel_demanda_digital = base_pred$nivel_demanda_digital,
  career_span = base_pred$career_span,
  years_since_last_release = base_pred$years_since_last_release,
  num_releases = base_pred$num_releases,
  releases_cap20 = base_pred$releases_cap20,
  log_tracks_last_release = base_pred$log_tracks_last_release,
  genres_raw_original = base_pred$genres,
  stringsAsFactors = FALSE
)

# Géneros limpios
cruda_ollama$genres_raw_clean <- limpiar_generos_ollama(
  cruda_ollama$genres_raw_original
)

# Indicador de ausencia de género
cruda_ollama$sin_genero <- ifelse(
  is.na(cruda_ollama$genres_raw_clean) |
    cruda_ollama$genres_raw_clean == "",
  1,
  0
)

# Texto de entrada para Ollama
cruda_ollama$genres_ollama_input <- ifelse(
  cruda_ollama$sin_genero == 1,
  "Music genres: sin genero",
  paste0("Music genres: ", cruda_ollama$genres_raw_clean)
)

# Reordenar columnas
cruda_ollama <- cruda_ollama[, c(
  "row_id",
  "nivel_demanda_digital",
  "career_span",
  "years_since_last_release",
  "num_releases",
  "releases_cap20",
  "log_tracks_last_release",
  "genres_raw_original",
  "genres_raw_clean",
  "genres_ollama_input",
  "sin_genero"
)]

# Comprobaciones
dim(cruda_ollama)
names(cruda_ollama)
head(cruda_ollama)
table(cruda_ollama$sin_genero)

# Guardar base en el nivel anterior del directorio.
write.csv(
  cruda_ollama,
  file.path("..", "cruda_ollama2.csv"),
  row.names = FALSE
)
