###########---------------------------------------#############################.
###########---------ANALISIS DESCRIPTIVO----------#############################.  
###########---------------------------------------#############################.

#----- Paquetes 
#- install.packages("psych")
#- install.packages("ggplot2")
library(psych) 
library(ggplot2) # Con ggplot se enmascara psych(comando: <<    psych::    >>  )

# Carga de base de datos:
data <- read.csv("CLEANED_Spotify_artist_info.csv")
head(data)

# índices desciptivos
psych::describe(data)

# Dimensiones
nrow(data); ncol(data)

# Duplicados por artista
sum(duplicated(data$ids))
sum(duplicated(data$names))

# Descipriptivos
psych::describe(data)

# Variables numéricas:
vars_numeric <- c(
  "popularity",
  "followers",
  "num_releases",
  "num_tracks",
  "monthly_listeners"
)

# Genero la base de datos con las predictoras de interés:
datos_num <- data[, vars_numeric]

# Cuartiles para conocer la distribución
sapply(
  datos_num,
  quantile,
  probs = c(0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1),
  na.rm = TRUE
)
#####################       VARIABLE CIRTERIO   ################################

#####--- VARIABLE FOLLOWERS

# Datos Crudos:
hist(
  data$followers,
  breaks = 50,
  main = "Distribución de followers",
  xlab = "Followers",
  col = "yellow"
)

# Transformación interpretable:
data$log_followers <- log10(data$followers + 1)

# Histograma de followers transformada en logaritmo
hist(
  data$log_followers,
  breaks = 50,
  main = "Distribución logarítmica de followers",
  xlab = "log10(followers + 1)",
  col = "lightyellow"
)

# Compara antes y después.
psych::describe(
  data[, c("followers", "log_followers")]
)

####--- VARIABLE DEMANDA DIGITAL 28 días antes

# Datos Crudos:
hist(
  data$monthly_listeners,
  breaks = 50,
  main = "Distribución de monthly_listeners",
  xlab = "Escuchas mensuales",
  col = "yellow"
)

# Transformación logarítmica para visualizar la demanda digital
data$log_monthly_listeners <- log10(data$monthly_listeners + 1)

hist(
  data$log_monthly_listeners,
  breaks = 50,
  main = "Distribución logarítmica de escuchas mensuales",
  xlab = "log10(monthly_listeners + 1)",
  col = "lightyellow"
)

# Compara antes y después.
psych::describe(
  data[, c("monthly_listeners", "log_monthly_listeners")]
)


####--- VARIABLE Popularidad

# Distribución:

psych::describe(data$popularity)

hist(
  data$popularity,
  breaks = 50,
  main = "Distribución de popularidad",
  xlab = "Popularidad",
  col = "yellow"
)

############### CONSTRUCCIÓN DEL INDICE DE DEMANDA DIGITAL #####################

# Estandarizaciíones
data$z_monthly_listeners <- as.numeric(scale(data$log_monthly_listeners))
data$z_followers <- as.numeric(scale(data$log_followers))
data$z_popularity <- as.numeric(scale(data$popularity))

# Índice de demanda digital
data$indice_demanda_digital <- rowMeans(
  data[, c("z_monthly_listeners", "z_followers", "z_popularity")],
  na.rm = TRUE
)

head(data[, c("z_monthly_listeners", "z_followers", "z_popularity", "indice_demanda_digital"
)])

# Descriptivos de la variable criterio:
psych::describe(data$indice_demanda_digital)

# Distribución:

hist(
  data$indice_demanda_digital,
  breaks = 50,
  main = "Distribución del índice de demanda digital",
  xlab = "Índice de demanda digital",
  col = "yellow"
)

# Los componentes están relacionados?

componentes_indice <- data[, c(
  "z_monthly_listeners",
  "z_followers",
  "z_popularity"
)]
round(
  cor(componentes_indice, use = "complete.obs"),
  2
)

#ACP
pca_demanda <- prcomp(
  componentes_indice,
  center = FALSE,
  scale. = FALSE
)

summary(pca_demanda)
pca_demanda$rotation

############### CONSTRUCCIÓN VARIABLE CRITERIO ################################

# Percentil 50 menos demanda, 50-85 demanda media, 85-100 alta demanda.
p50_indice <- quantile(data$indice_demanda_digital, 0.50, na.rm = TRUE)
p85_indice <- quantile(data$indice_demanda_digital, 0.85, na.rm = TRUE)


data$nivel_cartel <- ifelse(
  data$indice_demanda_digital >= p85_indice, "demanda_alta",
  ifelse(data$indice_demanda_digital >= p50_indice, "demanda_media", "demanda_baja")
)

data$nivel_cartel <- factor(
  data$nivel_cartel,
  levels = c("demanda_baja", "demanda_media", "demanda_alta")
)

# Proporciones variable:
table(data$nivel_cartel)
prop.table(table(data$nivel_cartel)) * 100

# Comprobación de coherencia:
psych::describeBy(
  data[, c(
    "monthly_listeners",
    "followers",
    "popularity",
    "log_monthly_listeners",
    "log_followers",
    "indice_demanda_digital"
  )],
  group = data$nivel_cartel
)

############### GRÁFICOS POR NIVEL DE CARTEL ##################################

par(mfrow = c(1, 3))

boxplot(
  log_monthly_listeners ~ nivel_cartel,
  data = data,
  main = "Monthly listeners por nivel - base 1",
  xlab = "Nivel de cartel",
  ylab = "log10(monthly_listeners + 1)",
  col = "lightyellow"
)

boxplot(
  log_followers ~ nivel_cartel,
  data = data,
  main = "Followers por nivel - base 1",
  xlab = "Nivel de cartel",
  ylab = "log10(followers + 1)",
  col = "lightyellow"
)

boxplot(
  popularity ~ nivel_cartel,
  data = data,
  main = "Popularity por nivel - base 1",
  xlab = "Nivel de cartel",
  ylab = "Popularity",
  col = "lightyellow"
)

par(mfrow = c(1, 1))





