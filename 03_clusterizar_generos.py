import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


INPUT_CSV = "outputs/base_con_embeddings_genero.csv"

OUTPUT_BASE_CLUSTER = "outputs/base_con_clusters_genero.csv"
OUTPUT_CLUSTER_SUMMARY = "outputs/resumen_clusters_genero.csv"
OUTPUT_PCA_2D = "outputs/pca_clusters_genero_2d.csv"

# Rango de clusters que quieres probar
K_MIN = 5
K_MAX = 30

RANDOM_STATE = 42


def main():
    os.makedirs("outputs", exist_ok=True)

    base = pd.read_csv(INPUT_CSV)

    # Columnas de embedding generadas por Ollama
    emb_cols = [
        col for col in base.columns
        if col.startswith("genre_emb_")
    ]

    if len(emb_cols) == 0:
        raise ValueError("No se encontraron columnas genre_emb_ en la base.")

    print(f"Número de columnas de embedding: {len(emb_cols)}")

    # Para clusterizar géneros, usamos textos únicos.
    # Así evitamos que los géneros muy frecuentes dominen artificialmente el clustering.
    genero_embeddings = (
        base[["genres_ollama_input"] + emb_cols]
        .dropna(subset=["genres_ollama_input"])
        .drop_duplicates(subset=["genres_ollama_input"])
        .reset_index(drop=True)
    )

    X = genero_embeddings[emb_cols].values

    # Normalización L2:
    # recomendable para embeddings porque hace que la comparación se base más
    # en dirección semántica que en magnitud del vector.
    X_norm = normalize(X, norm="l2")

    print(f"Número de géneros/textos únicos a clusterizar: {X_norm.shape[0]}")

    # Selección de número de clusters mediante silhouette
    resultados_k = []

    max_k_real = min(K_MAX, X_norm.shape[0] - 1)

    for k in range(K_MIN, max_k_real + 1):
        modelo = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20
        )

        labels = modelo.fit_predict(X_norm)

        score = silhouette_score(
            X_norm,
            labels,
            metric="cosine"
        )

        resultados_k.append({
            "k": k,
            "silhouette_cosine": score
        })

        print(f"k={k} | silhouette={score:.4f}")

    resultados_k = pd.DataFrame(resultados_k)

    k_optimo = int(
        resultados_k.sort_values(
            "silhouette_cosine",
            ascending=False
        ).iloc[0]["k"]
    )

    print(f"\nNúmero óptimo de clusters elegido: {k_optimo}")

    # Modelo final
    modelo_final = KMeans(
        n_clusters=k_optimo,
        random_state=RANDOM_STATE,
        n_init=50
    )

    genero_embeddings["genre_cluster"] = modelo_final.fit_predict(X_norm)

    # Guardar también los resultados de k probado
    resultados_k.to_csv(
        "outputs/evaluacion_k_clusters_genero.csv",
        index=False,
        encoding="utf-8"
    )

    # Unir cluster a la base completa de sujetos/artistas
    base_cluster = base.merge(
        genero_embeddings[["genres_ollama_input", "genre_cluster"]],
        on="genres_ollama_input",
        how="left"
    )

    # Sujetos sin género
    base_cluster["genre_cluster"] = base_cluster["genre_cluster"].astype("Int64")

    base_cluster.to_csv(
        OUTPUT_BASE_CLUSTER,
        index=False,
        encoding="utf-8"
    )

    print(f"\nBase con sujetos clasificados guardada en: {OUTPUT_BASE_CLUSTER}")

    # Resumen de clusters
    resumen = []

    for cluster_id in sorted(genero_embeddings["genre_cluster"].unique()):
        subset = genero_embeddings[
            genero_embeddings["genre_cluster"] == cluster_id
        ].copy()

        indices = subset.index.to_numpy()

        centroid = modelo_final.cluster_centers_[cluster_id]

        distancias = np.linalg.norm(
            X_norm[indices] - centroid,
            axis=1
        )

        subset["distancia_centroide"] = distancias

        representativos = (
            subset
            .sort_values("distancia_centroide")
            .head(15)["genres_ollama_input"]
            .tolist()
        )

        n_textos = subset.shape[0]

        n_sujetos = base_cluster[
            base_cluster["genre_cluster"] == cluster_id
        ].shape[0]

        resumen.append({
            "genre_cluster": cluster_id,
            "n_textos_genero_unicos": n_textos,
            "n_sujetos": n_sujetos,
            "generos_representativos": " | ".join(representativos)
        })

    resumen_clusters = pd.DataFrame(resumen)

    resumen_clusters.to_csv(
        OUTPUT_CLUSTER_SUMMARY,
        index=False,
        encoding="utf-8"
    )

    print(f"Resumen de clusters guardado en: {OUTPUT_CLUSTER_SUMMARY}")

    # PCA 2D solo para visualización
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_norm)

    pca_2d = pd.DataFrame({
        "genres_ollama_input": genero_embeddings["genres_ollama_input"],
        "genre_cluster": genero_embeddings["genre_cluster"],
        "pca_1": coords[:, 0],
        "pca_2": coords[:, 1]
    })

    pca_2d.to_csv(
        OUTPUT_PCA_2D,
        index=False,
        encoding="utf-8"
    )

    print(f"PCA 2D para visualizar clusters guardado en: {OUTPUT_PCA_2D}")

    print("\nPrimeras filas del resumen de clusters:")
    print(resumen_clusters.head(20))


if __name__ == "__main__":
    main()