import time
import requests
import numpy as np
import pandas as pd


INPUT_CSV = "cruda_ollama.csv"
OUTPUT_EMBEDDINGS_CSV = "outputs/embeddings_generos_ollama.csv"
OUTPUT_BASE_CSV = "outputs/base_con_embeddings_genero.csv"

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "bge-m3"
BATCH_SIZE = 32


def get_embeddings_batch(texts, model=MODEL):
    """
    Envía un lote de textos a Ollama y devuelve una matriz de embeddings.
    """
    payload = {
        "model": model,
        "input": texts
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)

    if response.status_code != 200:
        raise RuntimeError(
            f"Error en Ollama {response.status_code}: {response.text}"
        )

    data = response.json()
    return data["embeddings"]


def main():
    base = pd.read_csv(INPUT_CSV)

    # Textos únicos para no recalcular embeddings repetidos
    textos = (
        base["genres_ollama_input"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    print(f"Número de textos únicos para Ollama: {len(textos)}")

    all_embeddings = []

    for i in range(0, len(textos), BATCH_SIZE):
        batch = textos[i:i + BATCH_SIZE]
        print(f"Procesando lote {i // BATCH_SIZE + 1}: {i} - {i + len(batch)}")

        embeddings_batch = get_embeddings_batch(batch)
        all_embeddings.extend(embeddings_batch)

        # Pausa pequeña para evitar saturar el servidor local
        time.sleep(0.2)

    embeddings = np.array(all_embeddings)

    emb_cols = [f"genre_emb_{i+1}" for i in range(embeddings.shape[1])]

    embeddings_df = pd.DataFrame(embeddings, columns=emb_cols)
    embeddings_df["genres_ollama_input"] = textos

    embeddings_df.to_csv(
        OUTPUT_EMBEDDINGS_CSV,
        index=False,
        encoding="utf-8"
    )

    # Unir embeddings a la base original
    base_embeddings = base.merge(
        embeddings_df,
        on="genres_ollama_input",
        how="left"
    )

    base_embeddings.to_csv(
        OUTPUT_BASE_CSV,
        index=False,
        encoding="utf-8"
    )

    print("Embeddings generados correctamente.")
    print(f"Archivo embeddings: {OUTPUT_EMBEDDINGS_CSV}")
    print(f"Base con embeddings: {OUTPUT_BASE_CSV}")
    print(f"Dimensiones finales: {base_embeddings.shape}")


if __name__ == "__main__":
    main()