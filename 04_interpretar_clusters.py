import pandas as pd


RESUMEN_CSV = "outputs/resumen_clusters_genero.csv"
BASE_CSV = "outputs/base_con_clusters_genero.csv"


def main():
    resumen = pd.read_csv(RESUMEN_CSV)
    base = pd.read_csv(BASE_CSV)

    print("\n==============================")
    print("RESUMEN GENERAL DE CLUSTERS")
    print("==============================\n")

    print(
        resumen[
            [
                "genre_cluster",
                "n_textos_genero_unicos",
                "n_sujetos"
            ]
        ].sort_values("genre_cluster")
    )

    print("\n==============================")
    print("GÉNEROS REPRESENTATIVOS")
    print("==============================\n")

    for _, row in resumen.sort_values("genre_cluster").iterrows():
        cluster = row["genre_cluster"]
        n_sujetos = row["n_sujetos"]
        n_textos = row["n_textos_genero_unicos"]

        print("\n" + "-" * 80)
        print(f"CLUSTER {cluster}")
        print(f"Nº sujetos/artistas: {n_sujetos}")
        print(f"Nº textos de género únicos: {n_textos}")
        print("-" * 80)

        generos = str(row["generos_representativos"]).split(" | ")

        for g in generos[:20]:
            g = g.replace("Music genres: ", "")
            print(f"- {g}")

    print("\n==============================")
    print("DISTRIBUCIÓN DE DEMANDA POR CLUSTER")
    print("==============================\n")

    tabla = pd.crosstab(
        base["genre_cluster"],
        base["nivel_demanda_digital"],
        normalize="index"
    ) * 100

    print(round(tabla, 2))

    tabla.to_csv(
        "outputs/distribucion_demanda_por_cluster.csv",
        encoding="utf-8"
    )

    print("\nArchivo guardado:")
    print("outputs/distribucion_demanda_por_cluster.csv")


if __name__ == "__main__":
    main()