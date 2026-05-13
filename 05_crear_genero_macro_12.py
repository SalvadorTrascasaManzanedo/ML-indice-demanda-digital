import pandas as pd


INPUT_CSV = "outputs/base_con_clusters_genero.csv"
OUTPUT_CSV = "outputs/base_con_genero_macro_12.csv"
OUTPUT_DISTRIBUCION_CSV = "outputs/distribucion_demanda_por_genero_macro_12.csv"
OUTPUT_RESUMEN_CSV = "outputs/resumen_genero_macro_12.csv"


cluster_to_macro_12 = {
    0: "indie",
    1: "ambient_instrumental",
    2: "pop",
    3: "clasica_orquestal",
    4: "jazz_blues_reggae",
    5: "folk_country_singer",
    6: "electronica",
    7: "rock",
    8: "jazz_blues_reggae",
    9: "metal_punk",
    10: "folk_country_singer",
    11: "electronica",
    12: "rock",
    13: "rnb_soul",
    14: "electronica",
    15: "metal_punk",
    16: "folk_country_singer",
    17: "hiphop_rap",
    18: "latin_global",
    19: "indie",
    20: "jazz_blues_reggae",
    21: "rock"
}


def main():
    base = pd.read_csv(INPUT_CSV)

    if "genre_cluster" not in base.columns:
        raise ValueError("No existe la columna 'genre_cluster' en la base.")

    # Crear macro-género interpretable de 12 categorías
    base["genre_macro_12"] = base["genre_cluster"].map(cluster_to_macro_12)

    # Comprobar si algún cluster no ha sido mapeado
    sin_mapear = (
        base.loc[
            base["genre_cluster"].notna() & base["genre_macro_12"].isna(),
            "genre_cluster"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if len(sin_mapear) > 0:
        raise ValueError(
            f"Hay clusters sin asignar a macro-género: {sin_mapear}"
        )

    # Resumen de sujetos por macro-género
    resumen_macro = (
        base
        .groupby("genre_macro_12", dropna=False)
        .agg(
            n_sujetos=("genre_macro_12", "size"),
            n_clusters_originales=("genre_cluster", "nunique")
        )
        .reset_index()
        .sort_values("n_sujetos", ascending=False)
    )

    # Relación cluster original → macro-género
    relacion_cluster_macro = (
        base[["genre_cluster", "genre_macro_12"]]
        .drop_duplicates()
        .sort_values("genre_cluster")
    )

    print("\nRelación entre cluster original y macro-género:")
    print(relacion_cluster_macro)

    print("\nDistribución de sujetos por genre_macro_12:")
    print(resumen_macro)

    # Distribución de demanda digital por macro-género
    tabla_demanda = pd.crosstab(
        base["genre_macro_12"],
        base["nivel_demanda_digital"],
        normalize="index"
    ) * 100

    tabla_demanda = round(tabla_demanda, 2)

    print("\nDistribución porcentual de demanda por genre_macro_12:")
    print(tabla_demanda)

    # Guardar outputs
    base.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8"
    )

    resumen_macro.to_csv(
        OUTPUT_RESUMEN_CSV,
        index=False,
        encoding="utf-8"
    )

    tabla_demanda.to_csv(
        OUTPUT_DISTRIBUCION_CSV,
        encoding="utf-8"
    )

    print("\nArchivos guardados:")
    print(OUTPUT_CSV)
    print(OUTPUT_RESUMEN_CSV)
    print(OUTPUT_DISTRIBUCION_CSV)


if __name__ == "__main__":
    main()