"""
Módulo de postprocesamiento — Scoring y segmentación de la población.

Toma los scores generados por el modelo entrenado y aplica reglas de negocio
para calcular una puntuación compuesta (puntuacion_tlv) que combina:
  - Probabilidad de conversión (score del modelo)
  - Valor potencial del cliente (monto)
  - Contactabilidad (prob_value_contact)
  - Frescura de contacto (grp_campecs06m)

La población se segmenta en 10 grupos de ejecución (grupo_ejec_tlv) usando
cuantiles predefinidos de la puntuación compuesta.

Entry point:
    python -m src.postprocessing \
        --scores-path data/scores.csv \
        --data-path   data/raw/Data_CU_venta.csv \
        --output-path data/postprocessed/output.csv
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Distribución de cuantiles para segmentación (10 grupos de ejecución)
# Definidos por el negocio — NO modificar sin alineación con el equipo.
# ---------------------------------------------------------------------------

DIST_GE = [0, 0.035, 0.087, 0.237, 0.393, 0.529, 0.664, 0.787, 0.862, 0.95, 1.0]

# Mapping de frescura de campaña (grp_campecs06m → prob_frescura)
# Clientes con más campañas recientes tienen mayor factor de frescura.
FRESCURA_MAP = {
    "G1": 0.066,
    "G2": 0.028,
    "G3": 0.022,
    "G4": 0.008,
}
FRESCURA_DEFAULT = 0.004  # Para cualquier grupo no listado (G5+, Otro, etc.)

# Directorios de salida para la réplica (configurar según entorno)
DIR_S3         = "s3://cu-venta-processed/replica"
DIR_ATHENA     = "/mnt/athena/replica"
DIR_ONPREMISE  = "/mnt/onpremise/replica"


# ---------------------------------------------------------------------------
# Función principal de postprocesamiento
# ---------------------------------------------------------------------------

def get_groups(scores: np.ndarray | pd.Series, df_post: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la puntuación compuesta TLV y segmenta la población en grupos de ejecución.

    La puntuación combina:
        puntuacion_tlv = prob × prob_value_contact × log(monto + 1) × prob_frescura

    Donde:
        - prob              = score del modelo (probabilidad de conversión)
        - prob_value_contact= contactabilidad del cliente (ya en df_post)
        - monto             = monto asociado al cliente (ya en df_post)
        - prob_frescura     = factor de frescura según grupo de campaña

    Los grupos de ejecución van del 1 (mayor puntuación) al 10 (menor puntuación).

    Args:
        scores:   Array con las probabilidades predichas por el modelo [0, 1].
                  Debe tener el mismo orden que las filas de df_post.
        df_post:  DataFrame con las columnas originales del cliente.
                  Debe contener: grp_campecs06m, prob_value_contact, monto.

    Returns:
        DataFrame original con las columnas adicionales:
            - prob              : score del modelo
            - prob_frescura     : factor de frescura según grp_campecs06m
            - puntuacion_tlv    : puntuación compuesta
            - grupo_ejec_tlv    : grupo de ejecución (1=mayor prioridad, 10=menor)
    """
    df_post = df_post.copy()

    # 1. Asignar score del modelo
    df_post["prob"] = np.array(scores)

    # 2. Factor de frescura según grupo de campaña
    df_post["prob_frescura"] = np.where(
        df_post["grp_campecs06m"] == "G1", FRESCURA_MAP["G1"],
        np.where(
            df_post["grp_campecs06m"] == "G2", FRESCURA_MAP["G2"],
            np.where(
                df_post["grp_campecs06m"] == "G3", FRESCURA_MAP["G3"],
                np.where(
                    df_post["grp_campecs06m"] == "G4", FRESCURA_MAP["G4"],
                    FRESCURA_DEFAULT,
                ),
            ),
        ),
    )

    # 3. Contactabilidad: reemplazar NaN por un valor mínimo para no anular la puntuación
    df_post["prob_value_contact"] = df_post["prob_value_contact"].fillna(0.000001)

    # 4. Puntuación compuesta TLV
    df_post["puntuacion_tlv"] = (
        df_post["prob"]
        * df_post["prob_value_contact"]
        * np.log(df_post["monto"] + 1)
        * df_post["prob_frescura"]
    )

    # 5. Segmentación en grupos de ejecución usando cuantiles predefinidos
    #    Labels: 10=mayor puntuación ... 1=menor puntuación
    df_post["grupo_ejec_tlv"] = pd.qcut(
        df_post["puntuacion_tlv"],
        q=DIST_GE,
        labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    )

    logger.info(
        "Postprocesamiento completado — %d registros segmentados en %d grupos",
        len(df_post),
        df_post["grupo_ejec_tlv"].nunique(),
    )

    return df_post


def save_replica(
    df_post: pd.DataFrame,
    table: str,
    partition: str,
    dir_s3: str = DIR_S3,
    dir_athena: str = DIR_ATHENA,
    dir_onpremise: str = DIR_ONPREMISE,
) -> pd.DataFrame:
    """
    Construye el DataFrame de réplica en formato estándar y lo exporta como
    archivo pipe-delimitado (|) a tres destinos: S3, Athena y On-Premise.

    Requiere que df_post ya tenga las columnas generadas por get_groups():
        prob, puntuacion_tlv, grupo_ejec_tlv.

    Args:
        df_post:      DataFrame postprocesado (salida de get_groups).
        table:        Nombre base de la tabla, ej. 'EC_OMNICANAL'.
        partition:    Partición de la tabla, ej. '202412' (codmes de ejecución).
        dir_s3:       Directorio S3 (o ruta local) para la réplica S3.
        dir_athena:   Directorio de Athena para la réplica.
        dir_onpremise:Directorio on-premise para la réplica.

    Returns:
        DataFrame de réplica generado.
    """
    df_replica = pd.DataFrame()
    # codmes: derivado de p_fecinformacion (YYYYMMDD → YYYYMM) o partition (p10 → usar partition)
    if "p_fecinformacion" in df_post.columns:
        df_replica["codmes"] = (df_post["p_fecinformacion"] // 100).astype(str)
    elif "partition" in df_post.columns:
        df_replica["codmes"] = df_post["partition"]
    else:
        df_replica["codmes"] = partition  # fallback al parámetro de función
    df_replica["tipdoc"]       = "1"
    df_replica["coddoc"]       = df_post["key_value"]
    df_replica["puntuacion"]   = df_post["puntuacion_tlv"]
    df_replica["modelo"]       = "EC OMNICANAL"
    df_replica["fec_replica"]  = datetime.date.today().strftime("%Y%m%d")
    df_replica["grupo_ejec"]   = df_post["grupo_ejec_tlv"]
    df_replica["score"]        = df_post["prob"]
    df_replica["orden"]        = ""
    df_replica["variable1"]    = df_post["codunicocli"].apply(lambda x: str(x).zfill(10))
    df_replica["variable2"]    = df_post["monto"]
    df_replica["variable3"]    = ""

    # Eliminar duplicados manteniendo el de mayor puntuación
    df_replica = df_replica.sort_values("puntuacion", ascending=False)
    df_replica = df_replica.drop_duplicates("coddoc", keep="first")

    # Generar orden de prioridad (1 = mayor puntuación)
    df_replica["orden"] = (
        df_replica["puntuacion"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # Exportar a los tres destinos
    filename = f"{table}_{partition}.txt"
    for destino, directorio in [
        ("S3",         dir_s3),
        ("Athena",     dir_athena),
        ("On-Premise", dir_onpremise),
    ]:
        path = Path(directorio) / filename
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df_replica.to_csv(path, index=False, sep="|")
            logger.info("Réplica guardada en %s: %s", destino, path)
        except Exception as exc:
            logger.warning("No se pudo guardar réplica en %s (%s): %s", destino, path, exc)

    logger.info(
        "save_replica completado — %d registros únicos | tabla=%s | partición=%s",
        len(df_replica), table, partition,
    )
    return df_replica


def run_postprocessing(
    scores: np.ndarray | pd.Series,
    df_post: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Wrapper de get_groups con logging y guardado opcional a CSV.

    Args:
        scores:      Probabilidades predichas por el modelo.
        df_post:     DataFrame con datos originales del cliente.
        output_path: Si se especifica, guarda el resultado en esa ruta CSV.

    Returns:
        DataFrame postprocesado con grupos de ejecución.
    """
    result = get_groups(scores, df_post)

    # Distribución de grupos
    dist = result["grupo_ejec_tlv"].value_counts().sort_index(ascending=False)
    logger.info("Distribución de grupos de ejecución:\n%s", dist.to_string())

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        logger.info("Resultado guardado en: %s", output_path)

    return result


# ---------------------------------------------------------------------------
# Entry point CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocesamiento — Segmentación TLV con scores del modelo"
    )
    parser.add_argument(
        "--scores-path",
        required=True,
        help="CSV con columna 'prob' (scores del modelo). Una fila por registro.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="CSV con datos originales de clientes (debe tener grp_campecs06m, prob_value_contact, monto).",
    )
    parser.add_argument(
        "--output-path",
        default="data/postprocessed/output_tlv.csv",
        help="Ruta de salida del CSV postprocesado.",
    )
    parser.add_argument(
        "--score-col",
        default="prob",
        help="Nombre de la columna de scores en el CSV de entrada (default: 'prob').",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    args = _parse_args()

    logger.info("Cargando scores desde: %s", args.scores_path)
    df_scores = pd.read_csv(args.scores_path)
    scores = df_scores[args.score_col].values

    logger.info("Cargando datos de clientes desde: %s", args.data_path)
    df_post = pd.read_csv(args.data_path)

    if len(scores) != len(df_post):
        raise ValueError(
            f"El número de scores ({len(scores)}) no coincide con "
            f"el número de filas en data ({len(df_post)})."
        )

    run_postprocessing(scores, df_post, output_path=args.output_path)
    logger.info("Postprocesamiento completado.")


if __name__ == "__main__":
    main()
