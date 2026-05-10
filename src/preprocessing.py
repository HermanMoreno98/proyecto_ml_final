"""
Preprocessing module — adaptado al schema real del dataset CU Venta.

El dataset tiene 10 periodos mensuales (p1_extrac.csv ... p10_extrac.csv)
identificados por las columnas ``partition`` (p1, p2, …, p10) y
``p_fecinformacion`` (YYYYMMDD como entero).

Split temporal OOT:
  - Validación : partition == 'p10' (diciembre 2022)
  - Train / Test: partition in [p1..p9], split 70/30 (random_state=123)

Entry point CLI::

    python -m src.preprocessing --input data/raw/_unified.csv --output-dir data/processed
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes ajustadas al schema real
# ---------------------------------------------------------------------------

NAN_THRESHOLD = 80  # columnas con más del 80% NaN se eliminan

# Columnas de identificación/metadata — se conservan en los CSVs para
# trazabilidad y postprocesamiento, pero NO se usan como features del modelo
ID_COLS: list[str] = [
    "partition",
    "key_value",
    "codunicocli",
    "tip_doc",
    "fch_creacion",
    "p_fecinformacion",
]

TARGET_COL = "target"

# Columnas binarias (flag) o conteos donde NaN ≡ 0
COLS_FILL_ZERO: list[str] = [
    "flg_saltotppe12m",
    "num_incrsaldispefe06m",
    "seg_un",
    "ctd_entrdm01",
]

# Columnas categóricas a encodificar con LabelEncoder
FEATURES_ENCODER: list[str] = [
    "grp_campecs06m",
    "ent_1erlntcrallsfm01",
]

# Periodo más reciente → validación OOT (fuera del tiempo)
VALIDATION_PARTITION = "p10"
TEST_SIZE = 0.30
RANDOM_STATE = 123


# ---------------------------------------------------------------------------
# Funciones de preprocesamiento
# ---------------------------------------------------------------------------


def load_data(data_path: str | Path) -> pd.DataFrame:
    """Lee el CSV unificado de entrada."""
    df = pd.read_csv(data_path)
    logger.info("Datos cargados: %d filas × %d columnas", *df.shape)
    return df


def drop_high_nan_columns(
    df: pd.DataFrame, threshold: float = NAN_THRESHOLD
) -> tuple[pd.DataFrame, list[str]]:
    """Elimina columnas con más del ``threshold`` % de valores nulos."""
    cols_to_drop = []
    for col in df.columns:
        pct_nan = df[col].isna().sum() / len(df) * 100
        if pct_nan > threshold:
            logger.info("  Eliminando columna '%s' (%.1f %% NaN)", col, pct_nan)
            cols_to_drop.append(col)
    df = df.drop(columns=cols_to_drop)
    logger.info("Columnas eliminadas por NaN > %d%%: %d", threshold, len(cols_to_drop))
    return df, cols_to_drop


def impute_zeros(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Imputa con 0 las columnas de flags y conteos definidas en COLS_FILL_ZERO."""
    if cols is None:
        cols = COLS_FILL_ZERO
    existing = [c for c in cols if c in df.columns]
    for col in existing:
        df[col] = df[col].fillna(0)
    return df


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa con la mediana todas las columnas numéricas que aún tengan NaN,
    excluyendo las columnas de ID y la variable target.
    """
    skip = set(ID_COLS) | {TARGET_COL}
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        if col in skip:
            continue
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug("  Imputando '%s' con mediana=%.4f (%d NaN)", col, median_val, n_nan)
    return df


def impute_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa con 'Otro' las columnas categóricas (object) que tengan NaN."""
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    skip = set(ID_COLS)
    for col in obj_cols:
        if col in skip:
            continue
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            df[col] = df[col].fillna("Otro")
            logger.debug("  Imputando '%s' con 'Otro' (%d NaN)", col, n_nan)
    return df


def encode_categorical_features(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """LabelEncoding de las features categóricas. Devuelve encoders para trazabilidad."""
    if features is None:
        features = FEATURES_ENCODER
    encoders: dict[str, LabelEncoder] = {}
    for col in features:
        if col not in df.columns:
            logger.warning("Columna '%s' no encontrada para encoding, se omite.", col)
            continue
        enc = LabelEncoder()
        enc.fit(df[col].astype(str))
        df[col] = enc.transform(df[col].astype(str))
        encoders[col] = enc
        logger.info("  Encoded '%s': %d categorías", col, len(enc.classes_))
    return df, encoders


def split_data(
    df: pd.DataFrame,
    val_partition: str = VALIDATION_PARTITION,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separa en train, test y validación OOT.

    - Validación: partition == val_partition (p10 = diciembre 2022)
    - Train / Test: resto, split 70/30 con stratify en target
    """
    df_val = df[df["partition"] == val_partition].copy()
    df_main = df[df["partition"] != val_partition].copy()
    df_train, df_test = train_test_split(
        df_main, test_size=test_size, random_state=random_state,
        stratify=df_main[TARGET_COL]
    )
    logger.info(
        "Split — train: %d | test: %d | val (OOT): %d",
        len(df_train), len(df_test), len(df_val),
    )
    return df_train, df_test, df_val


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------


def run_preprocessing(
    data_path: str | Path,
    nan_threshold: float = NAN_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Ejecuta el pipeline de preprocesamiento completo.

    Pasos:
        1. Cargar datos
        2. Eliminar columnas con > 80% NaN
        3. Imputar flags/conteos con 0
        4. Imputar numéricas restantes con mediana
        5. Imputar categóricas con 'Otro'
        6. Encodificar columnas categóricas (LabelEncoder)
        7. Split temporal (p10 → val OOT, p1-p9 → train/test 70/30)

    Returns:
        (df_train, df_test, df_val, metadata)
    """
    df = load_data(data_path)
    df, dropped_cols = drop_high_nan_columns(df, threshold=nan_threshold)
    df = impute_zeros(df)
    df = impute_median(df)
    df = impute_categoricals(df)
    df, encoders = encode_categorical_features(df)
    df_train, df_test, df_val = split_data(df)

    metadata = {
        "dropped_columns": dropped_cols,
        "encoders": encoders,
        "nan_threshold": nan_threshold,
        "validation_partition": VALIDATION_PARTITION,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "shapes": {
            "train": df_train.shape,
            "test": df_test.shape,
            "val": df_val.shape,
        },
    }

    return df_train, df_test, df_val, metadata


# ---------------------------------------------------------------------------
# Entry point CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocessing pipeline — CU Venta ML")
    parser.add_argument("--input", required=True,
                        help="Ruta al CSV de entrada (e.g. data/raw/_unified.csv)")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Directorio de salida para los tres splits CSV")
    parser.add_argument("--nan-threshold", type=float, default=NAN_THRESHOLD,
                        help="Umbral NaN (%%) para eliminar columnas (default: 80)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train, df_test, df_val, meta = run_preprocessing(
        data_path=args.input,
        nan_threshold=args.nan_threshold,
    )

    df_train.to_csv(out_dir / "df_train.csv", index=False)
    df_test.to_csv(out_dir / "df_test.csv", index=False)
    df_val.to_csv(out_dir / "df_val.csv", index=False)

    logger.info("Archivos guardados en: %s", out_dir)
    logger.info(
        "Shapes → train: %s | test: %s | val: %s",
        meta["shapes"]["train"], meta["shapes"]["test"], meta["shapes"]["val"],
    )


if __name__ == "__main__":
    main()
