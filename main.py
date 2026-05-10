"""
Orquestador principal del pipeline ML — CU Venta.

Ejecuta el pipeline completo de forma secuencial, sin necesidad de Airflow.
Útil para desarrollo local, debugging y ejecuciones manuales puntuales.

Modos de ejecución:
    # Entrenamiento completo con HPO
    python main.py --mode train --hpo

    # Entrenamiento estándar (parámetros por defecto)
    python main.py --mode train

    # Inferencia sobre un mes específico
    python main.py --mode inference --period 12

    # Solo preprocesamiento
    python main.py --mode preprocess

Referencia S3:
    s3://ml-project-ucsp-s3/raw/p1_extrac.csv ... p10_extrac.csv
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directorios
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DIR_RAW = ROOT / "data" / "raw"
DIR_PROCESSED = ROOT / "data" / "processed"
DIR_POSTPROCESSED = ROOT / "data" / "postprocessed"
DIR_MONITORING = ROOT / "data" / "monitoring"
DIR_REPLICA = ROOT / "data" / "replica"

# Parámetros del pipeline
S3_BUCKET = os.environ.get("S3_BUCKET_RAW", "ml-project-ucsp-s3")
S3_PREFIX = os.environ.get("S3_PREFIX_RAW", "raw/")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "cu_venta_e2e")
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "cu_venta_xgb")


# ---------------------------------------------------------------------------
# Funciones de etapas
# ---------------------------------------------------------------------------

def stage_ingest(use_s3: bool = False) -> Path:
    """
    Descarga / localiza los datos crudos.
    Si use_s3=True, descarga desde S3. Si no, usa data/raw/ local.

    Returns:
        Path al directorio con los CSV crudos.
    """
    if use_s3:
        logger.info("Descargando datos desde S3: s3://%s/%s", S3_BUCKET, S3_PREFIX)
        try:
            import boto3
            s3 = boto3.client("s3")
            DIR_RAW.mkdir(parents=True, exist_ok=True)
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    dest = DIR_RAW / Path(key).name
                    logger.info("  Descargando %s → %s", key, dest)
                    s3.download_file(S3_BUCKET, key, str(dest))
        except Exception as exc:
            logger.warning("Error descargando desde S3: %s. Usando datos locales.", exc)
    else:
        logger.info("Usando datos locales en: %s", DIR_RAW)

    csv_files = [f for f in DIR_RAW.glob("*.csv") if not f.name.startswith("_")]
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV en {DIR_RAW}")

    logger.info("Archivos disponibles: %d CSV", len(csv_files))
    return DIR_RAW


def stage_preprocess(data_dir: Path) -> dict:
    """
    Ejecuta el preprocesamiento y guarda los splits.

    Returns:
        dict con rutas de train/test/val y metadata.
    """
    from src.preprocessing import run_preprocessing

    # Unificar todos los CSV en un único DataFrame
    csv_files = sorted(f for f in data_dir.glob("*.csv") if not f.name.startswith("_"))
    logger.info("Unificando %d archivos CSV...", len(csv_files))
    dfs = []
    for f in csv_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as exc:
            logger.warning("Error leyendo %s: %s", f.name, exc)

    if not dfs:
        raise ValueError("No se pudo leer ningún archivo CSV.")

    unified_df = pd.concat(dfs, ignore_index=True)
    unified_path = DIR_PROCESSED / "_unified.csv"
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    unified_df.to_csv(unified_path, index=False)
    logger.info("Dataset unificado guardado en: %s", unified_path)

    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Guardar val crudo (sin encoding) antes de preprocesar — necesario para get_groups()
    df_val_raw = unified_df[unified_df["partition"] == "p10"].copy()
    val_raw_path = DIR_PROCESSED / "df_val_raw.csv"
    df_val_raw.to_csv(val_raw_path, index=False)

    df_train, df_test, df_val, metadata = run_preprocessing(data_path=unified_path)

    train_path = DIR_PROCESSED / "df_train.csv"
    test_path = DIR_PROCESSED / "df_test.csv"
    val_path = DIR_PROCESSED / "df_val.csv"

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    df_val.to_csv(val_path, index=False)

    logger.info(
        "Splits guardados — train: %d | test: %d | val: %d",
        len(df_train), len(df_test), len(df_val),
    )

    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "val_path": str(val_path),
        "val_raw_path": str(val_raw_path),
        "metadata": metadata,
    }


def stage_train(paths: dict, hpo: bool = False, n_trials: int = 30) -> str:
    """
    Entrena el modelo (estándar o con HPO) y lo registra en MLflow.

    Returns:
        run_id del experimento MLflow.
    """
    if hpo:
        from src.training import train_and_log_hpo
        run_id = train_and_log_hpo(
            train_path=paths["train_path"],
            test_path=paths["test_path"],
            val_path=paths["val_path"],
            n_trials=n_trials,
            experiment_name=EXPERIMENT_NAME,
            model_name=MODEL_NAME,
            mlflow_tracking_uri=MLFLOW_URI,
        )
    else:
        from src.training import train_and_log
        run_id = train_and_log(
            train_path=paths["train_path"],
            test_path=paths["test_path"],
            val_path=paths["val_path"],
            experiment_name=EXPERIMENT_NAME,
            model_name=MODEL_NAME,
            mlflow_tracking_uri=MLFLOW_URI,
        )

    logger.info("Entrenamiento completado. MLflow run_id: %s", run_id)
    return run_id


def stage_monitor(paths: dict, run_id: str) -> dict:
    """
    Ejecuta monitoreo PSI, AUC y Recall por decil.

    Además computa métricas por partición (mes) y las guarda en
    data/monitoring/metrics_by_month.csv para el dashboard.

    Returns:
        dict con psi_score y métricas de validación.
    """
    import mlflow
    import mlflow.xgboost
    from sklearn.metrics import roc_auc_score
    from src.monitoring import run_monitoring, compute_psi_score_deciles

    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.xgboost.load_model(model_uri)

    df_train = pd.read_csv(paths["train_path"])
    df_val = pd.read_csv(paths["val_path"])

    id_cols = ["partition", "key_value", "codunicocli", "tip_doc", "fch_creacion", "p_fecinformacion"]
    target_col = "target"
    drop_cols = [c for c in id_cols + [target_col] if c in df_train.columns]

    X_train = df_train.drop(columns=drop_cols)
    y_train = df_train[target_col]
    X_val = df_val.drop(columns=drop_cols)
    y_val = df_val[target_col]

    train_scores = model.predict_proba(X_train)[:, 1]
    val_scores = model.predict_proba(X_val)[:, 1]

    DIR_MONITORING.mkdir(parents=True, exist_ok=True)

    results = run_monitoring(
        df_train_raw=df_train,
        df_val_raw=df_val,
        df_train_processed=df_train,
        df_val_processed=df_val,
        y_train=y_train,
        y_val=y_val,
        train_scores=train_scores,
        val_scores=val_scores,
        output_dir=str(DIR_MONITORING),
        mlflow_active=True,
    )

    psi = results["psi_score"]
    metrics = results["model_metrics_val"]
    logger.info(
        "Monitoreo — PSI=%.4f | AUC=%.4f | Recall=%.4f",
        psi, metrics["auc"], metrics["recall"],
    )

    # --- Métricas por partición (mes) para el dashboard ---
    # Referencia: scores de p1 (primer periodo de entrenamiento)
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    all_scores = model.predict_proba(df_all.drop(columns=[c for c in drop_cols if c in df_all.columns]))[:, 1]
    df_all = df_all.copy()
    df_all["_score"] = all_scores

    # Orden natural: p1, p2, ..., p10
    partition_order = [f"p{i}" for i in range(1, 11)]
    partitions_present = [p for p in partition_order if p in df_all["partition"].unique()]
    ref_scores = df_all[df_all["partition"] == partitions_present[0]]["_score"].values

    month_rows = []
    for part in partitions_present:
        sub = df_all[df_all["partition"] == part]
        part_scores = sub["_score"].values
        part_y = sub[target_col].values
        try:
            auc = round(float(roc_auc_score(part_y, part_scores)), 4) if part_y.sum() > 0 else None
        except Exception:
            auc = None
        try:
            psi_val, _ = compute_psi_score_deciles(ref_scores, part_scores)
            psi_val = round(float(psi_val), 6)
        except Exception:
            psi_val = None
        month_rows.append({"partition": part, "auc": auc, "psi_score": psi_val, "n": len(sub)})

    df_months = pd.DataFrame(month_rows)
    months_path = DIR_MONITORING / "metrics_by_month.csv"
    df_months.to_csv(months_path, index=False)
    logger.info("Métricas por mes guardadas en: %s", months_path)

    # Auto-reentrenamiento si PSI supera umbral (deriva severa)
    if psi > 0.25:
        logger.warning(
            "DERIVA SEVERA detectada (PSI=%.4f > 0.25). "
            "Iniciando re-entrenamiento automático...",
            psi,
        )
        new_run_id = stage_train(paths, hpo=False)
        logger.info("Re-entrenamiento completado. Nuevo run_id: %s", new_run_id)
        return {
            "psi_score": psi,
            "val_auc": metrics["auc"],
            "val_recall": metrics["recall"],
            "retrained": True,
            "new_run_id": new_run_id,
        }

    return {"psi_score": psi, "val_auc": metrics["auc"], "val_recall": metrics["recall"], "retrained": False}


def stage_postprocess(paths: dict, run_id: str, partition: str = "202412") -> None:
    """Ejecuta postprocesamiento, segmentación TLV y réplica."""
    import mlflow
    import mlflow.xgboost
    from src.postprocessing import get_groups, save_replica

    mlflow.set_tracking_uri(MLFLOW_URI)
    model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")

    id_cols = ["partition", "key_value", "codunicocli", "tip_doc", "fch_creacion", "p_fecinformacion"]
    target_col = "target"

    # Val procesado (encoded) — para predict_proba
    df_val = pd.read_csv(paths["val_path"])
    drop_cols = [c for c in id_cols + [target_col] if c in df_val.columns]
    X_val = df_val.drop(columns=drop_cols)
    val_scores = model.predict_proba(X_val)[:, 1]

    # Val crudo (sin encoding) — para get_groups() que usa grp_campecs06m como string
    df_val_raw = pd.read_csv(paths["val_raw_path"])
    df_post = get_groups(val_scores, df_val_raw)

    DIR_POSTPROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DIR_POSTPROCESSED / "output_tlv.csv"
    df_post.to_csv(output_path, index=False)

    DIR_REPLICA.mkdir(parents=True, exist_ok=True)
    save_replica(
        df_post=df_post,
        table="EC_OMNICANAL",
        partition=partition,
        dir_s3=str(DIR_REPLICA),
        dir_athena=str(DIR_REPLICA),
        dir_onpremise=str(DIR_REPLICA),
    )

    logger.info("Postprocesamiento completado. Salida: %s", output_path)


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------

def run_training_pipeline(use_s3: bool = False, hpo: bool = False, n_trials: int = 30) -> None:
    """Ejecuta el pipeline completo de entrenamiento."""
    logger.info("=== PIPELINE DE ENTRENAMIENTO ===")

    # 1. Ingestión
    logger.info("--- Etapa 1: Ingestión ---")
    data_dir = stage_ingest(use_s3=use_s3)

    # 2. Preprocesamiento
    logger.info("--- Etapa 2: Preprocesamiento ---")
    paths = stage_preprocess(data_dir)

    # 3. Entrenamiento
    logger.info("--- Etapa 3: Entrenamiento%s ---", " + HPO (Optuna)" if hpo else "")
    run_id = stage_train(paths, hpo=hpo, n_trials=n_trials)

    # 4. Monitoreo PSI — con auto-reentrenamiento si PSI > 0.25
    logger.info("--- Etapa 4: Monitoreo PSI ---")
    monitoring = stage_monitor(paths, run_id)

    # Si hubo re-entrenamiento automático, usar el nuevo run_id para el postprocesamiento
    active_run_id = monitoring.get("new_run_id", run_id)
    if monitoring.get("retrained"):
        logger.info("Se usará el nuevo modelo (run_id=%s) para postprocesamiento.", active_run_id)

    # 5. Postprocesamiento
    logger.info("--- Etapa 5: Postprocesamiento y Réplica ---")
    stage_postprocess(paths, active_run_id)

    logger.info("=== PIPELINE COMPLETADO ===")
    logger.info("  MLflow run_id : %s", active_run_id)
    logger.info("  Re-entrenado  : %s", monitoring.get("retrained", False))
    logger.info("  PSI score     : %.4f", monitoring["psi_score"])
    logger.info("  Val AUC       : %.4f", monitoring["val_auc"])
    logger.info("  Val Recall    : %.4f", monitoring["val_recall"])


def run_inference_pipeline(period: str, use_s3: bool = False) -> None:
    """
    Ejecuta el pipeline de inferencia sobre un periodo específico.

    Carga el modelo registrado en MLflow (Production → Staging como fallback),
    aplica el preprocesamiento estándar y genera el scoring TLV + réplica.

    Args:
        period: Número de periodo (ej. "10" → p10_extrac.csv)
        use_s3: Si True, descarga los datos desde S3 antes de inferir.
    """
    logger.info("=== PIPELINE DE INFERENCIA (periodo=%s) ===", period)

    data_dir = stage_ingest(use_s3=use_s3)

    # Localizar archivo del periodo
    csv_path = data_dir / f"p{period}_extrac.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró archivo para el periodo {period}: {csv_path}")

    # Cargar modelo desde MLflow Registry (Production → Staging fallback)
    import mlflow
    import mlflow.xgboost
    from src.postprocessing import get_groups, save_replica
    from src.preprocessing import (
        drop_high_nan_columns, impute_zeros, impute_median,
        impute_categoricals, encode_categorical_features,
    )

    mlflow.set_tracking_uri(MLFLOW_URI)
    for stage in ("Production", "Staging"):
        try:
            model_uri = f"models:/{MODEL_NAME}/{stage}"
            model = mlflow.xgboost.load_model(model_uri)
            logger.info("Modelo cargado desde %s: %s", stage, model_uri)
            break
        except Exception as exc:
            logger.warning("No se pudo cargar desde %s: %s", stage, exc)
    else:
        raise RuntimeError(f"No hay versión Production ni Staging del modelo '{MODEL_NAME}' en MLflow.")

    # Preprocesamiento (mismo pipeline que en entrenamiento)
    df_raw = pd.read_csv(csv_path)
    df, _ = drop_high_nan_columns(df_raw)
    df = impute_zeros(df)
    df = impute_median(df)
    df = impute_categoricals(df)
    df, _ = encode_categorical_features(df)

    id_cols = ["partition", "key_value", "codunicocli", "tip_doc", "fch_creacion", "p_fecinformacion"]
    target_col = "target"
    drop_cols = [c for c in id_cols + [target_col] if c in df.columns]
    X = df.drop(columns=drop_cols)
    scores = model.predict_proba(X)[:, 1]

    # get_groups necesita el dataframe crudo (grp_campecs06m como string)
    df_post = get_groups(scores, df_raw)
    DIR_POSTPROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DIR_POSTPROCESSED / f"output_tlv_p{period}.csv"
    df_post.to_csv(output_path, index=False)

    DIR_REPLICA.mkdir(parents=True, exist_ok=True)
    # Derivar codmes desde p_fecinformacion del primer registro
    codmes = str(int(df_raw["p_fecinformacion"].iloc[0]) // 100) if "p_fecinformacion" in df_raw.columns else period
    save_replica(df_post, table="EC_OMNICANAL", partition=codmes,
                 dir_s3=str(DIR_REPLICA), dir_athena=str(DIR_REPLICA),
                 dir_onpremise=str(DIR_REPLICA))

    logger.info("=== INFERENCIA COMPLETADA. Salida: %s ===", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline ML CU Venta — Orquestador principal"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "inference", "preprocess"],
        default="train",
        help="Modo de ejecución del pipeline",
    )
    parser.add_argument(
        "--hpo", action="store_true",
        help="Activar HPO con Optuna (solo en modo train)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=30,
        help="Número de trials de Optuna",
    )
    parser.add_argument(
        "--period", type=str, default="",
        help="Período de inferencia (ej: 12). Solo en modo inference.",
    )
    parser.add_argument(
        "--use-s3", action="store_true",
        help="Descargar datos desde S3 (requiere credenciales AWS configuradas)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode == "train":
        run_training_pipeline(use_s3=args.use_s3, hpo=args.hpo, n_trials=args.n_trials)
    elif args.mode == "inference":
        if not args.period:
            raise ValueError("--period es requerido en modo inference")
        run_inference_pipeline(period=args.period, use_s3=args.use_s3)
    elif args.mode == "preprocess":
        data_dir = stage_ingest(use_s3=args.use_s3)
        stage_preprocess(data_dir)
        logger.info("Preprocesamiento completado.")


if __name__ == "__main__":
    main()

