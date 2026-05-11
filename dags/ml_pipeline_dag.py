"""
DAG de Airflow para el pipeline completo de ML — CU Venta.

Etapas:
    1. ingest            — Descarga el CSV desde S3 (con versionado) al worker.
    2. preprocess        — Ejecuta src.preprocessing y genera train/test/val.
    3. validate_outputs  — Verifica que los splits no estén vacíos y tienen el schema.
    4. train             — Entrena XGBoost con HPO (Optuna) y registra en MLflow.
    5. monitor           — Calcula PSI, AUC y Recall por decil; logea en MLflow.
    6. check_drift       — Evalúa PSI del score: si supera umbral, dispara re-entrenamiento.
    7. register_model    — Transiciona el modelo a Staging en el Model Registry.
    8. postprocess       — Scoring TLV, segmentación en grupos y réplica.

Estrategia de auto-retraining:
    Si el PSI del score de validación supera PSI_ALERT_THRESHOLD (0.15),
    la tarea check_drift dispara automáticamente una nueva corrida completa
    del pipeline de entrenamiento (train → monitor → register_model) mediante
    un TriggerDagRunOperator hacia sí mismo con el flag force_retrain=True.

Variables de Airflow requeridas (Admin → Variables):
    S3_BUCKET_RAW           Bucket S3 con datos crudos.
    S3_PREFIX_RAW           Prefijo S3 donde están p1_extrac.csv … p10_extrac.csv (default: raw/).
    S3_BUCKET_PROCESSED     Bucket S3 para artefactos procesados.
    MLFLOW_TRACKING_URI     URI del servidor MLflow.
    MLFLOW_EXPERIMENT_NAME  Nombre del experimento (default: cu_venta_e2e).
    MLFLOW_MODEL_NAME       Nombre en Model Registry (default: cu_venta_xgb).
    HPO_N_TRIALS            Número de trials Optuna (default: 30).
    PSI_ALERT_THRESHOLD     Umbral PSI para disparar re-entrenamiento (default: 0.25).

Conexiones de Airflow (Admin → Connections):
    aws_default  — credenciales AWS (Access Key + Secret Key + Region).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración del DAG
# ---------------------------------------------------------------------------

DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# Directorio temporal compartido entre tareas (en el worker de Airflow)
WORKDIR = Path("/tmp/cu_venta_pipeline")

# Umbral PSI a partir del cual se considera deriva severa y se re-entrena
# Según instrucciones: > 0.25 → ALERT (re-entrenamiento automático)
PSI_ALERT_THRESHOLD = float(Variable.get("PSI_ALERT_THRESHOLD", default_var="0.25"))


# ---------------------------------------------------------------------------
# Tareas del pipeline
# ---------------------------------------------------------------------------

def _consolidate_csvs_to_disk(csv_paths: list, output_path: Path) -> int:
    """
    Concatena una lista de CSV escribiendo al disco de forma incremental
    para evitar cargar todos los DataFrames en memoria simultáneamente.
    Retorna el número total de filas escritas.
    """
    import pandas as pd

    total_rows = 0
    write_header = True
    for path in csv_paths:
        for chunk in pd.read_csv(path, chunksize=50_000):
            chunk.to_csv(output_path, mode="a", index=False, header=write_header)
            total_rows += len(chunk)
            write_header = False
    return total_rows


def task_ingest(**context) -> None:
    """
    Descarga o copia los CSV al worker según la variable DATA_SOURCE:

    - "s3"    → descarga p1_extrac.csv … p10_extrac.csv desde AWS S3.
    - "local" → usa los CSV ya presentes en /opt/airflow/data/raw/
                 (montados como volumen desde ./data/raw/ del host).

    En ambos casos consolida todos los CSV en un único Data_CU_venta.csv
    y pushea la ruta al XCom.
    """
    data_source = Variable.get("DATA_SOURCE", default_var="s3").strip().lower()

    raw_dir = WORKDIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    unified_path = raw_dir / "Data_CU_venta.csv"

    if data_source == "local":
        # ------------------------------------------------------------------
        # Modo local: leer desde el volumen montado /opt/airflow/data/raw/
        # ------------------------------------------------------------------
        local_raw = Path("/opt/airflow/data/raw")
        csv_files = sorted(local_raw.glob("p*_extrac.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No se encontraron archivos p*_extrac.csv en {local_raw}. "
                "Asegúrate de que ./data/raw/ tiene los CSV en el host."
            )
        logger.info("DATA_SOURCE=local → leyendo %d archivos desde %s", len(csv_files), local_raw)
        version_info_list = [{"key": str(p), "version_id": "local", "etag": "local"} for p in csv_files]
        csv_to_consolidate = list(csv_files)

    else:
        # ------------------------------------------------------------------
        # Modo S3: descarga desde AWS con streaming para no cargar en RAM
        # ------------------------------------------------------------------
        import boto3
        from src.ingestion import ensure_bucket_versioning

        bucket = Variable.get("S3_BUCKET_RAW")
        s3_prefix = Variable.get("S3_PREFIX_RAW", default_var="raw/")

        ensure_bucket_versioning(bucket)

        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")

        downloaded = []
        version_info_list = []

        for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".csv"):
                    continue
                dest = raw_dir / key.split("/")[-1]
                logger.info("Descargando s3://%s/%s → %s", bucket, key, dest)
                # Streaming: descarga en bloques de 8 MB para no cargar el
                # archivo completo en RAM antes de escribirlo al disco.
                response = s3.get_object(Bucket=bucket, Key=key)
                with open(dest, "wb") as f:
                    for block in response["Body"].iter_chunks(chunk_size=8 * 1024 * 1024):
                        f.write(block)
                downloaded.append(dest)
                version_info_list.append({
                    "key": key,
                    "version_id": response.get("VersionId", "N/A"),
                    "etag": response.get("ETag", "").strip('"'),
                })

        if not downloaded:
            raise FileNotFoundError(
                f"No se encontraron archivos CSV en s3://{bucket}/{s3_prefix}"
            )
        csv_to_consolidate = sorted(downloaded)

    # ------------------------------------------------------------------
    # Consolidación incremental: un CSV a la vez, por chunks de 50k filas.
    # Nunca se tienen >1 DataFrame en memoria al mismo tiempo.
    # ------------------------------------------------------------------
    logger.info("Consolidando %d archivos CSV...", len(csv_to_consolidate))
    if unified_path.exists():
        unified_path.unlink()  # limpiar ejecución anterior
    total_rows = _consolidate_csvs_to_disk(csv_to_consolidate, unified_path)
    logger.info("Dataset unificado guardado en: %s (%d filas)", unified_path, total_rows)

    context["ti"].xcom_push(key="data_version_info", value=version_info_list)
    context["ti"].xcom_push(key="raw_path", value=str(unified_path))


def task_preprocess(**context) -> None:
    """
    Ejecuta el pipeline de preprocesamiento y guarda los splits.
    También guarda el val crudo (antes de encoding) para que task_postprocess
    pueda pasarlo a get_groups() con los valores string originales de grp_campecs06m.
    Pushea las rutas de los splits al XCom.
    """
    import gc
    import pandas as pd
    from src.preprocessing import run_preprocessing, VALIDATION_PARTITION

    raw_path = context["ti"].xcom_pull(key="raw_path", task_ids="ingest")
    output_dir = WORKDIR / "processed"
    raw_dir = WORKDIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    val_raw_path = str(raw_dir / "df_val_raw.csv")

    # Guardar val crudo (partition == p10) ANTES del encoding para postprocesamiento.
    # Se lee solo la partición p10 con chunksize para no cargar el CSV entero en RAM.
    with pd.read_csv(raw_path, chunksize=100_000) as reader:
        chunks_val_raw = [
            chunk[chunk["partition"] == VALIDATION_PARTITION]
            for chunk in reader
        ]
    pd.concat(chunks_val_raw, ignore_index=True).to_csv(val_raw_path, index=False)
    del chunks_val_raw
    gc.collect()

    df_train, df_test, df_val, metadata = run_preprocessing(data_path=raw_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = str(output_dir / "df_train.csv")
    test_path = str(output_dir / "df_test.csv")
    val_path = str(output_dir / "df_val.csv")

    # Guardar y liberar cada split inmediatamente para no tener los 3 en RAM a la vez
    df_train.to_csv(train_path, index=False)
    del df_train
    gc.collect()

    df_test.to_csv(test_path, index=False)
    del df_test
    gc.collect()

    df_val.to_csv(val_path, index=False)
    del df_val
    gc.collect()

    context["ti"].xcom_push(key="train_path", value=train_path)
    context["ti"].xcom_push(key="test_path", value=test_path)
    context["ti"].xcom_push(key="val_path", value=val_path)
    context["ti"].xcom_push(key="val_raw_path", value=val_raw_path)
    context["ti"].xcom_push(key="preprocessing_metadata", value={
        "nan_threshold": metadata["nan_threshold"],
        "dropped_columns_count": len(metadata["dropped_columns"]),
        "dropped_columns": metadata["dropped_columns"],
        "test_size": metadata["test_size"],
        "random_state": metadata["random_state"],
        "validation_partition": metadata["validation_partition"],
        "shapes": {k: list(v) for k, v in metadata["shapes"].items()},
    })

    logger.info(
        "Preprocesamiento completado. Shapes — train=%s | test=%s | val=%s",
        metadata["shapes"]["train"],
        metadata["shapes"]["test"],
        metadata["shapes"]["val"],
    )


def task_validate_outputs(**context) -> None:
    """
    Control de calidad: verifica que los splits tienen filas y el schema esperado.
    Falla el DAG si alguna validación no pasa.
    """
    import pandas as pd

    train_path = context["ti"].xcom_pull(key="train_path", task_ids="preprocess")
    test_path = context["ti"].xcom_pull(key="test_path", task_ids="preprocess")
    val_path = context["ti"].xcom_pull(key="val_path", task_ids="preprocess")

    min_rows = {"train": 1000, "test": 300, "val": 100}
    required_cols = {"target", "partition"}  # columnas clave del schema real

    for name, path in [("train", train_path), ("test", test_path), ("val", val_path)]:
        df = pd.read_csv(path)

        if len(df) < min_rows[name]:
            raise ValueError(
                f"Split '{name}' tiene {len(df)} filas (mínimo: {min_rows[name]})."
            )

        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Split '{name}' no tiene columnas requeridas: {missing}")

        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            logger.warning("Split '%s': %d NaN totales detectados.", name, nan_count)

        logger.info(
            "Split '%s': %d filas | %d columnas | %d NaN totales",
            name, len(df), df.shape[1], nan_count,
        )

    logger.info("Validación de outputs: OK")


def task_train(**context) -> None:
    """
    Entrena el modelo con HPO (Optuna) y registra todo en MLflow.
    """
    from src.training import train_and_log_hpo

    train_path = context["ti"].xcom_pull(key="train_path", task_ids="preprocess")
    test_path = context["ti"].xcom_pull(key="test_path", task_ids="preprocess")
    val_path = context["ti"].xcom_pull(key="val_path", task_ids="preprocess")
    data_version_info = context["ti"].xcom_pull(key="data_version_info", task_ids="ingest")
    preprocessing_metadata = context["ti"].xcom_pull(
        key="preprocessing_metadata", task_ids="preprocess"
    )

    experiment_name = Variable.get("MLFLOW_EXPERIMENT_NAME", default_var="cu_venta_e2e")
    model_name = Variable.get("MLFLOW_MODEL_NAME", default_var="cu_venta_xgb")
    tracking_uri = Variable.get("MLFLOW_TRACKING_URI", default_var=None)
    n_trials = int(Variable.get("HPO_N_TRIALS", default_var="30"))

    run_id = train_and_log_hpo(
        train_path=train_path,
        test_path=test_path,
        val_path=val_path,
        n_trials=n_trials,
        experiment_name=experiment_name,
        model_name=model_name,
        data_version_info=data_version_info,
        preprocessing_params=preprocessing_metadata,
        mlflow_tracking_uri=tracking_uri,
    )

    context["ti"].xcom_push(key="mlflow_run_id", value=run_id)
    logger.info("Entrenamiento completado. MLflow run_id: %s", run_id)


def task_monitor(**context) -> None:
    """
    Monitoreo post-entrenamiento:
    - PSI de variables crudas (raw): train vs val.
    - PSI de variables preprocesadas (processed): train vs val.
    - PSI de deciles del score del modelo (train scores vs val scores).
    - AUC y Recall sobre la población de validación.
    Pushea los resultados al XCom para que check_drift los evalúe.
    """
    import mlflow
    import mlflow.xgboost
    import pandas as pd
    from src.monitoring import run_monitoring

    train_path = context["ti"].xcom_pull(key="train_path", task_ids="preprocess")
    val_path = context["ti"].xcom_pull(key="val_path", task_ids="preprocess")
    run_id = context["ti"].xcom_pull(key="mlflow_run_id", task_ids="train")

    tracking_uri = Variable.get("MLFLOW_TRACKING_URI", default_var=None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = Variable.get("MLFLOW_EXPERIMENT_NAME", default_var="cu_venta_e2e")
    mlflow.set_experiment(experiment_name)

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    id_cols = ["partition", "key_value", "codunicocli", "tip_doc", "fch_creacion", "p_fecinformacion"]
    target_col = "target"
    drop_cols = [c for c in id_cols + [target_col] if c in df_train.columns]

    X_train = df_train.drop(columns=drop_cols)
    y_train = df_train[target_col]
    X_val = df_val.drop(columns=drop_cols)
    y_val = df_val[target_col]

    # Cargar modelo desde el run de MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.xgboost.load_model(model_uri)

    train_scores = model.predict_proba(X_train)[:, 1]
    val_scores = model.predict_proba(X_val)[:, 1]

    output_dir = str(WORKDIR / "monitoring")

    with mlflow.start_run(run_id=run_id):
        results = run_monitoring(
            df_train_raw=df_train,
            df_val_raw=df_val,
            df_train_processed=df_train,
            df_val_processed=df_val,
            y_train=y_train,
            y_val=y_val,
            train_scores=train_scores,
            val_scores=val_scores,
            output_dir=output_dir,
            mlflow_active=True,
            id_cols=id_cols,
            target_col=target_col,
        )

    psi_score = results["psi_score"]
    metrics = results["model_metrics_val"]
    logger.info(
        "Monitoreo completado — PSI_score=%.4f | AUC=%.4f | Recall=%.4f",
        psi_score, metrics["auc"], metrics["recall"],
    )
    context["ti"].xcom_push(key="monitoring_results", value={
        "psi_score": psi_score,
        "val_auc": metrics["auc"],
        "val_recall": metrics["recall"],
    })


def task_check_drift(**context) -> str:
    """
    Evalúa si el PSI del score supera el umbral de deriva.

    Returns:
        'register_model' si PSI es aceptable.
        'trigger_retrain' si PSI supera el umbral (deriva severa).
    """
    monitoring_results = context["ti"].xcom_pull(
        key="monitoring_results", task_ids="monitor"
    )
    psi_score = monitoring_results.get("psi_score", 0.0)
    threshold = PSI_ALERT_THRESHOLD

    logger.info(
        "Evaluación de deriva — PSI=%.4f | Umbral=%.4f", psi_score, threshold
    )

    if psi_score > threshold:
        logger.warning(
            "DERIVA SEVERA detectada (PSI=%.4f > %.4f). "
            "Disparando re-entrenamiento automático.",
            psi_score, threshold,
        )
        return "trigger_retrain"

    logger.info("PSI dentro de límites aceptables. Continuando al registro del modelo.")
    return "register_model"


def task_register_model(**context) -> None:
    """
    Transiciona la versión más reciente del modelo registrado a 'Staging'.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = Variable.get("MLFLOW_TRACKING_URI", default_var=None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_name = Variable.get("MLFLOW_MODEL_NAME", default_var="cu_venta_xgb")
    client = MlflowClient()

    versions = client.get_latest_versions(model_name, stages=["None"])
    if not versions:
        logger.warning("No hay versiones nuevas del modelo '%s' para promover.", model_name)
        return

    latest = versions[0]
    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    logger.info(
        "Modelo '%s' v%s promovido a Staging.",
        model_name,
        latest.version,
    )


def task_postprocess(**context) -> None:
    """
    Ejecuta postprocesamiento: scoring TLV, segmentación en grupos y réplica.
    """
    import mlflow
    import mlflow.xgboost
    import pandas as pd
    from src.postprocessing import get_groups, save_replica

    val_path = context["ti"].xcom_pull(key="val_path", task_ids="preprocess")
    run_id = context["ti"].xcom_pull(key="mlflow_run_id", task_ids="train")

    tracking_uri = Variable.get("MLFLOW_TRACKING_URI", default_var=None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Cargar modelo
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.xgboost.load_model(model_uri)

    df_val = pd.read_csv(val_path)
    # val_raw_path: val crudo sin encoding — necesario para get_groups() (grp_campecs06m como string)
    val_raw_path = context["ti"].xcom_pull(key="val_raw_path", task_ids="preprocess")
    df_val_raw = pd.read_csv(val_raw_path)

    id_cols = ["partition", "key_value", "codunicocli", "tip_doc", "fch_creacion", "p_fecinformacion"]
    target_col = "target"
    drop_cols = [c for c in id_cols + [target_col] if c in df_val.columns]

    X_val = df_val.drop(columns=drop_cols)
    val_scores = model.predict_proba(X_val)[:, 1]

    # Postprocesamiento con datos CRUDOS para que get_groups() tenga grp_campecs06m original
    df_post = get_groups(val_scores, df_val_raw)

    # Derivar codmes (YYYYMM) desde p_fecinformacion (YYYYMMDD)
    codmes = str(df_val_raw["p_fecinformacion"].iloc[0] // 100)

    # Guardar resultados
    output_dir = WORKDIR / "postprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output_tlv.csv"
    df_post.to_csv(output_path, index=False)

    # Réplica pipe-delimitada
    replica_dir = WORKDIR / "replica"
    replica_dir.mkdir(parents=True, exist_ok=True)
    save_replica(
        df_post=df_post,
        table="EC_OMNICANAL",
        partition=codmes,
        dir_s3=str(replica_dir),
        dir_athena=str(replica_dir),
        dir_onpremise=str(replica_dir),
    )

    logger.info("Postprocesamiento completado. Salida: %s", output_path)
    context["ti"].xcom_push(key="postprocess_path", value=str(output_path))


# ---------------------------------------------------------------------------
# Definición del DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id="cu_venta_ml_pipeline",
    default_args=DEFAULT_ARGS,
    description=(
        "Pipeline completo ML CU Venta: "
        "ingestión S3 → preprocesamiento → HPO (Optuna) → monitoreo PSI → "
        "registro MLflow → postprocesamiento. "
        "Auto-reentrenamiento si PSI supera umbral."
    ),
    schedule_interval="@weekly",
    catchup=False,
    tags=["mlops", "cu_venta", "xgboost", "optuna", "mlflow", "monitoring"],
) as dag:

    ingest = PythonOperator(
        task_id="ingest",
        python_callable=task_ingest,
    )

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=task_preprocess,
    )

    validate_outputs = PythonOperator(
        task_id="validate_outputs",
        python_callable=task_validate_outputs,
    )

    train = PythonOperator(
        task_id="train",
        python_callable=task_train,
    )

    monitor = PythonOperator(
        task_id="monitor",
        python_callable=task_monitor,
    )

    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=task_check_drift,
    )

    # Rama 1: deriva severa → re-entrenamiento automático
    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="cu_venta_ml_pipeline",
        conf={"force_retrain": True},
        wait_for_completion=False,
    )

    # Rama 2: PSI aceptable → continuar al registro
    register_model = PythonOperator(
        task_id="register_model",
        python_callable=task_register_model,
    )

    postprocess = PythonOperator(
        task_id="postprocess",
        python_callable=task_postprocess,
        trigger_rule="none_failed_min_one_success",
    )

    # Orden del pipeline:
    # ingest → preprocess → validate_outputs → train → monitor → check_drift
    #   ├─ trigger_retrain (si PSI > umbral)
    #   └─ register_model → postprocess
    (
        ingest
        >> preprocess
        >> validate_outputs
        >> train
        >> monitor
        >> check_drift
        >> [trigger_retrain, register_model]
    )
    register_model >> postprocess
