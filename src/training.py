"""
Módulo de entrenamiento con HPO (Optuna) y tracking completo en MLflow.

Dos modos de entrenamiento:
  1. train_and_log()      — Entrenamiento con parámetros fijos (fiel al notebook).
  2. train_and_log_hpo()  — Búsqueda de hiperparámetros con Optuna + registro MLflow.

Registra:
- Parámetros del modelo y del pipeline de preprocesamiento.
- Métricas de evaluación (AUC, accuracy, precision, recall, F1, logloss).
- Artefactos: modelo serializado, importancia de variables, datasets procesados.
- Trazabilidad de datos: S3 version_id del dataset de entrada.
- Modelo en el MLflow Model Registry para versionado y promoción a producción.

Entry points:
    # Entrenamiento estándar
    python -m src.training --train data/processed/df_train.csv \\
        --test data/processed/df_test.csv --val data/processed/df_val.csv

    # Entrenamiento con HPO (Optuna)
    python -m src.training --train data/processed/df_train.csv \\
        --test data/processed/df_test.csv --val data/processed/df_val.csv \\
        --hpo --n-trials 30
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

# Silenciar logs verbose de Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

ID_COLS = ["partition", "key_value", "codunicocli", "tip_doc", "fch_creacion", "p_fecinformacion"]
TARGET_COL = "target"

# Hiperparámetros por defecto (fiel al notebook)
DEFAULT_PARAMS: dict = {
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": 42,
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def load_split(train_path: str, test_path: str, val_path: str) -> tuple:
    """Carga los tres splits generados por preprocessing.py."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_val = pd.read_csv(val_path)
    logger.info(
        "Splits cargados — train: %d | test: %d | val: %d",
        len(df_train), len(df_test), len(df_val),
    )
    return df_train, df_test, df_val


def _xy_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features de target."""
    drop_cols = [c for c in ID_COLS + [TARGET_COL] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL]
    return X, y


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calcula métricas de clasificación binaria."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "log_loss": log_loss(y_true, y_prob),
    }


def _log_feature_importance(model: xgb.XGBClassifier, feature_names: list[str]) -> None:
    """Registra importancia de variables como artefacto en MLflow."""
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_path = "/tmp/feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    mlflow.log_artifact(importance_path, artifact_path="feature_importance")


def _register_model(
    run_id: str,
    model_name: str,
    target_stage: str = "Staging",
) -> None:
    """
    Promueve la última versión del modelo registrado al stage indicado.

    Args:
        run_id:       Run ID del experimento MLflow.
        model_name:   Nombre en el MLflow Model Registry.
        target_stage: Stage destino ('Staging' o 'Production').
    """
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=["None"])
        if not versions:
            logger.warning(
                "No se encontraron versiones nuevas de '%s' para promover.", model_name
            )
            return

        latest = versions[0]
        client.transition_model_version_stage(
            name=model_name,
            version=latest.version,
            stage=target_stage,
            archive_existing_versions=True,
        )
        logger.info(
            "Modelo '%s' v%s promovido a '%s'.",
            model_name, latest.version, target_stage,
        )
    except Exception as exc:
        logger.warning("No se pudo promover el modelo al registry: %s", exc)


# ---------------------------------------------------------------------------
# Entrenamiento estándar
# ---------------------------------------------------------------------------

def train_and_log(
    train_path: str,
    test_path: str,
    val_path: str,
    experiment_name: str = "cu_venta_preprocessing",
    model_name: str = "cu_venta_xgb",
    params: dict | None = None,
    data_version_info: dict | None = None,
    preprocessing_params: dict | None = None,
    mlflow_tracking_uri: str | None = None,
) -> str:
    """
    Entrena XGBClassifier con parámetros fijos y registra TODO en MLflow.

    Args:
        train_path:           CSV de entrenamiento (salida de preprocessing).
        test_path:            CSV de test.
        val_path:             CSV de validación.
        experiment_name:      Nombre del experimento MLflow.
        model_name:           Nombre en el MLflow Model Registry.
        params:               Hiperparámetros del modelo (overrides DEFAULT_PARAMS).
        data_version_info:    Metadata de S3 (bucket, key, version_id, etag).
        preprocessing_params: Parámetros usados en el preprocesamiento.
        mlflow_tracking_uri:  URI del servidor MLflow (None = usa MLFLOW_TRACKING_URI).

    Returns:
        run_id del experimento MLflow.
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment(experiment_name)
    model_params = {**DEFAULT_PARAMS, **(params or {})}

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run iniciado: %s", run_id)

        # Trazabilidad de datos
        if data_version_info:
            mlflow.set_tags({
                "data.s3_bucket": data_version_info.get("bucket", ""),
                "data.s3_key": data_version_info.get("key", ""),
                "data.s3_version_id": data_version_info.get("version_id", ""),
                "data.s3_etag": data_version_info.get("etag", ""),
                "training.mode": "standard",
            })

        # Parámetros de preprocesamiento
        if preprocessing_params:
            mlflow.log_params({
                f"prep.{k}": v
                for k, v in preprocessing_params.items()
                if not isinstance(v, (list, dict))
            })

        mlflow.log_params(model_params)

        # Carga de datos
        df_train, df_test, df_val = load_split(train_path, test_path, val_path)
        X_train, y_train = _xy_split(df_train)
        X_test, y_test = _xy_split(df_test)
        X_val, y_val = _xy_split(df_val)

        mlflow.log_params({
            "data.train_rows": len(df_train),
            "data.test_rows": len(df_test),
            "data.val_rows": len(df_val),
            "data.n_features": X_train.shape[1],
        })

        # Entrenamiento
        logger.info("Entrenando XGBClassifier (modo estándar)...")
        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Métricas test
        y_pred_test = model.predict(X_test)
        y_prob_test = model.predict_proba(X_test)[:, 1]
        metrics_test = compute_metrics(y_test, y_pred_test, y_prob_test)
        mlflow.log_metrics({f"test.{k}": v for k, v in metrics_test.items()})

        # Métricas validación
        y_pred_val = model.predict(X_val)
        y_prob_val = model.predict_proba(X_val)[:, 1]
        metrics_val = compute_metrics(y_val, y_pred_val, y_prob_val)
        mlflow.log_metrics({f"val.{k}": v for k, v in metrics_val.items()})

        logger.info(
            "Test  → AUC=%.4f | F1=%.4f | Acc=%.4f",
            metrics_test["roc_auc"], metrics_test["f1"], metrics_test["accuracy"],
        )
        logger.info(
            "Val   → AUC=%.4f | F1=%.4f | Acc=%.4f",
            metrics_val["roc_auc"], metrics_val["f1"], metrics_val["accuracy"],
        )

        # Monitoreo PSI integrado
        try:
            from src.monitoring import run_monitoring
            run_monitoring(
                df_train_raw=df_train,
                df_val_raw=df_val,
                df_train_processed=df_train,
                df_val_processed=df_val,
                y_train=y_train,
                y_val=y_val,
                train_scores=model.predict_proba(X_train)[:, 1],
                val_scores=y_prob_val,
                output_dir="/tmp/monitoring",
                mlflow_active=True,
            )
        except Exception as exc:
            logger.warning("Monitoreo PSI falló (no bloquea el pipeline): %s", exc)

        # Importancia de variables
        _log_feature_importance(model, list(X_train.columns))

        # Registro en Model Registry
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name,
        )
        logger.info("Modelo registrado en MLflow Model Registry: '%s'", model_name)

    _register_model(run_id, model_name, target_stage="Staging")
    return run_id


# ---------------------------------------------------------------------------
# Entrenamiento con HPO (Optuna)
# ---------------------------------------------------------------------------

def train_and_log_hpo(
    train_path: str,
    test_path: str,
    val_path: str,
    n_trials: int = 30,
    experiment_name: str = "cu_venta_e2e",
    model_name: str = "cu_venta_xgb",
    data_version_info: dict | None = None,
    preprocessing_params: dict | None = None,
    mlflow_tracking_uri: str | None = None,
) -> str:
    """
    Busca hiperparámetros con Optuna y registra el mejor modelo en MLflow.

    Cada trial de Optuna se registra como un child run dentro del parent run
    principal, permitiendo comparar todos los trials en la UI de MLflow.

    Args:
        train_path:           CSV de entrenamiento.
        test_path:            CSV de test.
        val_path:             CSV de validación.
        n_trials:             Número de trials de Optuna.
        experiment_name:      Nombre del experimento MLflow.
        model_name:           Nombre en el MLflow Model Registry.
        data_version_info:    Metadata de S3 para trazabilidad.
        preprocessing_params: Parámetros de preprocesamiento.
        mlflow_tracking_uri:  URI del servidor MLflow.

    Returns:
        run_id del parent run en MLflow.
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment(experiment_name)

    # Cargar datos una sola vez (fuera del loop de trials para eficiencia)
    df_train, df_test, df_val = load_split(train_path, test_path, val_path)
    X_train, y_train = _xy_split(df_train)
    X_test, y_test = _xy_split(df_test)
    X_val, y_val = _xy_split(df_val)

    logger.info("Iniciando búsqueda HPO con Optuna (%d trials)...", n_trials)

    # Parent run que agrupa todos los trials
    with mlflow.start_run(run_name="hpo_parent") as parent_run:
        parent_run_id = parent_run.info.run_id

        # Trazabilidad de datos
        if data_version_info:
            # data_version_info puede ser una lista (un dict por archivo S3)
            # o un dict único; normalizamos siempre a lista
            info_list = data_version_info if isinstance(data_version_info, list) else [data_version_info]
            first = info_list[0] if info_list else {}
            mlflow.set_tags({
                "data.s3_bucket": first.get("bucket", first.get("key", "")),
                "data.s3_key": first.get("key", ""),
                "data.s3_version_id": first.get("version_id", ""),
                "data.num_files": str(len(info_list)),
                "training.mode": "hpo_optuna",
                "hpo.n_trials": str(n_trials),
            })

        if preprocessing_params:
            mlflow.log_params({
                f"prep.{k}": v
                for k, v in preprocessing_params.items()
                if not isinstance(v, (list, dict))
            })

        mlflow.log_params({
            "data.train_rows": len(df_train),
            "data.test_rows": len(df_test),
            "data.val_rows": len(df_val),
            "data.n_features": X_train.shape[1],
            "hpo.n_trials": n_trials,
            "hpo.direction": "maximize",
            "hpo.metric": "test_roc_auc",
        })

        # Función objetivo Optuna — cada trial es un child MLflow run
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": 42,
            }

            with mlflow.start_run(
                run_name=f"trial_{trial.number}",
                nested=True,
            ):
                mlflow.log_params({k: v for k, v in params.items()
                                   if k not in ("use_label_encoder", "eval_metric")})
                mlflow.set_tag("trial.number", trial.number)

                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

                y_prob_test = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob_test)
                mlflow.log_metrics({"test.roc_auc": auc})

            return auc

        # Ejecutar estudio Optuna
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = {
            **study.best_params,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
        }

        logger.info(
            "Mejor trial: #%d | AUC=%.4f | Params: %s",
            study.best_trial.number,
            study.best_value,
            study.best_params,
        )

        # Registrar mejores params en el parent run
        mlflow.log_params({f"best.{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best.test_roc_auc", study.best_value)
        mlflow.set_tag("hpo.best_trial", study.best_trial.number)

        # Entrenar modelo final con los mejores parámetros
        logger.info("Entrenando modelo final con mejores hiperparámetros...")
        best_model = xgb.XGBClassifier(**best_params)
        best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Métricas finales — test
        y_pred_test = best_model.predict(X_test)
        y_prob_test = best_model.predict_proba(X_test)[:, 1]
        metrics_test = compute_metrics(y_test, y_pred_test, y_prob_test)
        mlflow.log_metrics({f"final.test.{k}": v for k, v in metrics_test.items()})

        # Métricas finales — validación
        y_pred_val = best_model.predict(X_val)
        y_prob_val = best_model.predict_proba(X_val)[:, 1]
        metrics_val = compute_metrics(y_val, y_pred_val, y_prob_val)
        mlflow.log_metrics({f"final.val.{k}": v for k, v in metrics_val.items()})

        logger.info(
            "Modelo final — Test AUC=%.4f | Val AUC=%.4f",
            metrics_test["roc_auc"], metrics_val["roc_auc"],
        )

        # Monitoreo PSI integrado
        try:
            from src.monitoring import run_monitoring
            run_monitoring(
                df_train_raw=df_train,
                df_val_raw=df_val,
                df_train_processed=df_train,
                df_val_processed=df_val,
                y_train=y_train,
                y_val=y_val,
                train_scores=best_model.predict_proba(X_train)[:, 1],
                val_scores=y_prob_val,
                output_dir="/tmp/monitoring",
                mlflow_active=True,
            )
        except Exception as exc:
            logger.warning("Monitoreo PSI falló (no bloquea el pipeline): %s", exc)

        # Importancia de variables
        _log_feature_importance(best_model, list(X_train.columns))

        # Registro en Model Registry
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.xgboost.log_model(
            best_model,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name,
        )
        logger.info("Mejor modelo registrado en MLflow Model Registry: '%s'", model_name)

    _register_model(parent_run_id, model_name, target_stage="Staging")
    return parent_run_id


# ---------------------------------------------------------------------------
# Entry point CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento XGBoost con MLflow. Soporta modo estándar y HPO (Optuna)."
    )
    parser.add_argument("--train", required=True, help="CSV de entrenamiento")
    parser.add_argument("--test", required=True, help="CSV de test")
    parser.add_argument("--val", required=True, help="CSV de validación")
    parser.add_argument(
        "--hpo", action="store_true",
        help="Activar búsqueda de hiperparámetros con Optuna",
    )
    parser.add_argument(
        "--n-trials", type=int, default=30,
        help="Número de trials de Optuna (solo con --hpo)",
    )
    parser.add_argument(
        "--experiment-name", default="cu_venta_preprocessing",
        help="Nombre del experimento MLflow",
    )
    parser.add_argument(
        "--model-name", default="cu_venta_xgb",
        help="Nombre en el MLflow Model Registry",
    )
    parser.add_argument(
        "--mlflow-uri", default=None,
        help="URI del servidor MLflow (default: usa MLFLOW_TRACKING_URI del entorno)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.hpo:
        run_id = train_and_log_hpo(
            train_path=args.train,
            test_path=args.test,
            val_path=args.val,
            n_trials=args.n_trials,
            experiment_name=args.experiment_name,
            model_name=args.model_name,
            mlflow_tracking_uri=args.mlflow_uri,
        )
    else:
        run_id = train_and_log(
            train_path=args.train,
            test_path=args.test,
            val_path=args.val,
            experiment_name=args.experiment_name,
            model_name=args.model_name,
            mlflow_tracking_uri=args.mlflow_uri,
        )

    logger.info("Pipeline de entrenamiento completado. Run ID: %s", run_id)


if __name__ == "__main__":
    main()
