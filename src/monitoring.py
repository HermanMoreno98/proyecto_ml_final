"""
Módulo de monitoreo — PSI (Population Stability Index) y métricas de modelo.

Calcula:
  1. PSI de variables crudas (raw): compara distribución de referencia vs actual.
  2. PSI de variables preprocesadas (processed): misma lógica post-pipeline.
  3. PSI de deciles del score del modelo (ordenado por probabilidad predicha).
  4. AUC y Recall del modelo evaluado sobre la población de validación.

Convención PSI (según instrucciones del curso):
  PSI < 0.10  → OK (sin deriva)
  PSI 0.10–0.25 → WARN (deriva moderada)
  PSI > 0.25  → ALERT (deriva severa — re-entrenamiento)

Los resultados se loguean automáticamente como métricas y artefactos en MLflow
cuando se llama a run_monitoring() dentro de un run activo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PSI Core
# ---------------------------------------------------------------------------

def compute_psi_continuous(
    reference: pd.Series,
    current: pd.Series,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> tuple[float, pd.DataFrame]:
    """
    Calcula el PSI para una variable continua usando bins definidos sobre `reference`.

    Args:
        reference: Distribución de referencia (p.ej. datos de entrenamiento).
        current:   Distribución actual (p.ej. datos recientes o de validación).
        n_bins:    Número de bins (default=10 para cuantiles).
        epsilon:   Valor mínimo para evitar log(0).

    Returns:
        (psi_value, detail_df) donde detail_df tiene columnas:
            bin, ref_count, cur_count, ref_pct, cur_pct, psi_bin
    """
    reference = reference.dropna()
    current = current.dropna()

    # Definir bins sobre referencia usando cuantiles (más robusto que bins uniformes)
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(reference, quantiles))

    # Si todos los valores son iguales → PSI = 0
    if len(bin_edges) < 2:
        return 0.0, pd.DataFrame()

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = (ref_counts / len(reference)).clip(epsilon)
    cur_pct = (cur_counts / len(current)).clip(epsilon)

    psi_bins = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    psi_value = float(psi_bins.sum())

    detail = pd.DataFrame({
        "bin": range(1, len(psi_bins) + 1),
        "ref_count": ref_counts,
        "cur_count": cur_counts,
        "ref_pct": ref_pct,
        "cur_pct": cur_pct,
        "psi_bin": psi_bins,
    })

    return psi_value, detail


def compute_psi_categorical(
    reference: pd.Series,
    current: pd.Series,
    epsilon: float = 1e-6,
) -> tuple[float, pd.DataFrame]:
    """
    Calcula el PSI para una variable categórica.

    Args:
        reference: Distribución de referencia.
        current:   Distribución actual.
        epsilon:   Valor mínimo para evitar log(0).

    Returns:
        (psi_value, detail_df) con columnas: category, ref_pct, cur_pct, psi_bin
    """
    reference = reference.dropna().astype(str)
    current = current.dropna().astype(str)

    all_cats = set(reference.unique()) | set(current.unique())

    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()

    rows = []
    for cat in sorted(all_cats):
        ref_p = max(ref_counts.get(cat, 0) / len(reference), epsilon)
        cur_p = max(cur_counts.get(cat, 0) / len(current), epsilon)
        psi_bin = (cur_p - ref_p) * np.log(cur_p / ref_p)
        rows.append({
            "category": cat,
            "ref_pct": ref_p,
            "cur_pct": cur_p,
            "psi_bin": psi_bin,
        })

    detail = pd.DataFrame(rows)
    psi_value = float(detail["psi_bin"].sum())
    return psi_value, detail


def compute_psi_score_deciles(
    reference_scores: pd.Series,
    current_scores: pd.Series,
    n_deciles: int = 10,
    epsilon: float = 1e-6,
) -> tuple[float, pd.DataFrame]:
    """
    Calcula el PSI sobre los deciles del score del modelo (probabilidad predicha).
    Los bins se definen como deciles de la distribución de referencia (train o val base).

    Args:
        reference_scores: Scores de referencia (probas del modelo).
        current_scores:   Scores de la población actual a monitorear.
        n_deciles:        Número de deciles (default=10).
        epsilon:          Valor mínimo para evitar log(0).

    Returns:
        (psi_value, detail_df) con columnas:
            decile, score_min, score_max, ref_pct, cur_pct, psi_bin
    """
    reference_scores = pd.Series(reference_scores).dropna().values
    current_scores = pd.Series(current_scores).dropna().values

    # Bins de deciles sobre referencia
    quantiles = np.linspace(0, 100, n_deciles + 1)
    bin_edges = np.unique(np.percentile(reference_scores, quantiles))

    if len(bin_edges) < 2:
        return 0.0, pd.DataFrame()

    ref_counts, edges = np.histogram(reference_scores, bins=bin_edges)
    cur_counts, _ = np.histogram(current_scores, bins=bin_edges)

    ref_pct = (ref_counts / len(reference_scores)).clip(epsilon)
    cur_pct = (cur_counts / len(current_scores)).clip(epsilon)

    psi_bins = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    psi_value = float(psi_bins.sum())

    detail = pd.DataFrame({
        "decile": range(1, len(psi_bins) + 1),
        "score_min": edges[:-1],
        "score_max": edges[1:],
        "ref_pct": ref_pct,
        "cur_pct": cur_pct,
        "psi_bin": psi_bins,
    })

    return psi_value, detail


def psi_flag(psi_value: float) -> str:
    """Retorna la etiqueta de alerta según el valor de PSI."""
    if psi_value < 0.10:
        return "OK"
    elif psi_value < 0.25:
        return "WARN"
    else:
        return "ALERT"


# ---------------------------------------------------------------------------
# Resumen de PSI para múltiples variables
# ---------------------------------------------------------------------------

def compute_dataset_psi(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    exclude_cols: list[str] | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Calcula PSI para todas las columnas de un dataset.

    Args:
        df_reference:     Dataset de referencia.
        df_current:       Dataset actual.
        categorical_cols: Lista de columnas categóricas (para PSI categórico).
                          Si None, se infiere por dtype.
        exclude_cols:     Columnas a excluir (p.ej. IDs, target).
        n_bins:           Bins para variables continuas.

    Returns:
        DataFrame con columnas: variable, psi, flag, type
        ordenado por psi descendente.
    """
    if exclude_cols is None:
        exclude_cols = []
    if categorical_cols is None:
        categorical_cols = list(df_reference.select_dtypes(include=["object", "category"]).columns)

    cols = [c for c in df_reference.columns if c in df_current.columns and c not in exclude_cols]
    results = []

    for col in cols:
        col_type = "categorical" if col in categorical_cols else "continuous"
        try:
            if col_type == "categorical":
                psi_val, _ = compute_psi_categorical(df_reference[col], df_current[col])
            else:
                psi_val, _ = compute_psi_continuous(df_reference[col], df_current[col], n_bins=n_bins)
            results.append({
                "variable": col,
                "psi": round(psi_val, 6),
                "flag": psi_flag(psi_val),
                "type": col_type,
            })
        except Exception as exc:
            logger.warning("PSI no calculado para '%s': %s", col, exc)

    return pd.DataFrame(results).sort_values("psi", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Métricas del modelo: AUC y Recall por decil de score
# ---------------------------------------------------------------------------

def compute_model_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    recall_threshold: float = 0.5,
) -> dict:
    """
    Calcula AUC y Recall del modelo.

    Args:
        y_true:            Labels verdaderos.
        y_prob:            Probabilidades predichas para la clase positiva.
        recall_threshold:  Umbral de clasificación para el recall (default=0.5).

    Returns:
        dict con: auc, recall, n_positives, n_total, positive_rate
    """
    y_pred = (y_prob >= recall_threshold).astype(int)
    auc = float(roc_auc_score(y_true, y_prob))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    return {
        "auc": round(auc, 6),
        "recall": round(recall, 6),
        "n_positives": int(y_true.sum()),
        "n_total": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 6),
        "recall_threshold": recall_threshold,
    }


def compute_recall_by_decile(
    y_true: pd.Series,
    y_prob: np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Calcula recall acumulado por decil de score (ordenado descendente).
    Permite identificar cuántos positivos se capturan al seleccionar el top-N%.

    Args:
        y_true:    Labels verdaderos (0/1).
        y_prob:    Probabilidades predichas (clase positiva).
        n_deciles: Número de deciles.

    Returns:
        DataFrame con columnas:
            decile, score_threshold, n_selected, n_positive_selected,
            recall_at_decile, lift
    """
    df = pd.DataFrame({"y_true": y_true.values, "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)

    total_positives = df["y_true"].sum()
    n = len(df)
    rows = []

    for d in range(1, n_deciles + 1):
        cutoff = int(np.ceil(n * d / n_deciles))
        subset = df.iloc[:cutoff]
        n_pos = subset["y_true"].sum()
        recall_d = n_pos / total_positives if total_positives > 0 else 0
        # Lift = recall / fracción de población seleccionada
        pop_frac = cutoff / n
        lift = recall_d / pop_frac if pop_frac > 0 else 0

        rows.append({
            "decile": d,
            "score_threshold": float(subset["y_prob"].min()),
            "n_selected": cutoff,
            "n_positive_selected": int(n_pos),
            "recall_at_decile": round(recall_d, 6),
            "lift": round(lift, 6),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Runner principal — integra todo y logea en MLflow
# ---------------------------------------------------------------------------

def run_monitoring(
    df_train_raw: pd.DataFrame,
    df_val_raw: pd.DataFrame,
    df_train_processed: pd.DataFrame,
    df_val_processed: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    output_dir: str | Path = "/tmp/monitoring",
    mlflow_active: bool = True,
    id_cols: list[str] | None = None,
    target_col: str = "target",
    recall_threshold: float = 0.5,
) -> dict:
    """
    Ejecuta todos los monitoreos y opcionalmente logea en MLflow.

    Args:
        df_train_raw:       Dataset crudo de referencia (train original).
        df_val_raw:         Dataset crudo actual (val/producción).
        df_train_processed: Dataset procesado de referencia.
        df_val_processed:   Dataset procesado actual.
        y_train:            Labels de entrenamiento.
        y_val:              Labels de validación.
        train_scores:       Probabilidades del modelo sobre train (referencia PSI score).
        val_scores:         Probabilidades del modelo sobre val (actual PSI score).
        output_dir:         Directorio donde guardar CSVs de detalle.
        mlflow_active:      Si True, logea métricas y artefactos en el run activo de MLflow.
        id_cols:            Columnas de ID a excluir del PSI.
        target_col:         Nombre de la columna target.
        recall_threshold:   Umbral para calcular recall.

    Returns:
        dict con todos los resultados: psi_raw_summary, psi_processed_summary,
        psi_score, model_metrics_val, recall_by_decile.
    """
    import os
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if id_cols is None:
        id_cols = ["p_codmes", "key_value"]
    exclude = id_cols + [target_col]

    results = {}

    # ---- 1. PSI variables crudas ----
    logger.info("Calculando PSI de variables crudas (raw)...")
    psi_raw = compute_dataset_psi(
        df_reference=df_train_raw,
        df_current=df_val_raw,
        exclude_cols=exclude,
    )
    raw_path = output_dir / "psi_raw_variables.csv"
    psi_raw.to_csv(raw_path, index=False)
    results["psi_raw_summary"] = psi_raw

    n_alert_raw = (psi_raw["flag"] == "ALERT").sum()
    n_warn_raw = (psi_raw["flag"] == "WARN").sum()
    logger.info("PSI raw — ALERT: %d | WARN: %d | OK: %d", n_alert_raw, n_warn_raw,
                len(psi_raw) - n_alert_raw - n_warn_raw)

    # ---- 2. PSI variables preprocesadas ----
    logger.info("Calculando PSI de variables preprocesadas...")
    psi_proc = compute_dataset_psi(
        df_reference=df_train_processed,
        df_current=df_val_processed,
        exclude_cols=exclude,
    )
    proc_path = output_dir / "psi_processed_variables.csv"
    psi_proc.to_csv(proc_path, index=False)
    results["psi_processed_summary"] = psi_proc

    n_alert_proc = (psi_proc["flag"] == "ALERT").sum()
    n_warn_proc = (psi_proc["flag"] == "WARN").sum()
    logger.info("PSI processed — ALERT: %d | WARN: %d | OK: %d", n_alert_proc, n_warn_proc,
                len(psi_proc) - n_alert_proc - n_warn_proc)

    # ---- 3. PSI deciles del score ----
    logger.info("Calculando PSI de deciles del score...")
    psi_score_val, psi_score_detail = compute_psi_score_deciles(
        reference_scores=pd.Series(train_scores),
        current_scores=pd.Series(val_scores),
    )
    score_path = output_dir / "psi_score_deciles.csv"
    psi_score_detail.to_csv(score_path, index=False)
    results["psi_score"] = psi_score_val
    results["psi_score_detail"] = psi_score_detail
    logger.info("PSI score deciles: %.6f → %s", psi_score_val, psi_flag(psi_score_val))

    # ---- 4. Métricas del modelo (val) ----
    logger.info("Calculando AUC y Recall sobre validación...")
    model_metrics = compute_model_metrics(y_val, val_scores, recall_threshold)
    results["model_metrics_val"] = model_metrics
    logger.info(
        "Métricas val — AUC=%.4f | Recall@%.2f=%.4f",
        model_metrics["auc"],
        recall_threshold,
        model_metrics["recall"],
    )

    # ---- 5. Recall por decil ----
    recall_decile_df = compute_recall_by_decile(y_val, val_scores)
    recall_path = output_dir / "recall_by_decile.csv"
    recall_decile_df.to_csv(recall_path, index=False)
    results["recall_by_decile"] = recall_decile_df

    # ---- MLflow logging ----
    if mlflow_active:
        try:
            import mlflow
            # Métricas escalares
            mlflow.log_metrics({
                "monitor.psi_score_deciles": psi_score_val,
                "monitor.psi_score_flag": {"OK": 0, "WARN": 1, "ALERT": 2}.get(psi_flag(psi_score_val), -1),
                "monitor.n_raw_alert": int(n_alert_raw),
                "monitor.n_raw_warn": int(n_warn_raw),
                "monitor.n_proc_alert": int(n_alert_proc),
                "monitor.n_proc_warn": int(n_warn_proc),
                "monitor.val_auc": model_metrics["auc"],
                "monitor.val_recall": model_metrics["recall"],
                "monitor.val_positive_rate": model_metrics["positive_rate"],
            })
            # Artefactos
            mlflow.log_artifact(str(raw_path), artifact_path="monitoring")
            mlflow.log_artifact(str(proc_path), artifact_path="monitoring")
            mlflow.log_artifact(str(score_path), artifact_path="monitoring")
            mlflow.log_artifact(str(recall_path), artifact_path="monitoring")

            logger.info("Resultados de monitoreo registrados en MLflow.")
        except Exception as exc:
            logger.warning("No se pudo logear en MLflow: %s", exc)

    return results
