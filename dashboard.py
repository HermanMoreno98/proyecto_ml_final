"""
Dashboard interactivo — Pipeline ML CU Venta

Muestra:
  1. Distribución de grupos de ejecución TLV
  2. Evolución de AUC y PSI por mes (tracking MLflow)
  3. Tabla Top-N clientes por puntuación TLV
  4. Efectividad % y monto promedio por grupo y mes

Ejecutar:
    streamlit run dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CU Venta — Dashboard ML",
    page_icon="📊",
    layout="wide",
)

ROOT = Path(__file__).parent
DIR_POSTPROCESSED = ROOT / "data" / "postprocessed"
DIR_MONITORING    = ROOT / "data" / "monitoring"

PARTITION_LABEL = {f"p{i}": f"p{i}" for i in range(1, 11)}  # p1..p10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def load_output_tlv() -> pd.DataFrame | None:
    """Carga el CSV de scoring TLV más reciente disponible."""
    files = sorted(DIR_POSTPROCESSED.glob("output_tlv*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    return pd.read_csv(files[0])


@st.cache_data(ttl=120)
def load_metrics_by_month() -> pd.DataFrame | None:
    """Carga métricas AUC y PSI por partición (mes) generadas por stage_monitor."""
    path = DIR_MONITORING / "metrics_by_month.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    order = {f"p{i}": i for i in range(1, 11)}
    df["_order"] = df["partition"].map(order).fillna(99)
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


@st.cache_data(ttl=60)
def load_mlflow_runs() -> pd.DataFrame:
    """
    Lee los runs del experimento desde MLflow.
    Devuelve columnas normalizadas: start_time, auc, psi_score, run_name.
    """
    try:
        import os
        import mlflow
        # Usa la variable de entorno (http://mlflow:5000 en Docker,
        # http://localhost:5001 en local) y el hardcode solo como último fallback.
        tracking_uri = (
            os.environ.get("MLFLOW_TRACKING_URI")
            or "http://localhost:5001"
        )
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        all_exps = client.search_experiments()
        if not all_exps:
            return pd.DataFrame()

        rows = []
        for exp in all_exps:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time ASC"],
                max_results=200,
            )
            for r in runs:
                m = r.data.metrics
                row = {
                    "run_id":     r.info.run_id,
                    "run_name":   r.info.run_name or r.info.run_id[:8],
                    "start_time": pd.to_datetime(r.info.start_time, unit="ms"),
                    "status":     r.info.status,
                }
                for key in ("monitor.val_auc", "final.val.auc", "val.auc", "final.test.auc", "test.auc"):
                    if key in m and 0.0 <= m[key] <= 1.0:
                        row.setdefault("auc", round(m[key], 4))
                        break
                for key in ("monitor.psi_score_deciles", "val.psi_score", "psi_score"):
                    if key in m and 0.0 <= m[key] <= 2.0:
                        row.setdefault("psi_score", round(m[key], 6))
                        break
                rows.append(row)

        df = pd.DataFrame(rows)
        df = df[df[["auc", "psi_score"]].notna().any(axis=1)].copy()
        df["fecha"] = df["start_time"].dt.strftime("%d/%m %H:%M")
        return df.sort_values("start_time").reset_index(drop=True)
    except Exception as exc:
        st.warning(f"No se pudo conectar con MLflow: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------

st.title("📊 Dashboard — CU Venta ML Pipeline")
st.caption("Monitoreo de deriva, scoring TLV y segmentación de clientes")

df = load_output_tlv()
df_months = load_metrics_by_month()
df_runs = load_mlflow_runs()

if df is None:
    st.error(
        "No se encontró ningún archivo `output_tlv*.csv` en `data/postprocessed/`. "
        "Ejecuta primero `python main.py --mode train` para generar los datos."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — filtros
# ---------------------------------------------------------------------------

st.sidebar.header("Filtros")

# Selector de periodo si hay múltiples archivos
all_files = sorted(DIR_POSTPROCESSED.glob("output_tlv*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
file_labels = [p.name for p in all_files]
selected_file = st.sidebar.selectbox("Archivo de scoring", file_labels, index=0)
if selected_file != file_labels[0]:
    df = pd.read_csv(DIR_POSTPROCESSED / selected_file)

top_n = st.sidebar.slider("Top-N clientes", min_value=10, max_value=500, value=100, step=10)

grupos_disponibles = sorted(df["grupo_ejec_tlv"].unique())
grupos_sel = st.sidebar.multiselect(
    "Grupos de ejecución a mostrar",
    options=grupos_disponibles,
    default=grupos_disponibles,
)

df_filtered = df[df["grupo_ejec_tlv"].isin(grupos_sel)]

# ---------------------------------------------------------------------------
# Métricas resumen
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total clientes", f"{len(df_filtered):,}")
col2.metric("Tasa de conversión", f"{df_filtered['target'].mean()*100:.2f}%" if "target" in df_filtered.columns else "N/D")
col3.metric("Score promedio", f"{df_filtered['prob'].mean():.4f}" if "prob" in df_filtered.columns else "N/D")
col4.metric("TLV promedio", f"{df_filtered['puntuacion_tlv'].mean():.6f}" if "puntuacion_tlv" in df_filtered.columns else "N/D")

st.divider()

# ---------------------------------------------------------------------------
# Bloque 1: Distribución de grupos de ejecución
# ---------------------------------------------------------------------------

st.subheader("1. Distribución de grupos de ejecución TLV")

dist = (
    df_filtered.groupby("grupo_ejec_tlv")
    .size()
    .reset_index(name="clientes")
    .sort_values("grupo_ejec_tlv")
)
dist["porcentaje"] = (dist["clientes"] / dist["clientes"].sum() * 100).round(2)

col_a, col_b = st.columns([2, 1])
with col_a:
    st.bar_chart(dist.set_index("grupo_ejec_tlv")["clientes"])
with col_b:
    st.dataframe(
        dist.rename(columns={"grupo_ejec_tlv": "Grupo", "clientes": "Clientes", "porcentaje": "%"}),
        width="stretch",
        hide_index=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Bloque 2: Evolución AUC y PSI por mes (MLflow)
# ---------------------------------------------------------------------------

st.subheader("2. Evolución de AUC y PSI por mes")

if df_months is None:
    st.info(
        "No se encontró `data/monitoring/metrics_by_month.csv`. "
        "Ejecuta `python main.py --mode train` para generarlo."
    )
else:
    col_c, col_d = st.columns(2)

    with col_c:
        st.write("**AUC por mes (partición)**")
        chart = df_months.dropna(subset=["auc"]).set_index("partition")[["auc"]].rename(columns={"auc": "AUC"})
        st.line_chart(chart)

    with col_d:
        st.write("**PSI por mes (vs. p1 como referencia)**")
        chart = df_months.dropna(subset=["psi_score"]).set_index("partition")[["psi_score"]].rename(columns={"psi_score": "PSI"})
        st.line_chart(chart)
        st.markdown(
            "<span style='color:#f0ad4e'>▬ WARN &gt; 0.10</span> &nbsp;&nbsp; "
            "<span style='color:#d9534f'>▬ ALERT &gt; 0.25</span>",
            unsafe_allow_html=True,
        )

    def highlight_psi(row):
        psi = row.get("PSI") or 0
        if psi > 0.25:
            return ["background-color: #5c1a1a"] * len(row)
        elif psi > 0.10:
            return ["background-color: #5c4a00"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_months[["partition", "auc", "psi_score", "n"]]
        .rename(columns={"partition": "Mes", "auc": "AUC", "psi_score": "PSI", "n": "Registros"})
        .style.apply(highlight_psi, axis=1),
        width="stretch",
        hide_index=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Bloque 3: Top-N clientes por puntuación TLV
# ---------------------------------------------------------------------------

st.subheader(f"3. Top-{top_n} clientes por puntuación TLV")

id_display_cols = [c for c in ["codunicocli", "key_value", "partition", "monto", "prob_value_contact",
                                "grp_campecs06m", "prob", "puntuacion_tlv", "grupo_ejec_tlv"] if c in df_filtered.columns]
top_df = (
    df_filtered[id_display_cols]
    .sort_values("puntuacion_tlv", ascending=False)
    .head(top_n)
    .reset_index(drop=True)
)
top_df.index = top_df.index + 1  # ranking desde 1

st.dataframe(
    top_df.style.background_gradient(subset=["puntuacion_tlv"], cmap="YlOrRd"),
    width="stretch",
)

col_dl, _ = st.columns([1, 3])
with col_dl:
    csv_bytes = top_df.to_csv(index=True).encode("utf-8")
    st.download_button(
        label=f"Descargar Top-{top_n} (CSV)",
        data=csv_bytes,
        file_name=f"top{top_n}_clientes_tlv.csv",
        mime="text/csv",
    )

st.divider()

# ---------------------------------------------------------------------------
# Bloque 4: Efectividad % y monto promedio por grupo
# ---------------------------------------------------------------------------

st.subheader("4. Efectividad y monto promedio por grupo de ejecución")

agg: dict[str, str] = {"monto": "mean", "prob": "mean"}
if "target" in df_filtered.columns:
    agg["target"] = "mean"

efec = (
    df_filtered.groupby("grupo_ejec_tlv")
    .agg(agg)
    .reset_index()
    .sort_values("grupo_ejec_tlv")
)
efec = efec.rename(columns={
    "grupo_ejec_tlv": "Grupo",
    "monto": "Monto promedio (S/)",
    "prob": "Score promedio",
})
if "target" in efec.columns:
    efec = efec.rename(columns={"target": "Tasa de conversión"})
    efec["Tasa de conversión"] = (efec["Tasa de conversión"] * 100).round(2)

col_e, col_f = st.columns(2)
with col_e:
    st.write("**Monto promedio por grupo**")
    st.bar_chart(efec.set_index("Grupo")["Monto promedio (S/)"])

with col_f:
    if "Tasa de conversión" in efec.columns:
        st.write("**Tasa de conversión (%) por grupo**")
        st.bar_chart(efec.set_index("Grupo")["Tasa de conversión"])
    else:
        st.write("**Score promedio por grupo**")
        st.bar_chart(efec.set_index("Grupo")["Score promedio"])

st.dataframe(efec.set_index("Grupo"), width="stretch")

# ---------------------------------------------------------------------------
# Bloque 5: Curva de recall por decil (bonus)
# ---------------------------------------------------------------------------

if "target" in df.columns and "prob" in df.columns:
    with st.expander("📈 Curva de recall acumulado por decil (Lift)"):
        import numpy as np
        df_decil = df.sort_values("prob", ascending=False).reset_index(drop=True)
        total_pos = df_decil["target"].sum()
        n = len(df_decil)
        deciles = []
        for d in range(1, 11):
            cutoff = int(np.ceil(n * d / 10))
            subset = df_decil.iloc[:cutoff]
            n_pos = subset["target"].sum()
            recall_d = n_pos / total_pos if total_pos > 0 else 0
            lift = recall_d / (d / 10)
            deciles.append({"Decil": d, "Recall_pct": round(recall_d * 100, 2), "Lift": round(lift, 3)})
        df_decil_chart = pd.DataFrame(deciles).set_index("Decil")
        col_g, col_h = st.columns(2)
        with col_g:
            st.write("**Recall acumulado por decil (%)**")
            st.line_chart(df_decil_chart[["Recall_pct"]])
        with col_h:
            st.write("**Lift por decil**")
            st.line_chart(df_decil_chart[["Lift"]])
        st.dataframe(
            df_decil_chart.rename(columns={"Recall_pct": "Recall acum. (%)"}),
            width="stretch",
        )
