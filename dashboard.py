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
        import altair as alt
        chart_data = df_months.dropna(subset=["auc"])[["partition", "auc"]].rename(columns={"auc": "AUC"})
        auc_chart = (
            alt.Chart(chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("partition:O", title="Partición"),
                y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0.8, 0.9]), title="AUC"),
                tooltip=["partition", "AUC"],
            )
            .properties(height=300)
        )
        st.altair_chart(auc_chart, use_container_width=True)

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

st.divider()

# ---------------------------------------------------------------------------
# Bloque 5: Análisis de segmentación de clientes
# ---------------------------------------------------------------------------

segment_columns = [
    c for c in ["partition", "grp_campecs06m", "canal", "zona", "edad", "grupo_ejec_tlv"]
    if c in df_filtered.columns
]
if segment_columns:
    segment = st.selectbox("Segmento para análisis", segment_columns, index=0)
    agg_fields: dict[str, tuple[str, str]] = {}
    if "monto" in df_filtered.columns:
        agg_fields["Monto promedio (S/)"] = ("monto", "mean")
    if "prob" in df_filtered.columns:
        agg_fields["Score promedio"] = ("prob", "mean")
    if "target" in df_filtered.columns:
        agg_fields["Tasa de conversión (%)"] = ("target", "mean")

    count_col = "codunicocli" if "codunicocli" in df_filtered.columns else "prob"
    seg_df = (
        df_filtered.groupby(segment)
        .agg(Clientes=(count_col, "nunique" if count_col == "codunicocli" else "count"), **agg_fields)
        .sort_values("Clientes", ascending=False)
        .reset_index()
    )

    if "Tasa de conversión (%)" in seg_df.columns:
        seg_df["Tasa de conversión (%)"] = (seg_df["Tasa de conversión (%)"] * 100).round(2)
    if "Monto promedio (S/)" in seg_df.columns:
        seg_df["Monto promedio (S/)"] = seg_df["Monto promedio (S/)"] .round(2)
    if "Score promedio" in seg_df.columns:
        seg_df["Score promedio"] = seg_df["Score promedio"].round(4)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if "Monto promedio (S/)" in seg_df.columns:
            st.write("**Monto promedio por segmento**")
            st.bar_chart(seg_df.set_index(segment)["Monto promedio (S/)"])
        elif "Score promedio" in seg_df.columns:
            st.write("**Score promedio por segmento**")
            st.bar_chart(seg_df.set_index(segment)["Score promedio"])
    with col_s2:
        if "Tasa de conversión (%)" in seg_df.columns:
            st.write("**Tasa de conversión (%) por segmento**")
            st.bar_chart(seg_df.set_index(segment)["Tasa de conversión (%)"])
        else:
            st.write("**Tamaño de segmento**")
            st.bar_chart(seg_df.set_index(segment)["Clientes"])

    st.dataframe(seg_df.set_index(segment), width="stretch", hide_index=False)
else:
    st.info("No hay columnas de segmento reconocidas para análisis dinámico.")

st.divider()

# ---------------------------------------------------------------------------
# Bloque 6: Explicabilidad y ranking de prioridades
# ---------------------------------------------------------------------------

st.subheader("6. Explicabilidad y ranking de prioridades")
if "prob" in df_filtered.columns:
    prob_cutoff = st.slider(
        "Probabilidad mínima para cliente prioritario",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )
    monto_cutoff = 0.0
    if "monto" in df_filtered.columns:
        monto_cutoff = st.number_input(
            "Monto mínimo para cliente prioritario",
            min_value=0.0,
            value=float(df_filtered["monto"].quantile(0.75)),
            step=100.0,
            format="%.2f",
        )

    rule_df = df_filtered[df_filtered["prob"] >= prob_cutoff]
    if "monto" in rule_df.columns:
        rule_df = rule_df[rule_df["monto"] >= monto_cutoff]

    summary = {
        "Clientes priorizados": len(rule_df),
        "Porcentaje total": f"{len(rule_df) / len(df_filtered) * 100:.2f}%" if len(df_filtered) else "N/D",
    }
    if "monto" in rule_df.columns and len(rule_df):
        summary["Monto total (S/)"] = round(rule_df["monto"].sum(), 2)
        summary["Monto promedio (S/)"] = round(rule_df["monto"].mean(), 2)
    if "puntuacion_tlv" in rule_df.columns and len(rule_df):
        summary["TLV promedio"] = round(rule_df["puntuacion_tlv"].mean(), 4)

    cols_summary = st.columns(len(summary))
    for col, (label, value) in zip(cols_summary, summary.items()):
        col.metric(label, value)

    display_cols = [
        c for c in ["codunicocli", "partition", "grp_campecs06m", "grupo_ejec_tlv", "prob", "puntuacion_tlv", "monto", "target"]
        if c in rule_df.columns
    ]
    df_priority = (
        rule_df.sort_values(
            ["prob", "puntuacion_tlv"] if "puntuacion_tlv" in rule_df.columns else ["prob"],
            ascending=[False, False] if "puntuacion_tlv" in rule_df.columns else [False],
        )
        .head(10)
        .reset_index(drop=True)
    )
    df_priority.index = df_priority.index + 1

    st.write("**Clientes priorizados según regla de negocio**")
    st.dataframe(df_priority[display_cols], width="stretch")

    if "grupo_ejec_tlv" in rule_df.columns:
        st.write("**Distribución priorizada por grupo TLV**")
        group_prior = (
            rule_df.groupby("grupo_ejec_tlv")
            .size()
            .reset_index(name="Clientes")
            .sort_values("Clientes", ascending=False)
            .set_index("grupo_ejec_tlv")
        )
        st.bar_chart(group_prior)
else:
    st.info("No hay columna `prob` para generar el ranking de prioridades.")

# ---------------------------------------------------------------------------
# Bloque 7: Curva de recall por decil (bonus)
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