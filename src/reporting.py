"""
Módulo de reportería — Visualizaciones mes a mes por grupo de ejecución.

Genera tres gráficos a partir del DataFrame postprocesado + réplica:

  1. Cantidad de conversiones (target=1) por grupo de ejecución y mes.
  2. Efectividad (%) = conversiones / leads por grupo y mes.
  3. Monto promedio ofertado en conversiones por grupo y mes.

Uso:
    from src.reporting import plot_grupo_ejec_report

    fig = plot_grupo_ejec_report(df_merged, output_path="data/reports/reporte_grupos.png")
    fig.savefig("reporte.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Paleta de colores para los 10 grupos (del 10=verde oscuro al 1=rojo)
_PALETTE = [
    "#1a7a4a", "#2d9e5f", "#5ab87a", "#8ecf96", "#bde5b4",
    "#f7c59f", "#f4a261", "#e07b3b", "#c0522a", "#9b2918",
]


def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el DataFrame: {missing}")


def build_report_df(
    df_postprocessed: pd.DataFrame,
    codmes_col: str = "p_codmes",
    grupo_col: str = "grupo_ejec_tlv",
    target_col: str = "target",
    monto_col: str = "monto",
) -> pd.DataFrame:
    """
    Construye el DataFrame de reporte agrupado por mes y grupo de ejecución.

    Args:
        df_postprocessed: DataFrame con la salida de run_postprocessing().
                          Debe incluir columnas de mes, grupo, target y monto.
        codmes_col:  Columna con el mes (p.ej. p_codmes → 201912.0).
        grupo_col:   Columna con el grupo de ejecución (1-10).
        target_col:  Columna binaria con el resultado real (0/1).
        monto_col:   Columna con el monto ofertado al cliente.

    Returns:
        DataFrame con columnas: codmes, grupo_ejec, leads, conversiones,
        efectividad_pct, monto_promedio_conversiones.
    """
    _validate_columns(df_postprocessed, [codmes_col, grupo_col, target_col, monto_col])

    df = df_postprocessed.copy()
    df[grupo_col] = df[grupo_col].astype(int)
    df[codmes_col] = df[codmes_col].astype(int).astype(str)

    agg = (
        df.groupby([codmes_col, grupo_col], observed=True)
        .agg(
            leads=         (target_col, "count"),
            conversiones=  (target_col, "sum"),
            monto_sum_conv=(monto_col,  lambda x: x[df.loc[x.index, target_col] == 1].sum()),
        )
        .reset_index()
        .rename(columns={codmes_col: "codmes", grupo_col: "grupo_ejec"})
    )

    agg["efectividad_pct"]         = (agg["conversiones"] / agg["leads"] * 100).round(2)
    agg["monto_promedio_conv"]     = (agg["monto_sum_conv"] / agg["conversiones"].replace(0, np.nan)).round(0)

    return agg.sort_values(["codmes", "grupo_ejec"], ascending=[True, False])


def plot_grupo_ejec_report(
    df_postprocessed: pd.DataFrame,
    codmes_col: str = "p_codmes",
    grupo_col: str = "grupo_ejec_tlv",
    target_col: str = "target",
    monto_col: str = "monto",
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (18, 14),
) -> plt.Figure:
    """
    Genera el reporte visual de 3 paneles mes a mes por grupo de ejecución.

    Panel 1 — Cantidad de conversiones (1s) por grupo y mes.
    Panel 2 — Efectividad % (conversiones / leads) por grupo y mes.
    Panel 3 — Monto promedio ofertado en conversiones por grupo y mes.

    Args:
        df_postprocessed: DataFrame con salida de run_postprocessing().
        codmes_col:   Columna de mes.
        grupo_col:    Columna de grupo de ejecución (1-10).
        target_col:   Columna de target binario.
        monto_col:    Columna de monto.
        output_path:  Ruta para guardar la figura (opcional).
        figsize:      Tamaño de la figura en pulgadas.

    Returns:
        Objeto matplotlib.figure.Figure.
    """
    report = build_report_df(
        df_postprocessed,
        codmes_col=codmes_col,
        grupo_col=grupo_col,
        target_col=target_col,
        monto_col=monto_col,
    )

    meses  = sorted(report["codmes"].unique())
    grupos = sorted(report["grupo_ejec"].unique(), reverse=True)   # 10 → 1
    x      = np.arange(len(meses))
    n_g    = len(grupos)
    width  = 0.8 / n_g

    fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)
    fig.suptitle(
        "Reporte mes a mes por Grupo de Ejecución TLV",
        fontsize=15, fontweight="bold", y=1.01,
    )

    metrics = [
        ("conversiones",        "Cantidad de Conversiones (1s)",         "N° conversiones",  False),
        ("efectividad_pct",     "Efectividad % (Conversiones / Leads)",  "Efectividad (%)",   False),
        ("monto_promedio_conv", "Monto Promedio Ofertado en Conversiones","Monto promedio (S/)", True),
    ]

    for ax, (col, title, ylabel, money) in zip(axes, metrics):
        for i, (grupo, color) in enumerate(zip(grupos, _PALETTE)):
            subset = report[report["grupo_ejec"] == grupo]
            vals   = [
                subset.loc[subset["codmes"] == m, col].values[0]
                if m in subset["codmes"].values else 0
                for m in meses
            ]
            offset = (i - n_g / 2 + 0.5) * width
            bars   = ax.bar(x + offset, vals, width=width * 0.9,
                            color=color, label=f"Grupo {grupo}", zorder=3)

            # Etiquetas de valor sobre las barras (solo si hay espacio)
            if len(meses) <= 6:
                for bar, val in zip(bars, vals):
                    if val and not np.isnan(val):
                        label = f"{val:,.0f}" if money else (
                            f"{val:.1f}%" if col == "efectividad_pct" else f"{int(val):,}"
                        )
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + ax.get_ylim()[1] * 0.005,
                            label, ha="center", va="bottom",
                            fontsize=6, color="black",
                        )

        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(meses, rotation=30, ha="right", fontsize=9)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"S/ {v:,.0f}" if money else f"{v:,.0f}")
        )
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    # Leyenda única debajo de los paneles
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Grupo de Ejecución\n(10=mayor prioridad, 1=menor)",
        loc="lower center",
        ncol=min(n_g, 5),
        bbox_to_anchor=(0.5, -0.06),
        fontsize=8,
        title_fontsize=8,
        framealpha=0.9,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Reporte guardado en: %s", output_path)

    return fig


def plot_efectividad_heatmap(
    df_postprocessed: pd.DataFrame,
    codmes_col: str = "p_codmes",
    grupo_col: str = "grupo_ejec_tlv",
    target_col: str = "target",
    monto_col: str = "monto",
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Heatmap de efectividad (%) con filas=grupos y columnas=meses.
    Complementa los gráficos de barras con una vista tipo matriz.

    Returns:
        Objeto matplotlib.figure.Figure.
    """
    report = build_report_df(
        df_postprocessed,
        codmes_col=codmes_col,
        grupo_col=grupo_col,
        target_col=target_col,
        monto_col=monto_col,
    )

    pivot = report.pivot(index="grupo_ejec", columns="codmes", values="efectividad_pct")
    pivot = pivot.sort_index(ascending=False)   # grupo 10 arriba

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"Grupo {g}" for g in pivot.index], fontsize=9)
    ax.set_title("Heatmap de Efectividad % por Grupo y Mes", fontsize=12, fontweight="bold")

    # Anotaciones de valores
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=8, color="black" if val < 8 else "white")

    plt.colorbar(im, ax=ax, label="Efectividad (%)", shrink=0.8)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Heatmap guardado en: %s", output_path)

    return fig
