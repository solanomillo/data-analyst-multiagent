"""Estado compartido del sistema de análisis multi-agente."""

from __future__ import annotations

from typing import Any

from langchain.agents import AgentState


class AnalysisState(AgentState):
    """
    Estado compartido durante el proceso de análisis.

    Extiende el estado estándar de los agentes para incorporar
    la información específica de nuestro sistema de análisis.
    """

    # ------------------------------------------------------------------
    # Entrada del usuario
    # ------------------------------------------------------------------

    dataset: Any
    dataset_name: str
    user_question: str

    # ------------------------------------------------------------------
    # Información general del dataset
    # ------------------------------------------------------------------

    dataset_info: dict[str, Any]
    dataset_schema: dict[str, Any]

    # ------------------------------------------------------------------
    # Calidad y EDA
    # ------------------------------------------------------------------

    missing_values: dict[str, Any]
    duplicate_info: dict[str, Any]
    numeric_summary: dict[str, Any]
    categorical_summary: dict[str, Any]
    outliers: dict[str, Any]
    correlations: dict[str, Any]

    # ------------------------------------------------------------------
    # Visualizaciones
    # ------------------------------------------------------------------

    eda_charts: list[dict[str, Any]]
    chart_results: list[dict[str, Any]]

    # ------------------------------------------------------------------
    # Análisis SQL
    # ------------------------------------------------------------------

    sql_query: str
    sql_results: dict[str, Any]

    # ------------------------------------------------------------------
    # Interpretación y resultado final
    # ------------------------------------------------------------------

    eda_analysis: str
    narrative: str
    final_report: dict[str, Any]

    # ------------------------------------------------------------------
    # Control de errores
    # ------------------------------------------------------------------

    errors: list[str]