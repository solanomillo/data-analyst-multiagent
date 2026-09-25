"""Estado compartido del sistema de análisis multi-agente."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain.agents import AgentState


class AnalysisState(AgentState):
    """
    Estado compartido durante el proceso de análisis.

    Extiende el estado estándar de los agentes para incorporar
    la información específica del sistema de análisis.

    Los campos que representan colecciones de resultados utilizan
    reducers de LangGraph para permitir actualizaciones concurrentes
    durante la ejecución de herramientas y agentes especializados.
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

    eda_charts: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]

    chart_results: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]

    # ------------------------------------------------------------------
    # Análisis SQL
    # ------------------------------------------------------------------

    sql_query: Annotated[
        list[str],
        operator.add,
    ]

    sql_results: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]

    # ------------------------------------------------------------------
    # Interpretación y resultado final
    # ------------------------------------------------------------------

    eda_analysis: str
    narrative: str
    final_report: dict[str, Any]

    # ------------------------------------------------------------------
    # Control de errores
    # ------------------------------------------------------------------

    errors: Annotated[
        list[str],
        operator.add,
    ]