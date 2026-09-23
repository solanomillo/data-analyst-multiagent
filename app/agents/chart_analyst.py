"""Agente especializado en análisis y selección de visualizaciones."""

from __future__ import annotations

import logging

from langchain.agents import create_agent

from app.graph.state import AnalysisState
from app.prompts.prompts import CHART_ANALYST_SYSTEM_PROMPT
from app.services.llm import get_llm
from app.tools.chart_tools import (
    inspect_categorical_distribution,
    inspect_correlation_matrix,
    inspect_numeric_distribution,
    inspect_scatter_data,
)

logger = logging.getLogger(__name__)


def create_chart_analyst_agent():
    """
    Crea el agente especializado en visualización de datos.

    Returns:
        Agente LangChain configurado con las herramientas de
        visualización disponibles.
    """
    logger.info("Creando Chart Analyst Agent.")

    return create_agent(
        model=get_llm(),
        tools=[
            inspect_numeric_distribution,
            inspect_categorical_distribution,
            inspect_correlation_matrix,
            inspect_scatter_data,
        ],
        system_prompt=CHART_ANALYST_SYSTEM_PROMPT,
        state_schema=AnalysisState,
        name="chart_analyst",
    )