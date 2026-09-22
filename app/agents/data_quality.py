"""Agente especializado en calidad y análisis exploratorio."""

from __future__ import annotations

import logging

from langchain.agents import create_agent

from app.graph.state import AnalysisState
from app.prompts.prompts import DATA_QUALITY_SYSTEM_PROMPT
from app.services.llm import get_llm
from app.tools.data_tools import (
    inspect_categorical_statistics,
    inspect_correlations,
    inspect_dataset_schema,
    inspect_duplicates,
    inspect_missing_values,
    inspect_numeric_statistics,
    inspect_outliers,
)


logger = logging.getLogger(__name__)


def create_data_quality_agent():
    """
    Crea el agente especializado en calidad y EDA.

    Returns:
        Agente LangChain configurado con las herramientas de EDA.
    """
    logger.info(
        "Creando Data Quality Agent."
    )

    return create_agent(
        model=get_llm(),
        tools=[
            inspect_dataset_schema,
            inspect_missing_values,
            inspect_duplicates,
            inspect_numeric_statistics,
            inspect_categorical_statistics,
            inspect_outliers,
            inspect_correlations,
        ],
        system_prompt=DATA_QUALITY_SYSTEM_PROMPT,
        state_schema=AnalysisState,
        name="data_quality_agent",
    )