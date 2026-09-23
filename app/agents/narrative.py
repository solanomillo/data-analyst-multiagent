"""Agente especializado en generar el informe final."""

from __future__ import annotations

import logging

from langchain.agents import create_agent

from app.graph.state import AnalysisState
from app.prompts.prompts import NARRATIVE_SYSTEM_PROMPT
from app.services.llm import get_llm


logger = logging.getLogger(__name__)


def create_narrative_agent():
    """
    Crea el agente encargado de generar el informe narrativo.

    Returns:
        Agente LangChain configurado para comunicar los resultados.
    """
    logger.info("Creando Narrative Agent.")

    return create_agent(
        model=get_llm(),
        tools=[],
        system_prompt=NARRATIVE_SYSTEM_PROMPT,
        state_schema=AnalysisState,
        name="narrative_agent",
    )