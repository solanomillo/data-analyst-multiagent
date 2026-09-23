"""Agente supervisor del sistema de análisis multi-agente."""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent

from app.agents.registry import create_agent_registry
from app.graph.state import AnalysisState
from app.prompts.prompts import SUPERVISOR_SYSTEM_PROMPT
from app.services.llm import get_llm
from app.tools.agent_tools import create_agent_tools

logger = logging.getLogger(__name__)


def create_supervisor_agent() -> Any:
    """
    Crea el supervisor y sus agentes especializados.

    Returns:
        Agente supervisor configurado.
    """
    logger.info("Creando sistema supervisor.")

    agents = create_agent_registry()

    tools = create_agent_tools(
        agents
    )

    return create_agent(
        model=get_llm(),
        tools=tools,
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        state_schema=AnalysisState,
        name="supervisor",
    )