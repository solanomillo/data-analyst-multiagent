"""Herramientas para delegar tareas a agentes especializados."""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import ToolRuntime, tool

from app.graph.state import AnalysisState

logger = logging.getLogger(__name__)


def _run_specialist(
    agent: Any,
    state: AnalysisState,
) -> dict[str, Any]:
    """
    Ejecuta un agente especializado con el estado actual.

    Args:
        agent: Agente especializado.
        state: Estado compartido.

    Returns:
        Estado producido por el agente.
    """
    result = agent.invoke(state)

    if not isinstance(result, dict):
        raise RuntimeError(
            "El agente especializado no devolvió un estado válido."
        )

    return result


def create_agent_tools(
    agents: dict[str, Any],
) -> list[Any]:
    """
    Crea las herramientas de delegación del supervisor.

    Args:
        agents: Registro de agentes especializados.

    Returns:
        Lista de herramientas disponibles para el supervisor.
    """

    @tool
    def call_data_quality_agent(
        runtime: ToolRuntime,
    ) -> dict[str, Any]:
        """
        Ejecuta el Data Quality Agent.
        """
        logger.info(
            "Supervisor delegando tarea al Data Quality Agent."
        )

        return _run_specialist(
            agents["data_quality_agent"],
            runtime.state,
        )

    @tool
    def call_sql_agent(
        runtime: ToolRuntime,
    ) -> dict[str, Any]:
        """
        Ejecuta el SQL Analyst.
        """
        logger.info(
            "Supervisor delegando tarea al SQL Analyst."
        )

        return _run_specialist(
            agents["sql_analyst"],
            runtime.state,
        )

    @tool
    def call_chart_agent(
        runtime: ToolRuntime,
    ) -> dict[str, Any]:
        """
        Ejecuta el Chart Analyst.
        """
        logger.info(
            "Supervisor delegando tarea al Chart Analyst."
        )

        return _run_specialist(
            agents["chart_analyst"],
            runtime.state,
        )

    @tool
    def call_narrative_agent(
        runtime: ToolRuntime,
    ) -> dict[str, Any]:
        """
        Ejecuta el Narrative Agent.
        """
        logger.info(
            "Supervisor delegando tarea al Narrative Agent."
        )

        return _run_specialist(
            agents["narrative_agent"],
            runtime.state,
        )

    return [
        call_data_quality_agent,
        call_sql_agent,
        call_chart_agent,
        call_narrative_agent,
    ]