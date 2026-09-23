"""Registro centralizado de agentes especializados."""

from __future__ import annotations

from typing import Any

from app.agents.chart_analyst import create_chart_analyst_agent
from app.agents.data_quality import create_data_quality_agent
from app.agents.narrative import create_narrative_agent
from app.agents.sql_analyst import create_sql_analyst_agent


def create_agent_registry() -> dict[str, Any]:
    """
    Crea el registro de agentes especializados.

    Returns:
        Diccionario con todos los agentes disponibles.
    """
    return {
        "data_quality_agent": create_data_quality_agent(),
        "sql_analyst": create_sql_analyst_agent(),
        "chart_analyst": create_chart_analyst_agent(),
        "narrative_agent": create_narrative_agent(),
    }