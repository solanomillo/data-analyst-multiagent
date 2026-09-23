"""Pruebas para las herramientas de delegación de agentes."""

from unittest.mock import Mock

from app.tools.agent_tools import create_agent_tools


def test_create_agent_tools() -> None:
    """Verifica la creación de herramientas de delegación."""

    agents = {
        "data_quality_agent": Mock(),
        "sql_analyst": Mock(),
        "chart_analyst": Mock(),
        "narrative_agent": Mock(),
    }

    tools = create_agent_tools(agents)

    assert len(tools) == 4

    tool_names = {
        tool.name
        for tool in tools
    }

    assert tool_names == {
        "call_data_quality_agent",
        "call_sql_agent",
        "call_chart_agent",
        "call_narrative_agent",
    }