"""Pruebas para el registro de agentes."""

from unittest.mock import patch

from app.agents.registry import create_agent_registry


def test_create_agent_registry() -> None:
    """Verifica que se creen todos los agentes especializados."""

    with (
        patch(
            "app.agents.registry.create_data_quality_agent"
        ) as data_quality,
        patch(
            "app.agents.registry.create_sql_analyst_agent"
        ) as sql_analyst,
        patch(
            "app.agents.registry.create_chart_analyst_agent"
        ) as chart_analyst,
        patch(
            "app.agents.registry.create_narrative_agent"
        ) as narrative,
    ):
        data_quality.return_value = object()
        sql_analyst.return_value = object()
        chart_analyst.return_value = object()
        narrative.return_value = object()

        registry = create_agent_registry()

    assert set(registry) == {
        "data_quality_agent",
        "sql_analyst",
        "chart_analyst",
        "narrative_agent",
    }

    data_quality.assert_called_once()
    sql_analyst.assert_called_once()
    chart_analyst.assert_called_once()
    narrative.assert_called_once()