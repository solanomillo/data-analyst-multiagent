"""Pruebas de integración de las herramientas de agentes."""

from unittest.mock import Mock

import pandas as pd

from langchain.tools import ToolRuntime

from app.graph.workflow import create_initial_state
from app.tools.agent_tools import create_agent_tools


def _create_runtime(state):
    """
    Crea un ToolRuntime para ejecutar una herramienta de forma aislada.

    Args:
        state: Estado compartido del análisis.

    Returns:
        Instancia de ToolRuntime configurada para la prueba.
    """
    return ToolRuntime(
        state=state,
        context={},
        config={},
        stream_writer=lambda _: None,
        tool_call_id=None,
        store=None,
    )


def test_specialist_agent_result_is_returned() -> None:
    """Verifica que una herramienta devuelva el resultado del agente."""

    dataframe = pd.DataFrame(
        {
            "producto": ["A", "B"],
            "ventas": [100, 200],
        }
    )

    specialist_result = {
        "dataset_name": "ventas.csv",
        "eda_analysis": (
            "El dataset contiene 2 registros y 2 columnas."
        ),
    }

    mock_agent = Mock()
    mock_agent.invoke.return_value = specialist_result

    agents = {
        "data_quality_agent": mock_agent,
        "sql_analyst": Mock(),
        "chart_analyst": Mock(),
        "narrative_agent": Mock(),
    }

    tools = create_agent_tools(agents)

    state = create_initial_state(
        dataframe=dataframe,
        dataset_name="ventas.csv",
        user_question="Analiza las ventas.",
    )

    data_quality_tool = next(
        tool
        for tool in tools
        if tool.name == "call_data_quality_agent"
    )

    runtime = _create_runtime(state)

    result = data_quality_tool.func(
        runtime=runtime,
    )

    assert result == specialist_result

    mock_agent.invoke.assert_called_once_with(state)


def test_narrative_agent_receives_current_state() -> None:
    """Verifica que el Narrative Agent reciba el estado actual."""

    dataframe = pd.DataFrame(
        {
            "producto": ["A", "B"],
            "ventas": [100, 200],
        }
    )

    narrative_agent = Mock()

    narrative_agent.invoke.return_value = {
        "narrative": "Informe generado correctamente.",
    }

    agents = {
        "data_quality_agent": Mock(),
        "sql_analyst": Mock(),
        "chart_analyst": Mock(),
        "narrative_agent": narrative_agent,
    }

    tools = create_agent_tools(agents)

    state = create_initial_state(
        dataframe=dataframe,
        dataset_name="ventas.csv",
        user_question="¿Cuál producto tiene mayores ventas?",
    )

    state["eda_analysis"] = (
        "El producto B presenta mayores ventas."
    )

    narrative_tool = next(
        tool
        for tool in tools
        if tool.name == "call_narrative_agent"
    )

    runtime = _create_runtime(state)

    result = narrative_tool.func(
        runtime=runtime,
    )

    assert result == {
        "narrative": "Informe generado correctamente.",
    }

    narrative_agent.invoke.assert_called_once()

    received_state = (
        narrative_agent.invoke.call_args.args[0]
    )

    assert received_state["dataset_name"] == "ventas.csv"

    assert (
        received_state["eda_analysis"]
        == "El producto B presenta mayores ventas."
    )