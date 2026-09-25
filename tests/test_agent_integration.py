"""Pruebas de integración para las herramientas de agentes."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage
from langchain.tools import ToolRuntime

from app.graph.state import AnalysisState
from app.tools.agent_tools import create_agent_tools


def _create_initial_state(dataframe: pd.DataFrame) -> AnalysisState:
    """
    Crea un estado inicial para las pruebas de integración.

    Args:
        dataframe: DataFrame utilizado como dataset de prueba.

    Returns:
        Estado inicial del sistema de análisis.
    """
    return AnalysisState(
        dataset=dataframe,
        dataset_name="dataset_prueba.csv",
        user_question="Analizar las ventas.",
        dataset_info={
            "rows": dataframe.shape[0],
            "columns": dataframe.shape[1],
        },
        dataset_schema={},
        missing_values={},
        duplicate_info={},
        numeric_summary={},
        categorical_summary={},
        outliers={},
        correlations={},
        eda_charts=[],
        chart_results=[],
        sql_query=[],
        sql_results=[],
        eda_analysis="",
        narrative="",
        final_report={},
        errors=[],
        messages=[],
    )


def _create_runtime(state: AnalysisState) -> ToolRuntime:
    """
    Crea un ToolRuntime simulado para las pruebas.

    Args:
        state: Estado compartido del análisis.

    Returns:
        Instancia de ToolRuntime con el estado proporcionado.
    """
    return ToolRuntime(
        state=state,
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="test-tool-call",
        store=None,
    )


def test_specialist_agent_result_updates_state() -> None:
    """Verifica que un agente especializado actualiza el estado."""

    dataframe = pd.DataFrame(
        {
            "producto": ["A", "B"],
            "ventas": [100, 200],
        }
    )

    state = _create_initial_state(dataframe)

    data_quality_agent = Mock()

    data_quality_agent.invoke.return_value = {
        "eda_analysis": (
            "El dataset presenta una estructura válida."
        ),
    }

    agents = {
        "data_quality_agent": data_quality_agent,
        "sql_analyst": Mock(),
        "chart_analyst": Mock(),
        "narrative_agent": Mock(),
    }

    tools = create_agent_tools(agents)

    data_quality_tool = next(
        tool
        for tool in tools
        if tool.name == "call_data_quality_agent"
    )

    runtime = _create_runtime(state)

    # ToolRuntime es un argumento inyectado por LangChain.
    # La prueba ejecuta directamente la función de la herramienta.
    result = data_quality_tool.func(runtime)

    assert result.update["eda_analysis"] == (
        "El dataset presenta una estructura válida."
    )

    messages = result.update["messages"]

    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].tool_call_id == "test-tool-call"


def test_narrative_agent_receives_current_state() -> None:
    """Verifica que el Narrative Agent recibe el estado actual."""

    dataframe = pd.DataFrame(
        {
            "producto": ["A", "B"],
            "ventas": [100, 200],
        }
    )

    state = _create_initial_state(dataframe)

    state["eda_analysis"] = (
        "El producto B presenta mayores ventas."
    )

    narrative_agent = Mock()

    narrative_agent.invoke.return_value = {
        "messages": [
            AIMessage(
                content="Informe generado correctamente."
            )
        ],
    }

    agents = {
        "data_quality_agent": Mock(),
        "sql_analyst": Mock(),
        "chart_analyst": Mock(),
        "narrative_agent": narrative_agent,
    }

    tools = create_agent_tools(agents)

    narrative_tool = next(
        tool
        for tool in tools
        if tool.name == "call_narrative_agent"
    )

    runtime = _create_runtime(state)

    # ToolRuntime es un argumento inyectado por LangChain.
    # La prueba ejecuta directamente la función de la herramienta.
    result = narrative_tool.func(runtime)

    narrative_agent.invoke.assert_called_once()

    received_state = narrative_agent.invoke.call_args.args[0]

    assert received_state["dataset"] is dataframe
    assert received_state["dataset_name"] == "dataset_prueba.csv"
    assert received_state["user_question"] == "Analizar las ventas."
    assert received_state["eda_analysis"] == (
        "El producto B presenta mayores ventas."
    )

    assert result.update["narrative"] == (
        "Informe generado correctamente."
    )

    assert result.update["final_report"] == {
        "narrative": "Informe generado correctamente.",
    }

    messages = result.update["messages"]

    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].tool_call_id == "test-tool-call"