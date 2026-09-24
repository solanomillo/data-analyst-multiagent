"""Pruebas de integración entre herramientas y agentes especializados."""

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
from langchain_core.messages import ToolMessage
from langchain.tools import ToolRuntime
from langgraph.types import Command

from app.graph.state import AnalysisState
from app.tools.agent_tools import create_agent_tools


def _create_runtime(
    state: AnalysisState,
    tool_call_id: str = "test_tool_call_id",
) -> ToolRuntime:
    """
    Crea un ToolRuntime controlado para las pruebas.

    Args:
        state: Estado compartido utilizado durante la prueba.
        tool_call_id: Identificador simulado de la llamada.

    Returns:
        ToolRuntime configurado para testing.
    """
    return ToolRuntime(
        state=state,
        context={},
        config={},
        stream_writer=lambda _: None,
        tool_call_id=tool_call_id,
        store=None,
    )


def _create_initial_state(
    dataframe: pd.DataFrame,
) -> AnalysisState:
    """
    Crea un estado inicial mínimo para las pruebas.

    Args:
        dataframe: Dataset utilizado durante la prueba.

    Returns:
        Estado inicial del análisis.
    """
    return AnalysisState(
        dataset=dataframe,
        dataset_name="ventas.csv",
        user_question="Analiza las ventas.",
        dataset_info={
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
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
        sql_query="",
        sql_results={},
        eda_analysis="",
        narrative="",
        final_report={},
        errors=[],
        messages=[],
    )


def test_specialist_agent_result_updates_state() -> None:
    """Verifica que el resultado del agente actualiza el estado."""

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
            "El dataset contiene 2 registros y 2 columnas."
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
    # Para esta prueba unitaria ejecutamos directamente la función
    # Python que está detrás de la StructuredTool.
    result = data_quality_tool.func(runtime)

    assert isinstance(result, Command)

    assert result.update["eda_analysis"] == (
        "El dataset contiene 2 registros y 2 columnas."
    )

    assert "messages" in result.update
    assert len(result.update["messages"]) == 1

    tool_message = result.update["messages"][0]

    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "test_tool_call_id"

    data_quality_agent.invoke.assert_called_once_with(state)


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
        "narrative": "Informe generado correctamente.",
        "final_report": {
            "summary": "Análisis completado.",
        },
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

    assert isinstance(result, Command)

    assert result.update["narrative"] == (
        "Informe generado correctamente."
    )

    assert result.update["final_report"] == {
        "summary": "Análisis completado.",
    }

    assert "messages" in result.update
    assert len(result.update["messages"]) == 1

    tool_message = result.update["messages"][0]

    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "test_tool_call_id"

    narrative_agent.invoke.assert_called_once()

    received_state = (
        narrative_agent.invoke.call_args.args[0]
    )

    assert received_state["dataset_name"] == "ventas.csv"

    assert received_state["user_question"] == (
        "Analiza las ventas."
    )

    assert received_state["eda_analysis"] == (
        "El producto B presenta mayores ventas."
    )

    assert received_state["dataset"].equals(dataframe)