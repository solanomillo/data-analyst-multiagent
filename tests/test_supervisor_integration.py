"""Pruebas de integración del supervisor multi-agente."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import ConfigDict

from app.agents.supervisor import create_supervisor_agent
from app.graph.state import AnalysisState
from app.tools.agent_tools import create_agent_tools


class FakeToolCallingChatModel(BaseChatModel):
    """Modelo falso para probar agentes que utilizan herramientas."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    responses: list[AIMessage]
    bound_tools: list[BaseTool] = []

    @property
    def _llm_type(self) -> str:
        """
        Identifica el modelo utilizado exclusivamente en las pruebas.

        Returns:
            Nombre del modelo falso.
        """
        return "fake-tool-calling"

    def bind_tools(
        self,
        tools: list[BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "FakeToolCallingChatModel":
        """
        Simula el binding de herramientas realizado por create_agent.

        Args:
            tools: Herramientas disponibles para el modelo.
            tool_choice: Elección opcional de herramienta.
            **kwargs: Parámetros adicionales.

        Returns:
            Copia del modelo con las herramientas asociadas.
        """
        del tool_choice
        del kwargs

        return self.model_copy(
            update={
                "bound_tools": tools,
            }
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Devuelve la siguiente respuesta configurada.

        Args:
            messages: Mensajes enviados al modelo.
            stop: Secuencias de parada.
            run_manager: Gestor de ejecución.
            **kwargs: Parámetros adicionales.

        Returns:
            Resultado simulado del modelo.
        """
        del messages
        del stop
        del run_manager
        del kwargs

        if not self.responses:
            raise RuntimeError(
                "El modelo falso no tiene más respuestas configuradas."
            )

        response = self.responses.pop(0)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=response,
                )
            ]
        )


def _create_initial_state(
    dataframe: pd.DataFrame,
) -> AnalysisState:
    """
    Crea un estado inicial para las pruebas del supervisor.

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
        sql_query=[],
        sql_results=[],
        eda_analysis="",
        narrative="",
        final_report={},
        errors=[],
        messages=[],
    )


def test_supervisor_creation() -> None:
    """Verifica la creación del supervisor."""

    agents = {
        "data_quality_agent": Mock(),
        "sql_analyst": Mock(),
        "chart_analyst": Mock(),
        "narrative_agent": Mock(),
    }

    fake_llm = Mock()

    with patch(
        "app.agents.supervisor.create_agent_registry",
        return_value=agents,
    ), patch(
        "app.agents.supervisor.get_llm",
        return_value=fake_llm,
    ):
        supervisor = create_supervisor_agent()

    assert supervisor is not None


def test_supervisor_executes_data_quality_agent() -> None:
    """Verifica que el supervisor delega correctamente en EDA."""

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
            "El producto B presenta mayores ventas."
        ),
    }

    agents = {
        "data_quality_agent": data_quality_agent,
        "sql_analyst": Mock(),
        "chart_analyst": Mock(),
        "narrative_agent": Mock(),
    }

    agent_tools = create_agent_tools(agents)

    fake_model = FakeToolCallingChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "call_data_quality_agent",
                        "args": {},
                        "id": "call_data_quality_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "El análisis de calidad fue completado."
                ),
            ),
        ]
    )

    supervisor = create_agent(
        model=fake_model,
        tools=agent_tools,
        system_prompt=(
            "Coordina los agentes especializados de análisis."
        ),
        state_schema=AnalysisState,
        name="supervisor",
    )

    result = supervisor.invoke(state)

    assert isinstance(result, dict)

    assert result["eda_analysis"] == (
        "El producto B presenta mayores ventas."
    )

    data_quality_agent.invoke.assert_called_once()

    received_state = (
        data_quality_agent.invoke.call_args.args[0]
    )

    assert received_state["dataset_name"] == "ventas.csv"

    assert received_state["user_question"] == (
        "Analiza las ventas."
    )

    assert received_state["dataset"].equals(dataframe)

    messages = result["messages"]

    tool_messages = [
        message
        for message in messages
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1

    tool_message = tool_messages[0]

    assert tool_message.name == (
        "call_data_quality_agent"
    )

    assert tool_message.tool_call_id == (
        "call_data_quality_1"
    )

    ai_messages = [
        message
        for message in messages
        if isinstance(message, AIMessage)
    ]

    assert len(ai_messages) >= 2

    assert ai_messages[-1].content == (
        "El análisis de calidad fue completado."
    )