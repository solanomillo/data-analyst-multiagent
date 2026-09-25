"""Herramientas para delegar tareas a agentes especializados."""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from app.graph.state import AnalysisState


logger = logging.getLogger(__name__)


AGENT_STATE_FIELDS: dict[str, tuple[str, ...]] = {
    "data_quality_agent": (
        "dataset_schema",
        "missing_values",
        "duplicate_info",
        "numeric_summary",
        "categorical_summary",
        "outliers",
        "correlations",
        "eda_analysis",
    ),
    "sql_analyst": (
        "sql_query",
        "sql_results",
    ),
    "chart_analyst": (
        "chart_results",
    ),
    "narrative_agent": (
        "narrative",
        "final_report",
    ),
}


def _build_state_update(
    agent_name: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Filtra el resultado de un agente según su contrato de estado.

    Args:
        agent_name: Nombre del agente especializado.
        result: Estado devuelto por el agente.

    Returns:
        Diccionario con únicamente los campos permitidos.

    Raises:
        KeyError:
            Si no existe un contrato de estado para el agente.
    """
    if agent_name not in AGENT_STATE_FIELDS:
        raise KeyError(
            f"No existe un contrato de estado para el agente "
            f"'{agent_name}'."
        )

    allowed_fields = AGENT_STATE_FIELDS[agent_name]

    return {
        field: result[field]
        for field in allowed_fields
        if field in result
    }


def _extract_narrative(
    result: dict[str, Any],
) -> str:
    """
    Extrae el contenido narrativo generado por el agente.

    El Narrative Agent utiliza create_agent() sin herramientas.
    Por ello, su respuesta final se encuentra en el último
    AIMessage del historial de mensajes.

    Args:
        result: Estado devuelto por el Narrative Agent.

    Returns:
        Contenido textual de la respuesta del agente.

    Raises:
        RuntimeError:
            Si no existe un AIMessage con contenido textual.
    """
    messages = result.get("messages", [])

    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue

        content = message.content

        if isinstance(content, str) and content.strip():
            return content.strip()

    logger.error(
        "El Narrative Agent no produjo un AIMessage con contenido."
    )

    raise RuntimeError(
        "El Narrative Agent no produjo contenido narrativo."
    )


def _build_narrative_state_update(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Construye la actualización de estado del Narrative Agent.

    Args:
        result: Estado devuelto por el Narrative Agent.

    Returns:
        Estado con la narrativa y el informe final.
    """
    narrative = _extract_narrative(result)

    return {
        "narrative": narrative,
        "final_report": {
            "narrative": narrative,
        },
    }


def _create_specialist_state(
    state: AnalysisState,
) -> dict[str, Any]:
    """
    Crea el estado que recibirá un agente especializado.

    El agente especializado comparte los datos y resultados del
    análisis, pero inicia su propia conversación. Esto evita enviarle
    mensajes del supervisor que contengan tool_calls pendientes.

    Args:
        state: Estado actual del supervisor.

    Returns:
        Copia del estado con el historial de mensajes vacío.
    """
    specialist_state = dict(state)
    specialist_state["messages"] = []

    return specialist_state


def _run_specialist(
    agent_name: str,
    agent: Any,
    state: AnalysisState,
    tool_call_id: str | None,
) -> Command:
    """
    Ejecuta un agente especializado y propaga sus resultados.

    El agente especializado recibe el estado compartido del análisis,
    pero no recibe el historial conversacional del supervisor.

    Para los agentes analíticos, los resultados se obtienen directamente
    de los campos definidos en su contrato de estado.

    Para el Narrative Agent, la respuesta final del LLM se extrae del
    último AIMessage y se transforma explícitamente en los campos
    ``narrative`` y ``final_report``.

    Args:
        agent_name: Nombre del agente especializado.
        agent: Instancia del agente que será ejecutado.
        state: Estado actual del supervisor.
        tool_call_id: Identificador de la llamada de herramienta.

    Returns:
        Command con la actualización del estado y el ToolMessage
        correspondiente.

    Raises:
        RuntimeError:
            Si el agente no devuelve un estado válido, no produce
            narrativa cuando corresponde o no existe un tool_call_id.
        KeyError:
            Si el agente no tiene contrato de estado.
    """
    logger.info(
        "Ejecutando agente especializado: %s.",
        agent_name,
    )

    specialist_state = _create_specialist_state(
        state=state,
    )

    result = agent.invoke(specialist_state)

    if not isinstance(result, dict):
        logger.error(
            "El agente '%s' devolvió un resultado inválido.",
            agent_name,
        )

        raise RuntimeError(
            f"El agente '{agent_name}' no devolvió un estado válido."
        )

    if agent_name == "narrative_agent":
        state_update = _build_narrative_state_update(
            result=result,
        )

    else:
        state_update = _build_state_update(
            agent_name=agent_name,
            result=result,
        )

    logger.info(
        "El agente '%s' produjo %d actualización(es) de estado.",
        agent_name,
        len(state_update),
    )

    if tool_call_id is None:
        raise RuntimeError(
            "No existe un tool_call_id para completar la llamada "
            f"del agente '{agent_name}'."
        )

    tool_message = ToolMessage(
        content=(
            f"El agente '{agent_name}' completó la tarea "
            "correctamente."
        ),
        tool_call_id=tool_call_id,
        name=agent_name,
    )

    state_update["messages"] = [tool_message]

    return Command(
        update=state_update,
    )


def create_agent_tools(
    agents: dict[str, Any],
) -> list[Any]:
    """
    Crea las herramientas de delegación del supervisor.

    Args:
        agents: Registro de agentes especializados.

    Returns:
        Lista de herramientas disponibles para el supervisor.

    Raises:
        KeyError:
            Si falta algún agente requerido en el registro.
    """
    required_agents = set(AGENT_STATE_FIELDS)
    missing_agents = required_agents - agents.keys()

    if missing_agents:
        raise KeyError(
            "Faltan agentes requeridos en el registro: "
            f"{sorted(missing_agents)}."
        )

    @tool
    def call_data_quality_agent(
        runtime: ToolRuntime,
    ) -> Command:
        """Ejecuta el Data Quality Agent y actualiza el estado."""
        logger.info(
            "Supervisor delegando tarea al Data Quality Agent."
        )

        return _run_specialist(
            agent_name="data_quality_agent",
            agent=agents["data_quality_agent"],
            state=runtime.state,
            tool_call_id=runtime.tool_call_id,
        )

    @tool
    def call_sql_agent(
        runtime: ToolRuntime,
    ) -> Command:
        """Ejecuta el SQL Analyst y actualiza el estado."""
        logger.info(
            "Supervisor delegando tarea al SQL Analyst."
        )

        return _run_specialist(
            agent_name="sql_analyst",
            agent=agents["sql_analyst"],
            state=runtime.state,
            tool_call_id=runtime.tool_call_id,
        )

    @tool
    def call_chart_agent(
        runtime: ToolRuntime,
    ) -> Command:
        """Ejecuta el Chart Analyst y actualiza el estado."""
        logger.info(
            "Supervisor delegando tarea al Chart Analyst."
        )

        return _run_specialist(
            agent_name="chart_analyst",
            agent=agents["chart_analyst"],
            state=runtime.state,
            tool_call_id=runtime.tool_call_id,
        )

    @tool
    def call_narrative_agent(
        runtime: ToolRuntime,
    ) -> Command:
        """Ejecuta el Narrative Agent y actualiza el estado."""
        logger.info(
            "Supervisor delegando tarea al Narrative Agent."
        )

        return _run_specialist(
            agent_name="narrative_agent",
            agent=agents["narrative_agent"],
            state=runtime.state,
            tool_call_id=runtime.tool_call_id,
        )

    return [
        call_data_quality_agent,
        call_sql_agent,
        call_chart_agent,
        call_narrative_agent,
    ]