"""Herramientas para delegar tareas a agentes especializados."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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
    Construye la actualización del estado a partir del resultado
    de un agente especializado.

    Args:
        agent_name: Nombre del agente especializado.
        result: Resultado devuelto por el agente.

    Returns:
        Diccionario con los campos de estado relevantes.

    Raises:
        ValueError:
            Si el agente no tiene campos de estado configurados.
    """
    fields = AGENT_STATE_FIELDS.get(agent_name)

    if fields is None:
        raise ValueError(
            f"No existen campos de estado configurados para "
            f"el agente '{agent_name}'."
        )

    return {
        field: result[field]
        for field in fields
        if field in result
    }


def _create_specialist_state(
    state: AnalysisState,
) -> dict[str, Any]:
    """
    Crea una copia del estado para ejecutar un agente especializado.

    Los mensajes del supervisor no se reutilizan directamente dentro
    del agente especializado para mantener separadas las conversaciones
    de cada agente.

    Args:
        state: Estado compartido del workflow.

    Returns:
        Estado preparado para el agente especializado.
    """
    specialist_state = dict(state)
    specialist_state["messages"] = []

    return specialist_state


def _serialize_context(
    context: dict[str, Any],
) -> str:
    """
    Serializa un contexto estructurado para enviarlo al LLM.

    Args:
        context: Información contextual del análisis.

    Returns:
        Representación JSON legible del contexto.
    """
    return json.dumps(
        context,
        ensure_ascii=False,
        default=str,
        indent=2,
    )


def _extract_narrative(
    result: dict[str, Any],
) -> str:
    """
    Extrae el contenido de la última respuesta del Narrative Agent.

    Los agentes creados mediante create_agent() devuelven sus respuestas
    del modelo dentro de la colección de mensajes.

    Args:
        result: Resultado devuelto por el Narrative Agent.

    Returns:
        Texto generado por el modelo.

    Raises:
        RuntimeError:
            Si el agente no produce un AIMessage con contenido.
    """
    messages = result.get("messages", [])

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content

            if isinstance(content, str) and content.strip():
                return content.strip()

    raise RuntimeError(
        "El Narrative Agent no produjo una respuesta de texto válida."
    )


def _build_narrative_context(
    state: AnalysisState,
) -> str:
    """
    Construye el contexto de evidencia que recibirá el Narrative Agent.

    El DataFrame original no se incluye en el contexto para evitar
    enviar innecesariamente todos los registros al modelo. Se utilizan
    únicamente los resultados calculados por las herramientas
    especializadas y almacenados en el estado.

    Args:
        state: Estado actual del análisis.

    Returns:
        Contexto estructurado para el Narrative Agent.
    """
    evidence = {
        "dataset_name": state.get("dataset_name"),
        "user_question": state.get("user_question"),
        "dataset_info": state.get("dataset_info"),
        "dataset_schema": state.get("dataset_schema"),
        "missing_values": state.get("missing_values"),
        "duplicate_info": state.get("duplicate_info"),
        "numeric_summary": state.get("numeric_summary"),
        "categorical_summary": state.get(
            "categorical_summary"
        ),
        "outliers": state.get("outliers"),
        "correlations": state.get("correlations"),
        "eda_analysis": state.get("eda_analysis"),
        "sql_query": state.get("sql_query", []),
        "sql_results": state.get(
            "sql_results",
            [],
        ),
        "chart_results": state.get(
            "chart_results",
            [],
        ),
        "errors": state.get("errors", []),
    }

    serialized_evidence = _serialize_context(evidence)

    return (
        "A continuación se proporciona el estado consolidado "
        "del análisis realizado por los agentes especializados.\n\n"
        "Debes elaborar el informe utilizando únicamente esta "
        "evidencia. No inventes resultados ni afirmes que faltan "
        "datos cuando estén presentes en este contexto.\n\n"
        "ESTADO DEL ANÁLISIS:\n"
        f"{serialized_evidence}"
    )


def _build_narrative_state(
    state: AnalysisState,
) -> dict[str, Any]:
    """
    Prepara el estado específico que recibirá el Narrative Agent.

    Args:
        state: Estado compartido del análisis.

    Returns:
        Estado con el contexto narrativo incluido como mensaje.
    """
    narrative_state = _create_specialist_state(state)

    narrative_context = _build_narrative_context(state)

    narrative_state["messages"] = [
        HumanMessage(
            content=narrative_context,
        )
    ]

    return narrative_state


def _build_chart_context(
    state: AnalysisState,
) -> str:
    """
    Construye el contexto necesario para que el Chart Analyst
    determine y ejecute las herramientas de visualización apropiadas.

    El DataFrame original no se incluye en el mensaje porque ya forma
    parte del estado compartido y las herramientas de visualización
    pueden acceder directamente a él mediante ToolRuntime.

    Args:
        state: Estado actual del análisis.

    Returns:
        Contexto estructurado para el Chart Analyst.
    """
    chart_context = {
        "dataset_name": state.get("dataset_name"),
        "user_question": state.get("user_question"),
        "dataset_info": state.get("dataset_info"),
        "dataset_schema": state.get("dataset_schema"),
        "numeric_summary": state.get("numeric_summary"),
        "categorical_summary": state.get(
            "categorical_summary"
        ),
        "correlations": state.get("correlations"),
        "missing_values": state.get("missing_values"),
        "duplicate_info": state.get("duplicate_info"),
        "outliers": state.get("outliers"),
        "sql_query": state.get("sql_query", []),
        "sql_results": state.get("sql_results", []),
    }

    serialized_context = _serialize_context(chart_context)

    return (
        "A continuación se proporciona el contexto del dataset, "
        "los resultados del análisis de calidad y la evidencia SQL "
        "disponibles.\n\n"
        "Debes utilizar este contexto para determinar qué "
        "visualizaciones son relevantes para responder la pregunta "
        "del usuario.\n\n"
        "IMPORTANTE:\n"
        "- Analiza primero la pregunta del usuario.\n"
        "- Si la pregunta requiere una visualización, debes "
        "ejecutar las herramientas de visualización disponibles.\n"
        "- No te limites a recomendar un gráfico en texto.\n"
        "- Las herramientas tienen acceso al DataFrame original "
        "mediante el estado compartido.\n"
        "- Si existe un resultado SQL agregado que responda a la "
        "pregunta, utilízalo como fuente de evidencia para la "
        "visualización.\n"
        "- No recalcules métricas que ya fueron producidas por SQL.\n"
        "- No inventes columnas, métricas ni valores.\n\n"
        "CONTEXTO PARA EL CHART ANALYST:\n"
        f"{serialized_context}"
    )


def _build_chart_state(
    state: AnalysisState,
) -> dict[str, Any]:
    """
    Prepara el estado específico que recibirá el Chart Analyst.

    Args:
        state: Estado compartido del análisis.

    Returns:
        Estado con el contexto de visualización incluido como mensaje.
    """
    chart_state = _create_specialist_state(state)

    chart_context = _build_chart_context(state)

    chart_state["messages"] = [
        HumanMessage(
            content=chart_context,
        )
    ]

    return chart_state


def _build_sql_context(
    state: AnalysisState,
) -> str:
    """
    Construye el contexto que recibirá el SQL Analyst.

    El SQL Analyst necesita conocer explícitamente la pregunta del
    usuario y la estructura real del dataset para poder determinar
    qué consulta debe ejecutar.

    No se incluye el DataFrame completo. El DataFrame permanece en el
    estado compartido y las herramientas SQL acceden a él mediante
    ToolRuntime.

    Args:
        state: Estado actual del análisis.

    Returns:
        Contexto estructurado para el SQL Analyst.
    """
    sql_context = {
        "dataset_name": state.get("dataset_name"),
        "user_question": state.get("user_question"),
        "dataset_info": state.get("dataset_info"),
        "dataset_schema": state.get("dataset_schema"),
        "missing_values": state.get("missing_values"),
        "duplicate_info": state.get("duplicate_info"),
        "numeric_summary": state.get("numeric_summary"),
        "categorical_summary": state.get(
            "categorical_summary"
        ),
        "outliers": state.get("outliers"),
        "correlations": state.get("correlations"),
    }

    serialized_context = _serialize_context(sql_context)

    return (
        "A continuación se proporciona el contexto necesario "
        "para realizar el análisis SQL.\n\n"
        "La pregunta del usuario es el objetivo principal del "
        "análisis. Debes determinar qué operación analítica "
        "responde realmente a esa pregunta utilizando únicamente "
        "las columnas existentes en el dataset.\n\n"
        "IMPORTANTE:\n"
        "- Analiza primero la pregunta del usuario.\n"
        "- Inspecciona el schema real antes de construir la consulta.\n"
        "- Utiliza los nombres de columnas proporcionados por el "
        "schema real.\n"
        "- Utiliza los resultados de calidad como contexto adicional, "
        "no como sustituto de la pregunta.\n"
        "- No inventes columnas, métricas, valores ni resultados.\n"
        "- No ejecutes consultas genéricas de perfilado si la pregunta "
        "requiere una agregación o análisis específico.\n"
        "- Las herramientas SQL tienen acceso al DataFrame original "
        "mediante el estado compartido.\n"
        "- La consulta debe responder directamente a la pregunta "
        "del usuario.\n\n"
        "CONTEXTO PARA EL SQL ANALYST:\n"
        f"{serialized_context}"
    )


def _build_sql_state(
    state: AnalysisState,
) -> dict[str, Any]:
    """
    Prepara el estado específico que recibirá el SQL Analyst.

    Args:
        state: Estado compartido del análisis.

    Returns:
        Estado con el contexto SQL incluido como mensaje.
    """
    sql_state = _create_specialist_state(state)

    sql_context = _build_sql_context(state)

    sql_state["messages"] = [
        HumanMessage(
            content=sql_context,
        )
    ]

    return sql_state


def _build_narrative_state_update(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Construye la actualización del estado generada por el Narrative Agent.

    Args:
        result: Resultado devuelto por el Narrative Agent.

    Returns:
        Actualización con el informe narrativo.

    Raises:
        RuntimeError:
            Si no se puede extraer una respuesta narrativa válida.
    """
    narrative = _extract_narrative(result)

    return {
        "narrative": narrative,
        "final_report": {
            "narrative": narrative,
        },
    }


def _defer_chart_until_sql_available(
    state: AnalysisState,
    tool_call_id: str,
) -> Command | None:
    """Evita ejecutar Chart Analyst antes de disponer de evidencia SQL.

    LangGraph puede ejecutar varias herramientas solicitadas por el
    supervisor dentro de la misma ronda antes de consolidar sus
    actualizaciones de estado. Por ese motivo, Chart Analyst podría
    recibir un estado sin ``sql_results`` aunque SQL Analyst también
    haya sido solicitado en esa misma ronda.

    Si todavía no existe evidencia SQL, la llamada al Chart Analyst se
    difiere de forma explícita. El supervisor recibe un ToolMessage y
    puede continuar con SQL Analyst; en la siguiente ronda Chart Analyst
    encontrará los resultados ya consolidados.

    Args:
        state: Estado compartido del workflow.
        tool_call_id: Identificador de la llamada de herramienta.

    Returns:
        ``Command`` con el mensaje de dependencia si SQL todavía no
        está disponible; en caso contrario, ``None``.
    """
    if state.get("sql_results"):
        return None

    logger.info(
        "Chart Analyst diferido: todavía no existen resultados SQL "
        "consolidados en el estado compartido."
    )

    tool_message = ToolMessage(
        content=(
            "Chart Analyst no puede ejecutarse todavía porque no hay "
            "resultados SQL consolidados en el estado. Ejecuta primero "
            "el SQL Analyst y vuelve a ejecutar Chart Analyst cuando "
            "los resultados SQL estén disponibles."
        ),
        name="chart_analyst",
        tool_call_id=tool_call_id,
    )

    return Command(
        update={
            "messages": [tool_message],
        }
    )


def _run_specialist(
    agent_name: str,
    agents: dict[str, Any],
    state: AnalysisState,
    tool_call_id: str,
) -> Command:
    """
    Ejecuta un agente especializado y devuelve su actualización de estado.

    Args:
        agent_name: Nombre del agente que se ejecutará.
        agents: Registro de agentes especializados.
        state: Estado actual del workflow.
        tool_call_id: Identificador de la llamada de herramienta.

    Returns:
        Command con los resultados del agente.

    Raises:
        ValueError:
            Si el agente no existe o el tool_call_id no es válido.
        RuntimeError:
            Si el agente no devuelve un resultado válido.
    """
    if not tool_call_id:
        raise ValueError(
            "La herramienta requiere un tool_call_id válido."
        )

    agent = agents.get(agent_name)

    if agent is None:
        raise ValueError(
            f"No existe el agente especializado '{agent_name}'."
        )

    logger.info(
        "Ejecutando agente especializado: %s.",
        agent_name,
    )

    if agent_name == "chart_analyst":
        deferred = _defer_chart_until_sql_available(
            state=state,
            tool_call_id=tool_call_id,
        )

        if deferred is not None:
            return deferred

    if agent_name == "narrative_agent":
        specialist_state = _build_narrative_state(state)

    elif agent_name == "chart_analyst":
        specialist_state = _build_chart_state(state)

    elif agent_name == "sql_analyst":
        specialist_state = _build_sql_state(state)

    else:
        specialist_state = _create_specialist_state(state)

    result = agent.invoke(specialist_state)

    if not isinstance(result, dict):
        raise RuntimeError(
            f"El agente '{agent_name}' no devolvió un resultado "
            "en formato diccionario."
        )

    if agent_name == "narrative_agent":
        state_update = _build_narrative_state_update(result)

    else:
        state_update = _build_state_update(
            agent_name=agent_name,
            result=result,
        )

    logger.info(
        "El agente '%s' produjo %s actualización(es) de estado.",
        agent_name,
        len(state_update),
    )

    tool_message = ToolMessage(
        content=(
            f"El agente '{agent_name}' completó "
            "la tarea correctamente."
        ),
        name=agent_name,
        tool_call_id=tool_call_id,
    )

    return Command(
        update={
            **state_update,
            "messages": [tool_message],
        }
    )


def create_agent_tools(
    agents: dict[str, Any],
) -> list[Any]:
    """
    Crea las herramientas utilizadas por el supervisor.

    Args:
        agents: Registro de agentes especializados.

    Returns:
        Lista de herramientas de delegación.
    """

    @tool(
        "call_data_quality_agent",
        description=(
            "Ejecuta el Data Quality Agent para analizar "
            "la estructura y calidad del dataset."
        ),
    )
    def call_data_quality_agent(
        runtime: ToolRuntime,
    ) -> Command:
        """Ejecuta el Data Quality Agent."""
        logger.info(
            "Supervisor delegando tarea al Data Quality Agent."
        )

        return _run_specialist(
            agent_name="data_quality_agent",
            agents=agents,
            state=runtime.state,
            tool_call_id=runtime.tool_call_id,
        )

    @tool(
        "call_sql_agent",
        description=(
            "Ejecuta el SQL Analyst para realizar consultas "
            "de lectura sobre el dataset."
        ),
    )
    def call_sql_agent(
        runtime: ToolRuntime,
    ) -> Command:
        """Ejecuta el SQL Analyst."""
        logger.info(
            "Supervisor delegando tarea al SQL Analyst."
        )

        return _run_specialist(
            agent_name="sql_analyst",
            agents=agents,
            state=runtime.state,
            tool_call_id=runtime.tool_call_id,
        )

    @tool(
        "call_chart_agent",
        description=(
            "Ejecuta el Chart Analyst para preparar datos de "
            "visualización relevantes. Requiere que SQL Analyst haya "
            "terminado y que existan resultados SQL consolidados en "
            "el estado. Si todavía no existen, ejecuta primero "
            "call_sql_agent y después vuelve a llamar a esta herramienta."
        ),
    )
    def call_chart_agent(
        runtime: ToolRuntime,
    ) -> Command:
        """Ejecuta Chart Analyst después de disponer de SQL."""
        logger.info(
            "Supervisor delegando tarea al Chart Analyst."
        )

        return _run_specialist(
            agent_name="chart_analyst",
            agents=agents,
            state=runtime.state,
            tool_call_id=runtime.tool_call_id,
        )

    @tool(
        "call_narrative_agent",
        description=(
            "Ejecuta el Narrative Agent para generar el "
            "informe final utilizando la evidencia acumulada."
        ),
    )
    def call_narrative_agent(
        runtime: ToolRuntime,
    ) -> Command:
        """Ejecuta el Narrative Agent."""
        logger.info(
            "Supervisor delegando tarea al Narrative Agent."
        )

        return _run_specialist(
            agent_name="narrative_agent",
            agents=agents,
            state=runtime.state,
            tool_call_id=runtime.tool_call_id,
        )

    return [
        call_data_quality_agent,
        call_sql_agent,
        call_chart_agent,
        call_narrative_agent,
    ]