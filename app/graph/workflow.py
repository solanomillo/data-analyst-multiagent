"""Workflow principal del sistema de análisis multi-agente."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.agents.supervisor import create_supervisor_agent
from app.graph.state import AnalysisState


logger = logging.getLogger(__name__)


class WorkflowError(Exception):
    """Error controlado relacionado con la ejecución del workflow."""


def create_initial_state(
    dataframe: pd.DataFrame,
    dataset_name: str,
    user_question: str,
) -> AnalysisState:
    """
    Crea el estado inicial del análisis.

    Args:
        dataframe: Dataset que será analizado.
        dataset_name: Nombre del archivo del dataset.
        user_question: Pregunta realizada por el usuario.

    Returns:
        Estado inicial compatible con los agentes.

    Raises:
        WorkflowError:
            Si los datos de entrada no son válidos.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise WorkflowError(
            "El dataset debe ser un objeto pandas.DataFrame."
        )

    if dataframe.empty:
        raise WorkflowError(
            "No se puede iniciar el análisis con un dataset vacío."
        )

    if not dataset_name.strip():
        raise WorkflowError(
            "El nombre del dataset no puede estar vacío."
        )

    if not user_question.strip():
        raise WorkflowError(
            "La pregunta del usuario no puede estar vacía."
        )

    logger.info(
        "Creando estado inicial para '%s'.",
        dataset_name,
    )

    return AnalysisState(
        dataset=dataframe,
        dataset_name=dataset_name,
        user_question=user_question,
        dataset_info={
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
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


def create_analysis_workflow() -> Any:
    """
    Crea el workflow principal del sistema.

    Returns:
        Supervisor configurado como punto de entrada del análisis.
    """
    logger.info(
        "Creando workflow principal de análisis."
    )

    return create_supervisor_agent()


def run_analysis(
    dataframe: pd.DataFrame,
    dataset_name: str,
    user_question: str,
) -> dict[str, Any]:
    """
    Ejecuta un análisis completo sobre un dataset.

    Args:
        dataframe: Dataset que será analizado.
        dataset_name: Nombre del archivo del dataset.
        user_question: Pregunta realizada por el usuario.

    Returns:
        Estado final producido por el workflow.

    Raises:
        WorkflowError:
            Si ocurre un error durante la ejecución.
    """
    initial_state = create_initial_state(
        dataframe=dataframe,
        dataset_name=dataset_name,
        user_question=user_question,
    )

    logger.info(
        "Iniciando análisis del dataset '%s'.",
        dataset_name,
    )

    try:
        workflow = create_analysis_workflow()
        result = workflow.invoke(initial_state)

    except Exception as error:
        logger.exception(
            "Error durante la ejecución del workflow."
        )

        raise WorkflowError(
            "No fue posible completar el análisis del dataset."
        ) from error

    if not isinstance(result, dict):
        logger.error(
            "El workflow devolvió un resultado inválido."
        )

        raise WorkflowError(
            "El workflow no devolvió un estado válido."
        )

    logger.info(
        "Análisis finalizado correctamente para '%s'.",
        dataset_name,
    )

    return result