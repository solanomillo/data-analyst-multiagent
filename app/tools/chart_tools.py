"""Herramientas determinísticas para preparar datos de visualización."""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command


logger = logging.getLogger(__name__)


def _get_dataframe(
    runtime: ToolRuntime,
) -> pd.DataFrame:
    """
    Obtiene el DataFrame almacenado en el estado del agente.

    Args:
        runtime: Contexto de ejecución de la herramienta.

    Returns:
        DataFrame almacenado en el estado.

    Raises:
        ValueError:
            Si no existe un DataFrame válido en el estado.
    """
    dataframe = runtime.state.get("dataset")

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError(
            "No existe un DataFrame válido en el estado."
        )

    return dataframe


def get_numeric_distribution(
    dataframe: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """
    Prepara una distribución numérica para visualización.

    Args:
        dataframe: Dataset que será analizado.
        column: Nombre de la columna numérica.

    Returns:
        Datos necesarios para representar la distribución.

    Raises:
        ValueError:
            Si la columna no existe o no es numérica.
    """
    if column not in dataframe.columns:
        raise ValueError(
            f"La columna '{column}' no existe en el dataset."
        )

    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise ValueError(
            f"La columna '{column}' no es numérica."
        )

    values = dataframe[column].dropna().tolist()

    return {
        "column": column,
        "type": "numeric_distribution",
        "count": len(values),
        "values": [
            float(value)
            for value in values
        ],
    }


def get_categorical_distribution(
    dataframe: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """
    Obtiene las frecuencias de una variable categórica.

    Args:
        dataframe: Dataset que será analizado.
        column: Nombre de la columna categórica.

    Returns:
        Frecuencias de los valores de la columna.

    Raises:
        ValueError:
            Si la columna no existe.
    """
    if column not in dataframe.columns:
        raise ValueError(
            f"La columna '{column}' no existe en el dataset."
        )

    value_counts = (
        dataframe[column]
        .value_counts(dropna=False)
        .head(20)
    )

    values = {}

    for value, count in value_counts.items():
        key = "<NA>" if pd.isna(value) else str(value)
        values[key] = int(count)

    return {
        "column": column,
        "type": "categorical_distribution",
        "values": values,
    }


def get_correlation_matrix(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Prepara una matriz de correlaciones numéricas.

    Args:
        dataframe: Dataset que será analizado.

    Returns:
        Matriz de correlaciones.

    Raises:
        ValueError:
            Si existen menos de dos columnas numéricas.
    """
    numeric_dataframe = dataframe.select_dtypes(
        include="number",
    )

    if numeric_dataframe.shape[1] < 2:
        raise ValueError(
            "Se necesitan al menos dos columnas numéricas "
            "para calcular correlaciones."
        )

    correlation_matrix = numeric_dataframe.corr()

    return {
        "type": "correlation_matrix",
        "columns": list(correlation_matrix.columns),
        "values": {
            column: {
                other_column: round(
                    float(
                        correlation_matrix.loc[
                            column,
                            other_column,
                        ]
                    ),
                    4,
                )
                for other_column in correlation_matrix.columns
            }
            for column in correlation_matrix.columns
        },
    }


def get_scatter_data(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> dict[str, Any]:
    """
    Prepara datos para un gráfico de dispersión.

    Args:
        dataframe: Dataset que será analizado.
        x_column: Columna utilizada en el eje X.
        y_column: Columna utilizada en el eje Y.

    Returns:
        Datos necesarios para construir el gráfico.

    Raises:
        ValueError:
            Si alguna columna no existe o no es numérica.
    """
    for column in (x_column, y_column):
        if column not in dataframe.columns:
            raise ValueError(
                f"La columna '{column}' no existe en el dataset."
            )

        if not pd.api.types.is_numeric_dtype(
            dataframe[column],
        ):
            raise ValueError(
                f"La columna '{column}' no es numérica."
            )

    selected = dataframe[
        [x_column, y_column]
    ].dropna()

    return {
        "type": "scatter",
        "x_column": x_column,
        "y_column": y_column,
        "count": len(selected),
        "x_values": selected[x_column].tolist(),
        "y_values": selected[y_column].tolist(),
    }


def get_chart_metadata(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Obtiene información para seleccionar visualizaciones.

    Args:
        dataframe: Dataset que será analizado.

    Returns:
        Metadatos sobre columnas numéricas y categóricas.
    """
    numeric_columns = list(
        dataframe.select_dtypes(
            include="number",
        ).columns
    )

    categorical_columns = list(
        dataframe.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ],
        ).columns
    )

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "total_columns": len(dataframe.columns),
    }


def _create_chart_tool_command(
    result: dict[str, Any],
    runtime: ToolRuntime,
) -> Command:
    """
    Crea un Command para persistir un resultado de visualización.

    Cada ejecución de una herramienta devuelve únicamente su propio
    resultado. La acumulación de resultados es responsabilidad del
    reducer definido en AnalysisState.

    Args:
        result: Resultado generado por una herramienta determinística.
        runtime: Contexto de ejecución de la herramienta.

    Returns:
        Command con el resultado y el mensaje de herramienta.

    Raises:
        ValueError:
            Si no existe un tool_call_id válido.
    """
    if not runtime.tool_call_id:
        raise ValueError(
            "La herramienta requiere un tool_call_id válido."
        )

    tool_message = ToolMessage(
        content=json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        ),
        tool_call_id=runtime.tool_call_id,
    )

    return Command(
        update={
            "chart_results": [result],
            "messages": [tool_message],
        }
    )


@tool
def inspect_numeric_distribution(
    column: str,
    runtime: ToolRuntime,
) -> Command:
    """
    Obtiene y persiste la distribución de una variable numérica.

    Args:
        column: Nombre de la columna numérica.
        runtime: Contexto de ejecución del agente.

    Returns:
        Command con el resultado persistido en chart_results.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Preparando distribución numérica de '%s'.",
        column,
    )

    result = get_numeric_distribution(
        dataframe=dataframe,
        column=column,
    )

    return _create_chart_tool_command(
        result=result,
        runtime=runtime,
    )


@tool
def inspect_categorical_distribution(
    column: str,
    runtime: ToolRuntime,
) -> Command:
    """
    Obtiene y persiste la distribución de una variable categórica.

    Args:
        column: Nombre de la columna categórica.
        runtime: Contexto de ejecución del agente.

    Returns:
        Command con el resultado persistido en chart_results.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Preparando distribución categórica de '%s'.",
        column,
    )

    result = get_categorical_distribution(
        dataframe=dataframe,
        column=column,
    )

    return _create_chart_tool_command(
        result=result,
        runtime=runtime,
    )


@tool
def inspect_correlation_matrix(
    runtime: ToolRuntime,
) -> Command:
    """
    Obtiene y persiste la matriz de correlación.

    Args:
        runtime: Contexto de ejecución de la herramienta.

    Returns:
        Command con el resultado persistido en chart_results.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Preparando matriz de correlación.",
    )

    result = get_correlation_matrix(
        dataframe=dataframe,
    )

    return _create_chart_tool_command(
        result=result,
        runtime=runtime,
    )


@tool
def inspect_scatter_data(
    x_column: str,
    y_column: str,
    runtime: ToolRuntime,
) -> Command:
    """
    Obtiene y persiste los datos para un gráfico de dispersión.

    Args:
        x_column: Columna utilizada en el eje X.
        y_column: Columna utilizada en el eje Y.
        runtime: Contexto de ejecución del agente.

    Returns:
        Command con el resultado persistido en chart_results.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Preparando dispersión entre '%s' y '%s'.",
        x_column,
        y_column,
    )

    result = get_scatter_data(
        dataframe=dataframe,
        x_column=x_column,
        y_column=y_column,
    )

    return _create_chart_tool_command(
        result=result,
        runtime=runtime,
    )