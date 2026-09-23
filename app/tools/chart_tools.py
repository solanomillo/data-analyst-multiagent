"""Herramientas determinísticas para preparar datos de visualización."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from langchain.tools import ToolRuntime, tool

logger = logging.getLogger(__name__)


def _get_dataframe(runtime: ToolRuntime) -> pd.DataFrame:
    """
    Obtiene el DataFrame almacenado en el estado del agente.

    Args:
        runtime: Runtime proporcionado por LangChain.

    Returns:
        DataFrame actual del análisis.

    Raises:
        ValueError: Si no existe un DataFrame válido.
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
        dataframe: DataFrame que contiene los datos.
        column: Nombre de la columna numérica.

    Returns:
        Valores numéricos de la columna.

    Raises:
        ValueError: Si la columna no existe o no es numérica.
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
        dataframe: DataFrame que contiene los datos.
        column: Nombre de la columna categórica.

    Returns:
        Frecuencias de los valores de la columna.

    Raises:
        ValueError: Si la columna no existe.
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
        dataframe: DataFrame que contiene los datos.

    Returns:
        Matriz de correlación como diccionario.

    Raises:
        ValueError: Si existen menos de dos columnas numéricas.
    """
    numeric_dataframe = dataframe.select_dtypes(
        include="number"
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
        dataframe: DataFrame que contiene los datos.
        x_column: Variable del eje X.
        y_column: Variable del eje Y.

    Returns:
        Datos de ambas variables para visualización.

    Raises:
        ValueError: Si alguna columna no existe o no es numérica.
    """
    for column in (x_column, y_column):
        if column not in dataframe.columns:
            raise ValueError(
                f"La columna '{column}' no existe en el dataset."
            )

        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
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
    Obtiene información que ayuda a seleccionar visualizaciones.

    Args:
        dataframe: DataFrame que se desea analizar.

    Returns:
        Metadatos de columnas numéricas y categóricas.
    """
    numeric_columns = list(
        dataframe.select_dtypes(
            include="number"
        ).columns
    )

    categorical_columns = list(
        dataframe.select_dtypes(
            include=[
                "object",
                "string",
                "category",
                "bool",
            ]
        ).columns
    )

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "total_columns": len(dataframe.columns),
    }


@tool
def inspect_numeric_distribution(
    column: str,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """
    Obtiene los valores de una variable numérica para visualizar
    su distribución.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Preparando distribución numérica de '%s'.",
        column,
    )

    return get_numeric_distribution(
        dataframe,
        column,
    )


@tool
def inspect_categorical_distribution(
    column: str,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """
    Obtiene las frecuencias de una variable categórica.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Preparando distribución categórica de '%s'.",
        column,
    )

    return get_categorical_distribution(
        dataframe,
        column,
    )


@tool
def inspect_correlation_matrix(
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """
    Obtiene la matriz de correlación del dataset.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Preparando matriz de correlación."
    )

    return get_correlation_matrix(dataframe)


@tool
def inspect_scatter_data(
    x_column: str,
    y_column: str,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """
    Obtiene los datos necesarios para un gráfico de dispersión.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Preparando dispersión entre '%s' y '%s'.",
        x_column,
        y_column,
    )

    return get_scatter_data(
        dataframe,
        x_column,
        y_column,
    )