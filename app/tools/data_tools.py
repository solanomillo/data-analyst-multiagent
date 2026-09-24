"""Herramientas determinísticas para análisis exploratorio de datos."""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command


logger = logging.getLogger(__name__)


def get_dataset_schema(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Obtiene información estructural del dataset.

    Args:
        dataframe: DataFrame que se desea analizar.

    Returns:
        Diccionario con dimensiones y tipos de datos.
    """
    logger.info("Obteniendo esquema del dataset.")

    columns = []

    for column in dataframe.columns:
        columns.append(
            {
                "name": column,
                "dtype": str(dataframe[column].dtype),
                "non_null": int(
                    dataframe[column].notna().sum()
                ),
                "nulls": int(
                    dataframe[column].isna().sum()
                ),
                "unique_values": int(
                    dataframe[column].nunique(
                        dropna=True
                    )
                ),
            }
        )

    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_details": columns,
    }


def check_missing_values(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Analiza los valores nulos del dataset.

    Args:
        dataframe: DataFrame que se desea analizar.

    Returns:
        Información sobre valores nulos por columna.
    """
    logger.info("Analizando valores nulos.")

    null_counts = dataframe.isna().sum()
    total_rows = len(dataframe)

    details = {}

    for column, count in null_counts.items():
        count = int(count)

        details[column] = {
            "count": count,
            "percentage": (
                round((count / total_rows) * 100, 2)
                if total_rows > 0
                else 0.0
            ),
        }

    total_nulls = int(null_counts.sum())

    return {
        "total_null_values": total_nulls,
        "columns_with_nulls": {
            column: data
            for column, data in details.items()
            if data["count"] > 0
        },
    }


def check_duplicates(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Detecta registros duplicados.

    Args:
        dataframe: DataFrame que se desea analizar.

    Returns:
        Información sobre registros duplicados.
    """
    logger.info("Analizando registros duplicados.")

    duplicate_mask = dataframe.duplicated()
    duplicate_count = int(duplicate_mask.sum())
    total_rows = len(dataframe)

    percentage = (
        round((duplicate_count / total_rows) * 100, 2)
        if total_rows > 0
        else 0.0
    )

    return {
        "duplicate_rows": duplicate_count,
        "percentage": percentage,
    }


def get_numeric_summary(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """
    Obtiene estadísticas descriptivas de columnas numéricas.

    Args:
        dataframe: DataFrame que se desea analizar.

    Returns:
        Estadísticas descriptivas por columna numérica.
    """
    logger.info(
        "Calculando estadísticas descriptivas numéricas."
    )

    numeric_dataframe = dataframe.select_dtypes(
        include="number"
    )

    if numeric_dataframe.empty:
        return {}

    summary = numeric_dataframe.describe().transpose()

    result = {}

    for column in summary.index:
        result[column] = {
            "count": float(summary.loc[column, "count"]),
            "mean": float(summary.loc[column, "mean"]),
            "std": float(summary.loc[column, "std"]),
            "min": float(summary.loc[column, "min"]),
            "25%": float(summary.loc[column, "25%"]),
            "50%": float(summary.loc[column, "50%"]),
            "75%": float(summary.loc[column, "75%"]),
            "max": float(summary.loc[column, "max"]),
        }

    return result


def get_categorical_summary(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """
    Obtiene información básica de columnas categóricas.

    Args:
        dataframe: DataFrame que se desea analizar.

    Returns:
        Valores únicos y frecuencias de las columnas categóricas.
    """
    logger.info("Analizando variables categóricas.")

    categorical_dataframe = dataframe.select_dtypes(
        include=[
            "object",
            "string",
            "category",
            "bool",
        ]
    )

    result = {}

    for column in categorical_dataframe.columns:
        value_counts = (
            dataframe[column]
            .value_counts(dropna=False)
            .head(10)
        )

        values = {}

        for value, count in value_counts.items():
            if pd.isna(value):
                key = "<NA>"
            else:
                key = str(value)

            values[key] = int(count)

        result[column] = {
            "unique_values": int(
                dataframe[column].nunique(
                    dropna=True
                )
            ),
            "top_values": values,
        }

    return result


def detect_outliers(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """
    Detecta posibles valores atípicos utilizando el método IQR.

    El resultado identifica observaciones que se encuentran por
    debajo de Q1 - 1.5 * IQR o por encima de Q3 + 1.5 * IQR.

    Args:
        dataframe: DataFrame que se desea analizar.

    Returns:
        Información sobre posibles outliers por columna numérica.
    """
    logger.info("Detectando posibles valores atípicos.")

    numeric_dataframe = dataframe.select_dtypes(
        include="number"
    )

    result = {}

    for column in numeric_dataframe.columns:
        series = numeric_dataframe[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outlier_mask = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        outlier_count = int(outlier_mask.sum())

        result[column] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": outlier_count,
            "outlier_percentage": round(
                (outlier_count / len(series)) * 100,
                2,
            ),
        }

    return result


def calculate_correlations(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """
    Calcula las correlaciones entre variables numéricas.

    Utiliza el coeficiente de correlación de Pearson proporcionado
    por pandas.

    Args:
        dataframe: DataFrame que se desea analizar.

    Returns:
        Matriz de correlaciones representada como diccionario.
    """
    logger.info(
        "Calculando correlaciones entre variables numéricas."
    )

    numeric_dataframe = dataframe.select_dtypes(
        include="number"
    )

    if numeric_dataframe.shape[1] < 2:
        return {}

    correlation_matrix = numeric_dataframe.corr()

    result = {}

    for column in correlation_matrix.columns:
        result[column] = {
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

    return result


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
            Si el estado no contiene un DataFrame válido.
    """
    dataframe = runtime.state.get("dataset")

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError(
            "No existe un DataFrame válido en el estado."
        )

    return dataframe


def _create_tool_command(
    result: Any,
    state_field: str,
    tool_call_id: str | None,
) -> Command:
    """
    Crea el Command utilizado para actualizar el estado del agente.

    Args:
        result: Resultado calculado por la herramienta.
        state_field: Campo de AnalysisState que debe actualizarse.
        tool_call_id: Identificador de la llamada de herramienta.

    Returns:
        Command con el resultado y el ToolMessage correspondiente.

    Raises:
        ValueError:
            Si no existe un identificador de llamada de herramienta.
    """
    if not tool_call_id:
        raise ValueError(
            "La herramienta requiere un tool_call_id válido."
        )

    message_content = json.dumps(
        result,
        ensure_ascii=False,
        default=str,
    )

    tool_message = ToolMessage(
        content=message_content,
        tool_call_id=tool_call_id,
    )

    return Command(
        update={
            state_field: result,
            "messages": [tool_message],
        }
    )


@tool
def inspect_dataset_schema(
    runtime: ToolRuntime,
) -> Command:
    """
    Obtiene la estructura del dataset y actualiza el estado.

    Args:
        runtime: Contexto de ejecución del agente.

    Returns:
        Command que actualiza dataset_schema.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Ejecutando herramienta inspect_dataset_schema."
    )

    result = get_dataset_schema(dataframe)

    return _create_tool_command(
        result=result,
        state_field="dataset_schema",
        tool_call_id=runtime.tool_call_id,
    )


@tool
def inspect_missing_values(
    runtime: ToolRuntime,
) -> Command:
    """
    Analiza valores nulos y actualiza el estado.

    Args:
        runtime: Contexto de ejecución del agente.

    Returns:
        Command que actualiza missing_values.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Ejecutando herramienta inspect_missing_values."
    )

    result = check_missing_values(dataframe)

    return _create_tool_command(
        result=result,
        state_field="missing_values",
        tool_call_id=runtime.tool_call_id,
    )


@tool
def inspect_duplicates(
    runtime: ToolRuntime,
) -> Command:
    """
    Analiza registros duplicados y actualiza el estado.

    Args:
        runtime: Contexto de ejecución del agente.

    Returns:
        Command que actualiza duplicate_info.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Ejecutando herramienta inspect_duplicates."
    )

    result = check_duplicates(dataframe)

    return _create_tool_command(
        result=result,
        state_field="duplicate_info",
        tool_call_id=runtime.tool_call_id,
    )


@tool
def inspect_numeric_statistics(
    runtime: ToolRuntime,
) -> Command:
    """
    Obtiene estadísticas numéricas y actualiza el estado.

    Args:
        runtime: Contexto de ejecución del agente.

    Returns:
        Command que actualiza numeric_summary.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Ejecutando herramienta inspect_numeric_statistics."
    )

    result = get_numeric_summary(dataframe)

    return _create_tool_command(
        result=result,
        state_field="numeric_summary",
        tool_call_id=runtime.tool_call_id,
    )


@tool
def inspect_categorical_statistics(
    runtime: ToolRuntime,
) -> Command:
    """
    Analiza variables categóricas y actualiza el estado.

    Args:
        runtime: Contexto de ejecución del agente.

    Returns:
        Command que actualiza categorical_summary.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Ejecutando herramienta inspect_categorical_statistics."
    )

    result = get_categorical_summary(dataframe)

    return _create_tool_command(
        result=result,
        state_field="categorical_summary",
        tool_call_id=runtime.tool_call_id,
    )


@tool
def inspect_outliers(
    runtime: ToolRuntime,
) -> Command:
    """
    Detecta posibles outliers y actualiza el estado.

    Args:
        runtime: Contexto de ejecución del agente.

    Returns:
        Command que actualiza outliers.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Ejecutando herramienta inspect_outliers."
    )

    result = detect_outliers(dataframe)

    return _create_tool_command(
        result=result,
        state_field="outliers",
        tool_call_id=runtime.tool_call_id,
    )


@tool
def inspect_correlations(
    runtime: ToolRuntime,
) -> Command:
    """
    Calcula correlaciones y actualiza el estado.

    Args:
        runtime: Contexto de ejecución del agente.

    Returns:
        Command que actualiza correlations.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Ejecutando herramienta inspect_correlations."
    )

    result = calculate_correlations(dataframe)

    return _create_tool_command(
        result=result,
        state_field="correlations",
        tool_call_id=runtime.tool_call_id,
    )