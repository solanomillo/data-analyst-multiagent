"""Herramientas para generar información de visualizaciones."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


def get_numeric_histograms(
    dataframe: pd.DataFrame,
    max_columns: int = 6,
) -> dict[str, pd.Series]:
    """
    Obtiene las series numéricas necesarias para construir histogramas.

    Args:
        dataframe: DataFrame que se desea analizar.
        max_columns: Cantidad máxima de columnas a procesar.

    Returns:
        Diccionario donde cada clave es una columna numérica y
        su valor es la serie correspondiente sin valores nulos.

    Raises:
        ValueError: Si max_columns es menor que 1.
    """
    if max_columns < 1:
        raise ValueError(
            "max_columns debe ser mayor o igual a 1."
        )

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns[:max_columns]

    histograms = {}

    for column in numeric_columns:
        series = dataframe[column].dropna()

        if series.empty:
            continue

        histograms[column] = series

    logger.info(
        "Se prepararon %d histogramas.",
        len(histograms),
    )

    return histograms


def get_categorical_counts(
    dataframe: pd.DataFrame,
    max_columns: int = 4,
    max_categories: int = 10,
) -> dict[str, pd.Series]:
    """
    Obtiene frecuencias de las principales categorías.

    Args:
        dataframe: DataFrame que se desea analizar.
        max_columns: Cantidad máxima de columnas categóricas.
        max_categories: Cantidad máxima de categorías por columna.

    Returns:
        Diccionario con las frecuencias de las principales categorías.

    Raises:
        ValueError: Si alguno de los límites es menor que 1.
    """
    if max_columns < 1:
        raise ValueError(
            "max_columns debe ser mayor o igual a 1."
        )

    if max_categories < 1:
        raise ValueError(
            "max_categories debe ser mayor o igual a 1."
        )

    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "bool"]
    ).columns[:max_columns]

    result = {}

    for column in categorical_columns:
        counts = (
            dataframe[column]
            .value_counts(dropna=False)
            .head(max_categories)
        )

        if not counts.empty:
            result[column] = counts

    logger.info(
        "Se prepararon distribuciones para %d variables categóricas.",
        len(result),
    )

    return result


def get_correlation_matrix(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Obtiene la matriz de correlación de variables numéricas.

    Args:
        dataframe: DataFrame que se desea analizar.

    Returns:
        DataFrame con la matriz de correlaciones.

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

    logger.info(
        "Generando matriz de correlaciones para %d variables.",
        numeric_dataframe.shape[1],
    )

    return numeric_dataframe.corr()


def get_scatter_data(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> pd.DataFrame:
    """
    Obtiene los datos necesarios para un gráfico de dispersión.

    Args:
        dataframe: DataFrame que contiene los datos.
        x_column: Columna que se utilizará en el eje X.
        y_column: Columna que se utilizará en el eje Y.

    Returns:
        DataFrame con las dos variables seleccionadas.

    Raises:
        ValueError: Si alguna columna no existe o no es numérica.
    """
    required_columns = {x_column, y_column}

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Las siguientes columnas no existen: "
            f"{', '.join(sorted(missing_columns))}."
        )

    for column in (x_column, y_column):
        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        ):
            raise ValueError(
                f"La columna '{column}' debe ser numérica."
            )

    result = dataframe[[x_column, y_column]].dropna()

    logger.info(
        "Datos de dispersión preparados: %s vs %s.",
        x_column,
        y_column,
    )

    return result


def get_chart_metadata(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Obtiene información sobre las visualizaciones disponibles.

    Args:
        dataframe: DataFrame que se desea analizar.

    Returns:
        Diccionario con las variables disponibles para gráficos.
    """
    numeric_columns = list(
        dataframe.select_dtypes(
            include="number"
        ).columns
    )

    categorical_columns = list(
        dataframe.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
    )

    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_count": len(numeric_columns),
        "categorical_count": len(categorical_columns),
    }