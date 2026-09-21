"""Herramientas determinísticas para análisis exploratorio de datos."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


def get_dataset_schema(dataframe: pd.DataFrame) -> dict[str, Any]:
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
                "non_null": int(dataframe[column].notna().sum()),
                "nulls": int(dataframe[column].isna().sum()),
                "unique_values": int(
                    dataframe[column].nunique(dropna=True)
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
    logger.info("Calculando estadísticas descriptivas numéricas.")

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
        include=["object", "category", "bool"]
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
                dataframe[column].nunique(dropna=True)
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
    logger.info("Calculando correlaciones entre variables numéricas.")

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
                float(correlation_matrix.loc[column, other_column]),
                4,
            )
            for other_column in correlation_matrix.columns
        }

    return result