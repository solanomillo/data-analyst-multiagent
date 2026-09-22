"""Herramientas SQL para el análisis de datasets."""

from __future__ import annotations
import logging
import sqlite3
from typing import Any
from langchain.tools import ToolRuntime, tool
import pandas as pd


logger = logging.getLogger(__name__)


SQL_TABLE_NAME = "dataset"


def dataframe_to_sqlite(
    dataframe: pd.DataFrame,
) -> sqlite3.Connection:
    """
    Crea una base SQLite en memoria a partir de un DataFrame.

    Args:
        dataframe: DataFrame que se desea registrar.

    Returns:
        Conexión SQLite en memoria con el dataset cargado.

    Raises:
        ValueError: Si el DataFrame no contiene datos.
    """
    if dataframe.empty:
        raise ValueError(
            "No se puede crear una tabla SQL a partir de "
            "un DataFrame vacío."
        )

    connection = sqlite3.connect(":memory:")

    try:
        dataframe.to_sql(
            SQL_TABLE_NAME,
            connection,
            index=False,
            if_exists="replace",
        )

    except Exception:
        connection.close()

        logger.exception(
            "No fue posible cargar el DataFrame en SQLite."
        )

        raise

    logger.info(
        "Dataset cargado en SQLite con %d filas.",
        len(dataframe),
    )

    return connection


def get_sql_schema(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Obtiene el esquema SQL del dataset.

    Args:
        dataframe: DataFrame que se desea inspeccionar.

    Returns:
        Información sobre las columnas disponibles.
    """
    connection = dataframe_to_sqlite(dataframe)

    try:
        cursor = connection.execute(
            f"PRAGMA table_info({SQL_TABLE_NAME})"
        )

        columns = []

        for row in cursor.fetchall():
            columns.append(
                {
                    "column_id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "nullable": not bool(row[3]),
                }
            )

        return {
            "table_name": SQL_TABLE_NAME,
            "columns": columns,
        }

    finally:
        connection.close()


def validate_sql_query(query: str) -> str:
    """
    Valida las características básicas de una consulta SQL.

    Solamente se permiten consultas de lectura.

    Args:
        query: Consulta SQL generada por el agente.

    Returns:
        Consulta normalizada.

    Raises:
        ValueError: Si la consulta está vacía o contiene operaciones
            de modificación.
    """
    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError(
            "La consulta SQL no puede estar vacía."
        )

    if normalized_query.endswith(";"):
        normalized_query = normalized_query[:-1].strip()

    forbidden_keywords = (
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "REPLACE",
        "TRUNCATE",
        "ATTACH",
        "DETACH",
    )

    first_keyword = (
        normalized_query.split(maxsplit=1)[0].upper()
    )

    if first_keyword not in {
        "SELECT",
        "WITH",
    }:
        raise ValueError(
            "Solo se permiten consultas SQL de lectura "
            "utilizando SELECT o WITH."
        )

    upper_query = normalized_query.upper()

    for keyword in forbidden_keywords:
        if keyword in upper_query:
            raise ValueError(
                f"La consulta contiene una operación SQL no permitida: "
                f"{keyword}."
            )

    return normalized_query


def execute_sql_query(
    dataframe: pd.DataFrame,
    query: str,
) -> dict[str, Any]:
    """
    Ejecuta una consulta SQL de lectura sobre el dataset.

    Args:
        dataframe: DataFrame que contiene los datos.
        query: Consulta SQL a ejecutar.

    Returns:
        Resultado de la consulta y metadatos de ejecución.

    Raises:
        ValueError: Si la consulta no es válida.
        RuntimeError: Si SQLite no puede ejecutar la consulta.
    """
    validated_query = validate_sql_query(query)

    connection = dataframe_to_sqlite(dataframe)

    try:
        result = pd.read_sql_query(
            validated_query,
            connection,
        )

    except sqlite3.Error as error:
        logger.exception(
            "Error ejecutando consulta SQL."
        )

        raise RuntimeError(
            f"No fue posible ejecutar la consulta SQL: {error}"
        ) from error

    finally:
        connection.close()

    records = result.to_dict(orient="records")

    logger.info(
        "Consulta SQL ejecutada correctamente. Filas obtenidas: %d.",
        len(records),
    )

    return {
        "query": validated_query,
        "row_count": len(records),
        "columns": list(result.columns),
        "rows": records,
    }
    

@tool
def inspect_sql_schema(
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """
    Obtiene el esquema SQL del dataset actual.

    Args:
        runtime: Contexto de ejecución del agente.

    Returns:
        Esquema SQL disponible para realizar consultas.
    """
    dataframe = runtime.state.get("dataset")

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError(
            "No existe un DataFrame válido en el estado."
        )

    return get_sql_schema(dataframe)


@tool
def run_sql_query(
    query: str,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """
    Ejecuta una consulta SQL de lectura sobre el dataset actual.

    Args:
        query: Consulta SQL generada por el agente.
        runtime: Contexto de ejecución del agente.

    Returns:
        Resultado de la consulta SQL.
    """
    dataframe = runtime.state.get("dataset")

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError(
            "No existe un DataFrame válido en el estado."
        )

    return execute_sql_query(
        dataframe,
        query,
    )
