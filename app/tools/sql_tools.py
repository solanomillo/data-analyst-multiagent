"""Herramientas SQL para el análisis de datasets."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import pandas as pd
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command


logger = logging.getLogger(__name__)

SQL_TABLE_NAME = "dataset"


def dataframe_to_sqlite(
    dataframe: pd.DataFrame,
) -> sqlite3.Connection:
    """
    Crea una base SQLite en memoria a partir de un DataFrame.

    Args:
        dataframe: DataFrame que será cargado en SQLite.

    Returns:
        Conexión SQLite en memoria.

    Raises:
        ValueError:
            Si el DataFrame está vacío.
        RuntimeError:
            Si ocurre un error al cargar los datos.
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

    except Exception as error:
        connection.close()

        logger.exception(
            "No fue posible cargar el DataFrame en SQLite."
        )

        raise RuntimeError(
            "No fue posible cargar el dataset en SQLite."
        ) from error

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
        dataframe: DataFrame que será inspeccionado.

    Returns:
        Información de la tabla y sus columnas.
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


def validate_sql_query(
    query: str,
) -> str:
    """
    Valida las características básicas de una consulta SQL.

    La validación comprueba que la consulta no esté vacía y que
    comience con SELECT o WITH. La protección contra operaciones
    de modificación se refuerza durante la ejecución mediante
    el autorizer de SQLite.

    Args:
        query: Consulta SQL que se desea validar.

    Returns:
        Consulta normalizada sin punto y coma final.

    Raises:
        ValueError:
            Si la consulta está vacía o no corresponde a una
            consulta de lectura.
    """
    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError(
            "La consulta SQL no puede estar vacía."
        )

    if normalized_query.endswith(";"):
        normalized_query = normalized_query[:-1].strip()

    first_keyword = (
        normalized_query.split(maxsplit=1)[0].upper()
    )

    if first_keyword not in {"SELECT", "WITH"}:
        raise ValueError(
            "Solo se permiten consultas SQL de lectura "
            "utilizando SELECT o WITH."
        )

    return normalized_query


def _sql_authorizer(
    action: int,
    arg1: str | None,
    arg2: str | None,
    database: str | None,
    source: str | None,
) -> int:
    """
    Autoriza únicamente operaciones SQL de lectura.

    Args:
        action: Código de operación proporcionado por SQLite.
        arg1: Primer argumento de la operación.
        arg2: Segundo argumento de la operación.
        database: Base de datos afectada.
        source: Origen de la operación.

    Returns:
        SQLITE_OK para operaciones permitidas o SQLITE_DENY
        para operaciones no permitidas.
    """
    del arg1
    del arg2
    del database
    del source

    allowed_actions = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_TRANSACTION,
    }

    if action in allowed_actions:
        return sqlite3.SQLITE_OK

    return sqlite3.SQLITE_DENY


def execute_sql_query(
    dataframe: pd.DataFrame,
    query: str,
) -> dict[str, Any]:
    """
    Ejecuta una consulta SQL de solo lectura sobre el dataset.

    Args:
        dataframe: DataFrame que será consultado.
        query: Consulta SQL.

    Returns:
        Resultado de la consulta con columnas, filas y cantidad
        de registros.

    Raises:
        ValueError:
            Si la consulta no es válida.
        RuntimeError:
            Si SQLite o Pandas no pueden ejecutar la consulta.
    """
    validated_query = validate_sql_query(query)

    connection = dataframe_to_sqlite(dataframe)

    try:
        connection.set_authorizer(_sql_authorizer)

        result = pd.read_sql_query(
            validated_query,
            connection,
        )

    except (
        sqlite3.DatabaseError,
        pd.errors.DatabaseError,
    ) as error:
        logger.exception(
            "Error ejecutando consulta SQL."
        )

        raise RuntimeError(
            f"No fue posible ejecutar la consulta SQL: {error}"
        ) from error

    finally:
        connection.close()

    records = result.to_dict(
        orient="records"
    )

    logger.info(
        "Consulta SQL ejecutada correctamente. "
        "Filas obtenidas: %d.",
        len(records),
    )

    return {
        "query": validated_query,
        "row_count": len(records),
        "columns": list(result.columns),
        "rows": records,
    }


def _get_dataframe(
    runtime: ToolRuntime,
) -> pd.DataFrame:
    """
    Obtiene el DataFrame almacenado en el estado.

    Args:
        runtime: Contexto de ejecución de la herramienta.

    Returns:
        DataFrame almacenado en AnalysisState.

    Raises:
        ValueError:
            Si no existe un DataFrame válido.
    """
    dataframe = runtime.state.get("dataset")

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError(
            "No existe un DataFrame válido en el estado."
        )

    return dataframe


def _create_sql_tool_command(
    result: dict[str, Any],
    state_update: dict[str, Any],
    tool_call_id: str | None,
) -> Command:
    """
    Crea el Command utilizado por las herramientas SQL.

    Args:
        result: Resultado que será enviado al modelo.
        state_update: Campos que deben persistirse en el estado.
        tool_call_id: Identificador de la llamada de herramienta.

    Returns:
        Command con actualización del estado y ToolMessage.

    Raises:
        ValueError:
            Si no existe un tool_call_id válido.
    """
    if not tool_call_id:
        raise ValueError(
            "La herramienta requiere un tool_call_id válido."
        )

    tool_message = ToolMessage(
        content=str(result),
        tool_call_id=tool_call_id,
    )

    update = {
        **state_update,
        "messages": [tool_message],
    }

    return Command(
        update=update,
    )


@tool
def inspect_sql_schema(
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """
    Obtiene el esquema SQL del dataset actual.

    Args:
        runtime: Contexto de ejecución del agente.

    Returns:
        Esquema SQL de la tabla dataset.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Ejecutando herramienta inspect_sql_schema."
    )

    return get_sql_schema(dataframe)


@tool
def run_sql_query(
    query: str,
    runtime: ToolRuntime,
) -> Command:
    """
    Ejecuta una consulta SQL y persiste sus resultados.

    Args:
        query: Consulta SQL generada por el agente.
        runtime: Contexto de ejecución del agente.

    Returns:
        Command que acumula la consulta y su resultado.
    """
    dataframe = _get_dataframe(runtime)

    logger.info(
        "Ejecutando consulta SQL solicitada por el agente."
    )

    result = execute_sql_query(
        dataframe=dataframe,
        query=query,
    )

    return _create_sql_tool_command(
        result=result,
        state_update={
            "sql_query": [result["query"]],
            "sql_results": [result],
        },
        tool_call_id=runtime.tool_call_id,
    )