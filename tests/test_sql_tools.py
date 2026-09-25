"""Pruebas para las herramientas SQL."""

from __future__ import annotations

import pandas as pd
import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.tools.sql_tools import (
    execute_sql_query,
    get_sql_schema,
    inspect_sql_schema,
    run_sql_query,
    validate_sql_query,
)


def create_test_dataframe() -> pd.DataFrame:
    """Crea un dataset para las pruebas SQL."""
    return pd.DataFrame(
        {
            "producto": [
                "A",
                "B",
                "A",
                "C",
            ],
            "ventas": [
                100,
                200,
                150,
                50,
            ],
        }
    )


def create_test_runtime(
    dataframe: pd.DataFrame,
    tool_call_id: str = "test_tool_call_id",
) -> ToolRuntime:
    """
    Crea un ToolRuntime para las pruebas SQL.

    Args:
        dataframe: Dataset almacenado en el estado.
        tool_call_id: Identificador simulado de la herramienta.

    Returns:
        ToolRuntime configurado para testing.
    """
    return ToolRuntime(
        state={
            "dataset": dataframe,
        },
        context={},
        config={},
        stream_writer=lambda _: None,
        tool_call_id=tool_call_id,
        store=None,
    )


def test_get_sql_schema() -> None:
    """Verifica la obtención del esquema SQL."""
    dataframe = create_test_dataframe()

    result = get_sql_schema(dataframe)

    assert result["table_name"] == "dataset"
    assert len(result["columns"]) == 2
    assert result["columns"][0]["name"] == "producto"
    assert result["columns"][1]["name"] == "ventas"


def test_validate_sql_query() -> None:
    """Verifica la validación de consultas SELECT."""
    query = "SELECT producto FROM dataset;"

    result = validate_sql_query(query)

    assert result == "SELECT producto FROM dataset"


def test_validate_sql_query_accepts_with() -> None:
    """Verifica que las consultas WITH sean aceptadas."""
    query = """
        WITH resumen AS (
            SELECT producto, SUM(ventas) AS total
            FROM dataset
            GROUP BY producto
        )
        SELECT *
        FROM resumen
    """

    result = validate_sql_query(query)

    assert result.startswith("WITH resumen")


def test_validate_sql_query_rejects_modification() -> None:
    """Verifica que las consultas de modificación sean rechazadas."""
    with pytest.raises(ValueError):
        validate_sql_query(
            "DROP TABLE dataset"
        )


def test_validate_sql_query_rejects_empty_query() -> None:
    """Verifica que una consulta vacía sea rechazada."""
    with pytest.raises(ValueError):
        validate_sql_query("")


def test_execute_sql_query() -> None:
    """Verifica la ejecución de una consulta SQL."""
    dataframe = create_test_dataframe()

    result = execute_sql_query(
        dataframe,
        """
        SELECT
            producto,
            SUM(ventas) AS total_ventas
        FROM dataset
        GROUP BY producto
        ORDER BY total_ventas DESC
        """,
    )

    assert result["row_count"] == 3
    assert result["columns"] == [
        "producto",
        "total_ventas",
    ]

    assert result["rows"][0]["producto"] == "A"
    assert result["rows"][0]["total_ventas"] == 250


def test_execute_sql_query_rejects_update() -> None:
    """Verifica que UPDATE no pueda ejecutarse."""
    dataframe = create_test_dataframe()

    with pytest.raises(ValueError):
        execute_sql_query(
            dataframe,
            "UPDATE dataset SET ventas = 0",
        )


def test_execute_sql_query_rejects_delete() -> None:
    """Verifica que DELETE no pueda ejecutarse."""
    dataframe = create_test_dataframe()

    with pytest.raises(ValueError):
        execute_sql_query(
            dataframe,
            "DELETE FROM dataset",
        )


def test_execute_sql_query_authorizer_blocks_write() -> None:
    """Verifica que SQLite bloquee una operación de escritura."""
    dataframe = create_test_dataframe()

    with pytest.raises(
        RuntimeError,
        match="not authorized",
    ):
        execute_sql_query(
            dataframe,
            "WITH datos AS (SELECT * FROM dataset) "
            "DELETE FROM dataset",
        )


def test_inspect_sql_schema_tool() -> None:
    """Verifica la herramienta de inspección del esquema SQL."""
    dataframe = create_test_dataframe()
    runtime = create_test_runtime(dataframe)

    result = inspect_sql_schema.func(runtime)

    assert result["table_name"] == "dataset"
    assert len(result["columns"]) == 2


def test_run_sql_query_updates_state() -> None:
    """
    Verifica que run_sql_query actualice el estado acumulativo.
    """
    dataframe = create_test_dataframe()
    runtime = create_test_runtime(dataframe)

    result = run_sql_query.func(
        """
        SELECT
            producto,
            SUM(ventas) AS total_ventas
        FROM dataset
        GROUP BY producto
        ORDER BY total_ventas DESC
        """,
        runtime,
    )

    assert isinstance(result, Command)

    expected_query = (
        "SELECT\n"
        "            producto,\n"
        "            SUM(ventas) AS total_ventas\n"
        "        FROM dataset\n"
        "        GROUP BY producto\n"
        "        ORDER BY total_ventas DESC"
    )

    assert result.update["sql_query"] == [
        expected_query
    ]

    sql_results = result.update["sql_results"]

    assert len(sql_results) == 1

    sql_result = sql_results[0]

    assert sql_result["row_count"] == 3
    assert sql_result["columns"] == [
        "producto",
        "total_ventas",
    ]

    assert sql_result["rows"][0]["producto"] == "A"
    assert sql_result["rows"][0]["total_ventas"] == 250

    messages = result.update["messages"]

    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].tool_call_id == "test_tool_call_id"