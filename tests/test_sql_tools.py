"""Pruebas para las herramientas SQL."""

import pandas as pd
import pytest

from app.tools.sql_tools import (
    execute_sql_query,
    get_sql_schema,
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