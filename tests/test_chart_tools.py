"""Pruebas para las herramientas de visualización."""

from __future__ import annotations

import operator

import pandas as pd
import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.graph.state import AnalysisState
from app.tools.chart_tools import (
    get_categorical_distribution,
    get_correlation_matrix,
    get_grouped_sql_result,
    get_numeric_distribution,
    get_scatter_data,
    inspect_categorical_distribution,
    inspect_correlation_matrix,
    inspect_grouped_sql_result,
    inspect_numeric_distribution,
    inspect_scatter_data,
)


def create_test_dataframe() -> pd.DataFrame:
    """Crea un dataset para las pruebas de visualización."""
    return pd.DataFrame(
        {
            "edad": [
                20,
                25,
                30,
                35,
                40,
            ],
            "ingresos": [
                1000,
                1500,
                2000,
                2500,
                3000,
            ],
            "ciudad": [
                "Salta",
                "Tartagal",
                "Salta",
                "Orán",
                "Salta",
            ],
        }
    )


def create_test_runtime(
    dataframe: pd.DataFrame,
    chart_results: list[dict] | None = None,
    tool_call_id: str = "test_tool_call_id",
) -> ToolRuntime:
    """
    Crea un ToolRuntime para las pruebas.

    Args:
        dataframe: Dataset almacenado en el estado.
        chart_results: Resultados de visualización existentes.
        tool_call_id: Identificador simulado de la herramienta.

    Returns:
        ToolRuntime configurado para testing.
    """
    return ToolRuntime(
        state={
            "dataset": dataframe,
            "chart_results": (
                chart_results
                if chart_results is not None
                else []
            ),
        },
        context={},
        config={},
        stream_writer=lambda _: None,
        tool_call_id=tool_call_id,
        store=None,
    )


def test_get_numeric_distribution() -> None:
    """Verifica la preparación de una distribución numérica."""
    dataframe = create_test_dataframe()

    result = get_numeric_distribution(
        dataframe,
        "edad",
    )

    assert result["column"] == "edad"
    assert result["type"] == "numeric_distribution"
    assert result["count"] == 5
    assert result["values"] == [
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
    ]


def test_get_numeric_distribution_rejects_invalid_column() -> None:
    """Verifica que una columna inexistente sea rechazada."""
    dataframe = create_test_dataframe()

    with pytest.raises(ValueError):
        get_numeric_distribution(
            dataframe,
            "altura",
        )


def test_get_numeric_distribution_rejects_non_numeric_column() -> None:
    """Verifica que una columna no numérica sea rechazada."""
    dataframe = create_test_dataframe()

    with pytest.raises(ValueError):
        get_numeric_distribution(
            dataframe,
            "ciudad",
        )


def test_get_categorical_distribution() -> None:
    """Verifica la preparación de una distribución categórica."""
    dataframe = create_test_dataframe()

    result = get_categorical_distribution(
        dataframe,
        "ciudad",
    )

    assert result["column"] == "ciudad"
    assert result["type"] == "categorical_distribution"
    assert result["values"]["Salta"] == 3
    assert result["values"]["Tartagal"] == 1
    assert result["values"]["Orán"] == 1


def test_get_correlation_matrix() -> None:
    """Verifica el cálculo de la matriz de correlación."""
    dataframe = create_test_dataframe()

    result = get_correlation_matrix(dataframe)

    assert result["type"] == "correlation_matrix"
    assert result["columns"] == [
        "edad",
        "ingresos",
    ]
    assert result["values"]["edad"]["edad"] == 1.0
    assert result["values"]["edad"]["ingresos"] == 1.0


def test_get_correlation_matrix_requires_two_numeric_columns() -> None:
    """Verifica el requisito de dos columnas numéricas."""
    dataframe = pd.DataFrame(
        {
            "edad": [20, 25, 30],
            "ciudad": [
                "Salta",
                "Tartagal",
                "Orán",
            ],
        }
    )

    with pytest.raises(ValueError):
        get_correlation_matrix(dataframe)


def test_get_scatter_data() -> None:
    """Verifica la preparación de datos para dispersión."""
    dataframe = create_test_dataframe()

    result = get_scatter_data(
        dataframe,
        "edad",
        "ingresos",
    )

    assert result["type"] == "scatter"
    assert result["x_column"] == "edad"
    assert result["y_column"] == "ingresos"
    assert result["count"] == 5
    assert result["x_values"] == [
        20,
        25,
        30,
        35,
        40,
    ]
    assert result["y_values"] == [
        1000,
        1500,
        2000,
        2500,
        3000,
    ]


def test_inspect_numeric_distribution_updates_state() -> None:
    """Verifica la persistencia de una distribución numérica."""
    dataframe = create_test_dataframe()
    runtime = create_test_runtime(dataframe)

    result = inspect_numeric_distribution.func(
        "edad",
        runtime,
    )

    assert isinstance(result, Command)

    chart_results = result.update["chart_results"]

    assert len(chart_results) == 1
    assert chart_results[0]["column"] == "edad"
    assert chart_results[0]["type"] == (
        "numeric_distribution"
    )

    messages = result.update["messages"]

    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].tool_call_id == (
        "test_tool_call_id"
    )


def test_inspect_categorical_distribution_updates_state() -> None:
    """Verifica la persistencia de una distribución categórica."""
    dataframe = create_test_dataframe()
    runtime = create_test_runtime(dataframe)

    result = inspect_categorical_distribution.func(
        "ciudad",
        runtime,
    )

    assert isinstance(result, Command)

    chart_results = result.update["chart_results"]

    assert len(chart_results) == 1
    assert chart_results[0]["column"] == "ciudad"
    assert chart_results[0]["type"] == (
        "categorical_distribution"
    )


def test_inspect_correlation_matrix_updates_state() -> None:
    """Verifica la persistencia de la matriz de correlación."""
    dataframe = create_test_dataframe()
    runtime = create_test_runtime(dataframe)

    result = inspect_correlation_matrix.func(
        runtime,
    )

    assert isinstance(result, Command)

    chart_results = result.update["chart_results"]

    assert len(chart_results) == 1
    assert chart_results[0]["type"] == (
        "correlation_matrix"
    )


def test_inspect_scatter_data_updates_state() -> None:
    """Verifica la persistencia de los datos de dispersión."""
    dataframe = create_test_dataframe()
    runtime = create_test_runtime(dataframe)

    result = inspect_scatter_data.func(
        "edad",
        "ingresos",
        runtime,
    )

    assert isinstance(result, Command)

    chart_results = result.update["chart_results"]

    assert len(chart_results) == 1
    assert chart_results[0]["type"] == "scatter"
    assert chart_results[0]["x_column"] == "edad"
    assert chart_results[0]["y_column"] == "ingresos"


def test_chart_tool_returns_only_new_result() -> None:
    """
    Verifica que la herramienta no acumule manualmente resultados.

    La acumulación de resultados corresponde al reducer definido
    en AnalysisState.
    """
    dataframe = create_test_dataframe()

    previous_results = [
        {
            "column": "edad",
            "type": "numeric_distribution",
            "count": 5,
        }
    ]

    runtime = create_test_runtime(
        dataframe=dataframe,
        chart_results=previous_results,
    )

    result = inspect_categorical_distribution.func(
        "ciudad",
        runtime,
    )

    assert isinstance(result, Command)

    chart_results = result.update["chart_results"]

    assert len(chart_results) == 1
    assert chart_results[0]["column"] == "ciudad"
    assert chart_results[0]["type"] == (
        "categorical_distribution"
    )


def test_chart_results_reducer_accumulates_results() -> None:
    """
    Verifica que el reducer de AnalysisState acumule resultados.

    La herramienta produce un único resultado por ejecución.
    LangGraph utiliza el reducer operator.add para combinar
    múltiples actualizaciones del campo chart_results.
    """
    previous_results = [
        {
            "column": "edad",
            "type": "numeric_distribution",
            "count": 5,
        }
    ]

    new_result = {
        "column": "ciudad",
        "type": "categorical_distribution",
        "values": {
            "Salta": 3,
            "Tartagal": 1,
            "Orán": 1,
        },
    }

    reducer = operator.add

    accumulated_results = reducer(
        previous_results,
        [new_result],
    )

    assert len(accumulated_results) == 2
    assert accumulated_results[0] == previous_results[0]
    assert accumulated_results[1] == new_result


def test_chart_tool_requires_tool_call_id() -> None:
    """Verifica que una herramienta requiera un tool_call_id."""
    dataframe = create_test_dataframe()

    runtime = create_test_runtime(
        dataframe=dataframe,
        tool_call_id="",
    )

    with pytest.raises(
        ValueError,
        match="tool_call_id",
    ):
        inspect_numeric_distribution.func(
            "edad",
            runtime,
        )


def test_get_grouped_sql_result() -> None:
    """Verifica la preparación de un resultado SQL multidimensional."""
    sql_results = [
        {
            "query": "SELECT ...",
            "rows": [
                {
                    "categoria": "A",
                    "region": "Norte",
                    "ventas": 1200,
                },
                {
                    "categoria": "A",
                    "region": "Sur",
                    "ventas": 900,
                },
                {
                    "categoria": "B",
                    "region": "Norte",
                    "ventas": 1500,
                },
            ],
        }
    ]

    result = get_grouped_sql_result(
        sql_results=sql_results,
        result_index=0,
        x_column="categoria",
        series_column="region",
        metric_column="ventas",
    )

    assert result["type"] == "grouped_bar"
    assert result["source"] == "sql_result"
    assert result["x_column"] == "categoria"
    assert result["series_column"] == "region"
    assert result["metric_column"] == "ventas"
    assert result["data"] == [
        {
            "categoria": "A",
            "region": "Norte",
            "ventas": 1200.0,
        },
        {
            "categoria": "A",
            "region": "Sur",
            "ventas": 900.0,
        },
        {
            "categoria": "B",
            "region": "Norte",
            "ventas": 1500.0,
        },
    ]


def test_get_grouped_sql_result_rejects_invalid_index() -> None:
    """Verifica el rechazo de un índice SQL inexistente."""
    with pytest.raises(ValueError, match="índice"):
        get_grouped_sql_result(
            sql_results=[],
            result_index=0,
            x_column="categoria",
            series_column="region",
            metric_column="ventas",
        )


def test_get_grouped_sql_result_rejects_missing_column() -> None:
    """Verifica el rechazo de una columna ausente en SQL."""
    sql_results = [
        {
            "rows": [
                {
                    "categoria": "A",
                    "region": "Norte",
                    "ventas": 1200,
                }
            ]
        }
    ]

    with pytest.raises(ValueError, match="no existe"):
        get_grouped_sql_result(
            sql_results=sql_results,
            result_index=0,
            x_column="categoria",
            series_column="pais",
            metric_column="ventas",
        )


def test_get_grouped_sql_result_rejects_non_numeric_metric() -> None:
    """Verifica el rechazo de una métrica no numérica."""
    sql_results = [
        {
            "rows": [
                {
                    "categoria": "A",
                    "region": "Norte",
                    "ventas": "1200",
                }
            ]
        }
    ]

    with pytest.raises(ValueError, match="numérica"):
        get_grouped_sql_result(
            sql_results=sql_results,
            result_index=0,
            x_column="categoria",
            series_column="region",
            metric_column="ventas",
        )


def test_inspect_grouped_sql_result_updates_state() -> None:
    """Verifica la persistencia del resultado SQL agrupado."""
    dataframe = create_test_dataframe()
    runtime = ToolRuntime(
        state={
            "dataset": dataframe,
            "sql_results": [
                {
                    "rows": [
                        {
                            "categoria": "A",
                            "region": "Norte",
                            "ventas": 1200,
                        }
                    ]
                }
            ],
            "chart_results": [],
        },
        context={},
        config={},
        stream_writer=lambda _: None,
        tool_call_id="test_tool_call_id",
        store=None,
    )

    result = inspect_grouped_sql_result.func(
        0,
        "categoria",
        "region",
        "ventas",
        runtime,
    )

    assert isinstance(result, Command)
    assert len(result.update["chart_results"]) == 1
    assert result.update["chart_results"][0]["type"] == "grouped_bar"

