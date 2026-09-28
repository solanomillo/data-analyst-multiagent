"""Pruebas para las herramientas de análisis de datos."""

from __future__ import annotations

import pandas as pd
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.tools.data_tools import (
    calculate_correlations,
    check_duplicates,
    check_missing_values,
    detect_outliers,
    get_categorical_summary,
    get_dataset_schema,
    get_numeric_summary,
    inspect_categorical_statistics,
    inspect_correlations,
    inspect_dataset_schema,
    inspect_duplicates,
    inspect_missing_values,
    inspect_numeric_statistics,
    inspect_outliers,
)


def create_test_dataframe() -> pd.DataFrame:
    """Crea un dataset pequeño para las pruebas."""
    return pd.DataFrame(
        {
            "edad": [20, 25, 30, 35, 100],
            "ingresos": [1000, 1500, 2000, 2500, 3000],
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
    tool_call_id: str = "test_tool_call_id",
) -> ToolRuntime:
    """
    Crea un ToolRuntime para las pruebas de herramientas.

    Args:
        dataframe: Dataset que será almacenado en el estado.
        tool_call_id: Identificador simulado de la llamada.

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


def test_get_dataset_schema() -> None:
    """Verifica que el esquema del dataset sea correcto."""
    dataframe = create_test_dataframe()

    result = get_dataset_schema(dataframe)

    assert result["rows"] == 5
    assert result["columns"] == 3
    assert len(result["column_details"]) == 3


def test_check_missing_values() -> None:
    """Verifica la detección de valores nulos."""
    dataframe = create_test_dataframe()

    dataframe.loc[0, "edad"] = None

    result = check_missing_values(dataframe)

    assert result["total_null_values"] == 1
    assert result["columns_with_nulls"]["edad"]["count"] == 1


def test_check_duplicates() -> None:
    """Verifica la detección de registros duplicados."""
    dataframe = create_test_dataframe()

    dataframe = pd.concat(
        [dataframe, dataframe.iloc[[0]]],
        ignore_index=True,
    )

    result = check_duplicates(dataframe)

    assert result["duplicate_rows"] == 1


def test_get_numeric_summary() -> None:
    """Verifica las estadísticas de las columnas numéricas."""
    dataframe = create_test_dataframe()

    result = get_numeric_summary(dataframe)

    assert "edad" in result
    assert "ingresos" in result
    assert result["edad"]["count"] == 5.0


def test_get_numeric_summary_without_numeric_columns() -> None:
    """Verifica que no haya errores sin columnas numéricas."""
    dataframe = pd.DataFrame(
        {
            "nombre": ["Juan", "Maria", "Pedro"],
            "ciudad": ["Salta", "Orán", "Tartagal"],
        }
    )

    result = get_numeric_summary(dataframe)

    assert result == {}


def test_get_numeric_summary_with_all_null_column() -> None:
    """Verifica estadísticas cuando una columna numérica es completamente nula."""
    dataframe = pd.DataFrame(
        {
            "edad": pd.Series(
                [None, None, None],
                dtype="float64",
            ),
            "ingresos": [1000, 1500, 2000],
        }
    )

    result = get_numeric_summary(dataframe)

    assert "edad" in result
    assert "ingresos" in result
    assert result["edad"]["count"] == 0.0
    assert result["ingresos"]["count"] == 3.0


def test_get_categorical_summary() -> None:
    """Verifica el análisis de variables categóricas."""
    dataframe = create_test_dataframe()

    result = get_categorical_summary(dataframe)

    assert "ciudad" in result
    assert result["ciudad"]["unique_values"] == 3


def test_get_categorical_summary_without_categorical_columns() -> None:
    """Verifica que no haya resultados sin columnas categóricas."""
    dataframe = pd.DataFrame(
        {
            "edad": [20, 30, 40],
            "ingresos": [1000, 2000, 3000],
        }
    )

    result = get_categorical_summary(dataframe)

    assert result == {}


def test_detect_outliers() -> None:
    """Verifica la detección de posibles valores atípicos."""
    dataframe = create_test_dataframe()

    result = detect_outliers(dataframe)

    assert "edad" in result
    assert result["edad"]["outlier_count"] >= 1


def test_detect_outliers_without_numeric_columns() -> None:
    """Verifica que no haya errores sin columnas numéricas."""
    dataframe = pd.DataFrame(
        {
            "nombre": ["Juan", "Maria", "Pedro"],
            "ciudad": ["Salta", "Orán", "Tartagal"],
        }
    )

    result = detect_outliers(dataframe)

    assert result == {}


def test_calculate_correlations() -> None:
    """Verifica el cálculo de correlaciones."""
    dataframe = create_test_dataframe()

    result = calculate_correlations(dataframe)

    assert "edad" in result
    assert "ingresos" in result
    assert result["edad"]["ingresos"] > 0


def test_calculate_correlations_without_numeric_columns() -> None:
    """Verifica que no haya correlaciones sin variables numéricas."""
    dataframe = pd.DataFrame(
        {
            "nombre": ["Juan", "Maria", "Pedro"],
            "ciudad": ["Salta", "Orán", "Tartagal"],
        }
    )

    result = calculate_correlations(dataframe)

    assert result == {}


def test_calculate_correlations_with_single_numeric_column() -> None:
    """Verifica que no se calculen correlaciones con una sola variable."""
    dataframe = pd.DataFrame(
        {
            "edad": [20, 30, 40],
            "ciudad": ["Salta", "Orán", "Tartagal"],
        }
    )

    result = calculate_correlations(dataframe)

    assert result == {}


def test_calculate_correlations_with_constant_column() -> None:
    """Verifica el comportamiento con una variable numérica constante."""
    dataframe = pd.DataFrame(
        {
            "edad": [20, 20, 20, 20],
            "ingresos": [1000, 1500, 2000, 2500],
        }
    )

    result = calculate_correlations(dataframe)

    assert "edad" in result
    assert "ingresos" in result
    assert pd.isna(result["edad"]["ingresos"])


def test_inspect_dataset_schema_updates_state() -> None:
    """Verifica que el esquema actualice AnalysisState."""
    dataframe = create_test_dataframe()
    runtime = create_test_runtime(dataframe)

    result = inspect_dataset_schema.func(runtime)

    assert isinstance(result, Command)
    assert result.update["dataset_schema"]["rows"] == 5
    assert result.update["dataset_schema"]["columns"] == 3

    messages = result.update["messages"]

    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].tool_call_id == "test_tool_call_id"


def test_inspect_missing_values_updates_state() -> None:
    """Verifica que los valores nulos actualicen el estado."""
    dataframe = create_test_dataframe()

    dataframe.loc[0, "edad"] = None

    runtime = create_test_runtime(dataframe)

    result = inspect_missing_values.func(runtime)

    assert isinstance(result, Command)
    assert result.update["missing_values"][
        "total_null_values"
    ] == 1


def test_inspect_duplicates_updates_state() -> None:
    """Verifica que los duplicados actualicen el estado."""
    dataframe = create_test_dataframe()

    dataframe = pd.concat(
        [dataframe, dataframe.iloc[[0]]],
        ignore_index=True,
    )

    runtime = create_test_runtime(dataframe)

    result = inspect_duplicates.func(runtime)

    assert isinstance(result, Command)
    assert result.update["duplicate_info"][
        "duplicate_rows"
    ] == 1


def test_inspect_numeric_statistics_updates_state() -> None:
    """Verifica que las estadísticas numéricas actualicen el estado."""
    dataframe = create_test_dataframe()
    runtime = create_test_runtime(dataframe)

    result = inspect_numeric_statistics.func(runtime)

    assert isinstance(result, Command)
    assert "edad" in result.update["numeric_summary"]
    assert "ingresos" in result.update["numeric_summary"]


def test_inspect_categorical_statistics_updates_state() -> None:
    """Verifica que las estadísticas categóricas actualicen el estado."""
    dataframe = create_test_dataframe()
    runtime = create_test_runtime(dataframe)

    result = inspect_categorical_statistics.func(runtime)

    assert isinstance(result, Command)
    assert "ciudad" in result.update["categorical_summary"]


def test_inspect_outliers_updates_state() -> None:
    """Verifica que los outliers actualicen AnalysisState."""
    dataframe = create_test_dataframe()
    runtime = create_test_runtime(dataframe)

    result = inspect_outliers.func(runtime)

    assert isinstance(result, Command)
    assert "edad" in result.update["outliers"]


def test_inspect_correlations_updates_state() -> None:
    """Verifica que las correlaciones actualicen AnalysisState."""
    dataframe = create_test_dataframe()
    runtime = create_test_runtime(dataframe)

    result = inspect_correlations.func(runtime)

    assert isinstance(result, Command)
    assert "edad" in result.update["correlations"]
    assert "ingresos" in result.update["correlations"]