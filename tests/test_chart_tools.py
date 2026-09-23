"""Pruebas para las herramientas de visualización."""

import pandas as pd
import pytest

from app.tools.chart_tools import (
    get_categorical_distribution,
    get_chart_metadata,
    get_correlation_matrix,
    get_numeric_distribution,
    get_scatter_data,
)


def create_test_dataframe() -> pd.DataFrame:
    """Crea un dataset para las pruebas."""
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
            "cantidad": [
                2,
                4,
                3,
                1,
            ],
        }
    )


def test_get_numeric_distribution() -> None:
    """Verifica la preparación de una distribución numérica."""
    dataframe = create_test_dataframe()

    result = get_numeric_distribution(
        dataframe,
        "ventas",
    )

    assert result["column"] == "ventas"
    assert result["type"] == "numeric_distribution"
    assert result["count"] == 4
    assert result["values"] == [
        100.0,
        200.0,
        150.0,
        50.0,
    ]


def test_get_numeric_distribution_rejects_invalid_column() -> None:
    """Verifica el rechazo de una columna inexistente."""
    dataframe = create_test_dataframe()

    with pytest.raises(ValueError):
        get_numeric_distribution(
            dataframe,
            "inexistente",
        )


def test_get_numeric_distribution_rejects_non_numeric() -> None:
    """Verifica el rechazo de una columna no numérica."""
    dataframe = create_test_dataframe()

    with pytest.raises(ValueError):
        get_numeric_distribution(
            dataframe,
            "producto",
        )


def test_get_categorical_distribution() -> None:
    """Verifica la distribución categórica."""
    dataframe = create_test_dataframe()

    result = get_categorical_distribution(
        dataframe,
        "producto",
    )

    assert result["column"] == "producto"
    assert result["type"] == "categorical_distribution"
    assert result["values"]["A"] == 2
    assert result["values"]["B"] == 1


def test_get_correlation_matrix() -> None:
    """Verifica la matriz de correlación."""
    dataframe = create_test_dataframe()

    result = get_correlation_matrix(dataframe)

    assert result["type"] == "correlation_matrix"
    assert "ventas" in result["columns"]
    assert "cantidad" in result["columns"]


def test_get_scatter_data() -> None:
    """Verifica la preparación de datos para dispersión."""
    dataframe = create_test_dataframe()

    result = get_scatter_data(
        dataframe,
        "ventas",
        "cantidad",
    )

    assert result["type"] == "scatter"
    assert result["x_column"] == "ventas"
    assert result["y_column"] == "cantidad"
    assert result["count"] == 4


def test_get_chart_metadata() -> None:
    """Verifica los metadatos para selección de gráficos."""
    dataframe = create_test_dataframe()

    result = get_chart_metadata(dataframe)

    assert "ventas" in result["numeric_columns"]
    assert "cantidad" in result["numeric_columns"]
    assert "producto" in result["categorical_columns"]
    assert result["total_columns"] == 3