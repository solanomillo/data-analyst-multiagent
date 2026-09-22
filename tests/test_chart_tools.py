"""Pruebas para las herramientas de visualización."""

import pandas as pd
import pytest

from app.tools.chart_tools import (
    get_categorical_counts,
    get_chart_metadata,
    get_correlation_matrix,
    get_numeric_histograms,
    get_scatter_data,
)


def create_test_dataframe() -> pd.DataFrame:
    """Crea un dataset para las pruebas de visualización."""
    return pd.DataFrame(
        {
            "edad": [20, 25, 30, 35, 40],
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


def test_get_numeric_histograms() -> None:
    """Verifica la preparación de histogramas numéricos."""
    dataframe = create_test_dataframe()

    result = get_numeric_histograms(dataframe)

    assert "edad" in result
    assert "ingresos" in result
    assert len(result) == 2


def test_get_categorical_counts() -> None:
    """Verifica las frecuencias categóricas."""
    dataframe = create_test_dataframe()

    result = get_categorical_counts(dataframe)

    assert "ciudad" in result
    assert result["ciudad"]["Salta"] == 3


def test_get_correlation_matrix() -> None:
    """Verifica la matriz de correlaciones."""
    dataframe = create_test_dataframe()

    result = get_correlation_matrix(dataframe)

    assert result.shape == (2, 2)
    assert "edad" in result.columns
    assert "ingresos" in result.columns


def test_get_scatter_data() -> None:
    """Verifica la preparación de datos de dispersión."""
    dataframe = create_test_dataframe()

    result = get_scatter_data(
        dataframe,
        "edad",
        "ingresos",
    )

    assert list(result.columns) == [
        "edad",
        "ingresos",
    ]
    assert len(result) == 5


def test_get_scatter_data_invalid_column() -> None:
    """Verifica el error cuando una columna no existe."""
    dataframe = create_test_dataframe()

    with pytest.raises(ValueError):
        get_scatter_data(
            dataframe,
            "edad",
            "altura",
        )


def test_get_chart_metadata() -> None:
    """Verifica los metadatos de visualización."""
    dataframe = create_test_dataframe()

    result = get_chart_metadata(dataframe)

    assert result["numeric_count"] == 2
    assert result["categorical_count"] == 1