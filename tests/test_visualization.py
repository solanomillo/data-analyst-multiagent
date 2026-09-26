"""Pruebas para el servicio de visualización."""

from __future__ import annotations

import pytest
from plotly.graph_objects import Figure

from app.services.visualization import (
    VisualizationError,
    create_chart,
    create_charts,
)


def test_create_categorical_chart() -> None:
    """Verifica la creación de un gráfico categórico."""
    result = {
        "type": "categorical_distribution",
        "column": "Categoría",
        "values": {
            "Electrónica": 26,
            "Hogar": 14,
            "Mobiliario": 10,
        },
    }

    figure = create_chart(result)

    assert isinstance(figure, Figure)
    assert len(figure.data) == 1
    assert figure.data[0].type == "bar"


def test_create_numeric_distribution_chart() -> None:
    """Verifica la creación de un histograma numérico."""
    result = {
        "type": "numeric_distribution",
        "column": "Cantidad",
        "count": 5,
        "values": [1.0, 2.0, 3.0, 4.0, 5.0],
    }

    figure = create_chart(result)

    assert isinstance(figure, Figure)
    assert len(figure.data) == 1
    assert figure.data[0].type == "histogram"


def test_create_correlation_chart() -> None:
    """Verifica la creación del mapa de calor de correlaciones."""
    result = {
        "type": "correlation_matrix",
        "columns": [
            "Cantidad",
            "Precio_Unitario",
        ],
        "values": {
            "Cantidad": {
                "Cantidad": 1.0,
                "Precio_Unitario": -0.51,
            },
            "Precio_Unitario": {
                "Cantidad": -0.51,
                "Precio_Unitario": 1.0,
            },
        },
    }

    figure = create_chart(result)

    assert isinstance(figure, Figure)
    assert len(figure.data) == 1
    assert figure.data[0].type == "heatmap"


def test_create_scatter_chart() -> None:
    """Verifica la creación de un gráfico de dispersión."""
    result = {
        "type": "scatter",
        "x_column": "Cantidad",
        "y_column": "Precio_Unitario",
        "count": 4,
        "x_values": [1, 2, 3, 4],
        "y_values": [100, 200, 150, 300],
    }

    figure = create_chart(result)

    assert isinstance(figure, Figure)
    assert len(figure.data) == 1
    assert figure.data[0].type == "scatter"


def test_create_chart_rejects_unknown_type() -> None:
    """Verifica el rechazo de tipos de visualización desconocidos."""
    result = {
        "type": "unknown_chart",
    }

    with pytest.raises(VisualizationError):
        create_chart(result)


def test_create_chart_rejects_invalid_categorical_result() -> None:
    """Verifica la validación de resultados categóricos inválidos."""
    result = {
        "type": "categorical_distribution",
        "column": "Categoría",
        "values": None,
    }

    with pytest.raises(VisualizationError):
        create_chart(result)


def test_create_chart_rejects_invalid_scatter_result() -> None:
    """Verifica la validación de resultados de dispersión inválidos."""
    result = {
        "type": "scatter",
        "x_column": "Cantidad",
        "y_column": "Precio_Unitario",
        "x_values": [1, 2, 3],
        "y_values": [100, 200],
    }

    with pytest.raises(VisualizationError):
        create_chart(result)


def test_create_charts_builds_multiple_visualizations() -> None:
    """Verifica la construcción de múltiples visualizaciones."""
    results = [
        {
            "type": "categorical_distribution",
            "column": "Categoría",
            "values": {
                "Electrónica": 26,
                "Hogar": 14,
            },
        },
        {
            "type": "numeric_distribution",
            "column": "Cantidad",
            "values": [1.0, 2.0, 3.0, 4.0],
        },
        {
            "type": "scatter",
            "x_column": "Cantidad",
            "y_column": "Precio_Unitario",
            "x_values": [1, 2, 3],
            "y_values": [100, 200, 300],
        },
    ]

    figures = create_charts(results)

    assert len(figures) == 3
    assert all(isinstance(figure, Figure) for figure in figures)


def test_create_charts_skips_invalid_results() -> None:
    """Verifica que un resultado inválido no impida otros gráficos."""
    results = [
        {
            "type": "unknown_chart",
        },
        {
            "type": "categorical_distribution",
            "column": "Región",
            "values": {
                "Norte": 12,
                "Sur": 11,
            },
        },
    ]

    figures = create_charts(results)

    assert len(figures) == 1
    assert isinstance(figures[0], Figure)
