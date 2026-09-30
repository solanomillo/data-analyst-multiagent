"""Pruebas para la generación de visualizaciones."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from app.services.visualization import VisualizationError, create_chart


def test_create_bar_chart_from_sql_result() -> None:
    """Verifica la creación de un gráfico de barras simple desde SQL."""
    result = {
        "type": "bar",
        "source": "sql_result",
        "result_index": 0,
        "x_column": "categoria",
        "metric_column": "ventas_promedio",
        "data": [
            {"categoria": "A", "ventas_promedio": 1200.0},
            {"categoria": "B", "ventas_promedio": 900.0},
        ],
    }

    figure = create_chart(result)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1
    assert figure.data[0].type == "bar"
    assert list(figure.data[0].x) == ["A", "B"]
    assert list(figure.data[0].y) == [1200.0, 900.0]


def test_create_bar_chart_rejects_invalid_data() -> None:
    """Verifica el rechazo de datos inválidos para barras simples."""
    result = {
        "type": "bar",
        "x_column": "categoria",
        "metric_column": "ventas",
        "data": None,
    }

    with pytest.raises(VisualizationError):
        create_chart(result)


def test_create_grouped_bar_chart_remains_supported() -> None:
    """Verifica que el gráfico multidimensional existente siga funcionando."""
    result = {
        "type": "grouped_bar",
        "source": "sql_result",
        "result_index": 0,
        "x_column": "categoria",
        "series_column": "region",
        "metric_column": "ventas",
        "data": [
            {"categoria": "A", "region": "Norte", "ventas": 1200.0},
            {"categoria": "A", "region": "Sur", "ventas": 900.0},
            {"categoria": "B", "region": "Norte", "ventas": 1500.0},
        ],
    }

    figure = create_chart(result)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 2
    assert all(trace.type == "bar" for trace in figure.data)
