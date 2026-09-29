"""Pruebas del servicio de visualización."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from app.services.visualization import (
    VisualizationError,
    create_chart,
)


def test_create_grouped_bar_chart() -> None:
    """Verifica la creación de un gráfico agrupado desde SQL."""
    result = {
        "type": "grouped_bar",
        "source": "sql_result",
        "result_index": 0,
        "x_column": "categoria",
        "series_column": "region",
        "metric_column": "ventas",
        "data": [
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
        ],
    }

    figure = create_chart(result)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 2
    assert all(trace.type == "bar" for trace in figure.data)


def test_create_grouped_bar_chart_rejects_invalid_data() -> None:
    """Verifica el rechazo de datos agrupados inválidos."""
    result = {
        "type": "grouped_bar",
        "x_column": "categoria",
        "series_column": "region",
        "metric_column": "ventas",
        "data": None,
    }

    with pytest.raises(VisualizationError):
        create_chart(result)
