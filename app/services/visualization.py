"""Servicio para construir visualizaciones a partir de resultados analíticos."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


logger = logging.getLogger(__name__)


class VisualizationError(Exception):
    """Error controlado relacionado con la generación de visualizaciones."""


def _apply_common_layout(
    figure: go.Figure,
    title: str,
    description: str,
) -> go.Figure:
    """
    Aplica una configuración visual común a todos los gráficos.

    Args:
        figure: Figura de Plotly que será configurada.
        title: Título principal del gráfico.
        description: Descripción contextual del gráfico.

    Returns:
        Figura configurada.
    """
    figure.update_layout(
        title={
            "text": (
                f"<b>{title}</b>"
                f"<br><sup>{description}</sup>"
            ),
            "x": 0,
            "xanchor": "left",
        },
        template="plotly_white",
        hovermode="closest",
        margin={
            "l": 60,
            "r": 30,
            "t": 90,
            "b": 60,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )

    return figure


def _create_categorical_chart(
    result: dict[str, Any],
) -> go.Figure:
    """
    Construye un gráfico de barras para una distribución categórica.

    Args:
        result: Resultado generado por la herramienta categórica.

    Returns:
        Figura de Plotly.
    """
    column = result.get("column")
    values = result.get("values")

    if not column or not isinstance(values, dict):
        raise VisualizationError(
            "El resultado categórico no contiene la estructura esperada."
        )

    dataframe = pd.DataFrame(
        {
            "category": list(values.keys()),
            "count": list(values.values()),
        }
    )

    dataframe = dataframe.sort_values(
        by="count",
        ascending=True,
    )

    figure = px.bar(
        dataframe,
        x="count",
        y="category",
        orientation="h",
        labels={
            "category": str(column),
            "count": "Cantidad de registros",
        },
        text="count",
    )

    figure.update_traces(
        textposition="outside",
        hovertemplate=(
            f"{column}: %{{y}}"
            "<br>Registros: %{x}"
            "<extra></extra>"
        ),
    )

    figure.update_xaxes(
        rangemode="tozero",
        showgrid=True,
        title_text="Cantidad de registros",
    )

    figure.update_yaxes(
        title_text=str(column),
    )

    return _apply_common_layout(
        figure=figure,
        title=f"Distribución de {column}",
        description=(
            f"Cantidad de registros para cada valor de {column}."
        ),
    )


def _create_numeric_distribution_chart(
    result: dict[str, Any],
) -> go.Figure:
    """
    Construye un histograma para una distribución numérica.

    Args:
        result: Resultado generado por la herramienta numérica.

    Returns:
        Figura de Plotly.
    """
    column = result.get("column")
    values = result.get("values")

    if not column or not isinstance(values, list):
        raise VisualizationError(
            "El resultado numérico no contiene la estructura esperada."
        )

    dataframe = pd.DataFrame(
        {
            str(column): values,
        }
    )

    figure = px.histogram(
        dataframe,
        x=str(column),
        nbins=min(20, max(5, len(values) // 3)),
        labels={
            str(column): str(column),
            "count": "Cantidad de registros",
        },
    )

    figure.update_traces(
        hovertemplate=(
            f"{column}: %{{x}}"
            "<br>Registros: %{y}"
            "<extra></extra>"
        ),
    )

    figure.update_xaxes(
        title_text=str(column),
    )

    figure.update_yaxes(
        title_text="Cantidad de registros",
    )

    return _apply_common_layout(
        figure=figure,
        title=f"Distribución de {column}",
        description=(
            f"Frecuencia de los valores observados en {column}."
        ),
    )


def _create_correlation_chart(
    result: dict[str, Any],
) -> go.Figure:
    """
    Construye un mapa de calor a partir de una matriz de correlación.

    Args:
        result: Resultado generado por la herramienta de correlaciones.

    Returns:
        Figura de Plotly.
    """
    columns = result.get("columns")
    values = result.get("values")

    if not isinstance(columns, list) or not isinstance(values, dict):
        raise VisualizationError(
            "El resultado de correlación no contiene la estructura esperada."
        )

    matrix = [
        [
            float(values[row][column])
            for column in columns
        ]
        for row in columns
    ]

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=columns,
            y=columns,
            zmin=-1,
            zmax=1,
            text=matrix,
            texttemplate="%{text:.2f}",
            hovertemplate=(
                "Variable X: %{x}"
                "<br>Variable Y: %{y}"
                "<br>Correlación: %{z:.2f}"
                "<extra></extra>"
            ),
            colorbar={
                "title": "Correlación",
            },
        )
    )

    figure.update_xaxes(
        title_text="Variables",
    )

    figure.update_yaxes(
        title_text="Variables",
    )

    return _apply_common_layout(
        figure=figure,
        title="Matriz de correlaciones",
        description=(
            "Relación lineal entre las variables numéricas."
        ),
    )


def _create_scatter_chart(
    result: dict[str, Any],
) -> go.Figure:
    """
    Construye un gráfico de dispersión.

    Args:
        result: Resultado generado por la herramienta de dispersión.

    Returns:
        Figura de Plotly.
    """
    x_column = result.get("x_column")
    y_column = result.get("y_column")
    x_values = result.get("x_values")
    y_values = result.get("y_values")

    if not x_column or not y_column:
        raise VisualizationError(
            "El resultado de dispersión no contiene las columnas."
        )

    if not isinstance(x_values, list) or not isinstance(
        y_values,
        list,
    ):
        raise VisualizationError(
            "El resultado de dispersión no contiene los valores esperados."
        )

    if len(x_values) != len(y_values):
        raise VisualizationError(
            "Los valores de los ejes X e Y tienen longitudes diferentes."
        )

    dataframe = pd.DataFrame(
        {
            str(x_column): x_values,
            str(y_column): y_values,
        }
    )

    figure = px.scatter(
        dataframe,
        x=str(x_column),
        y=str(y_column),
        labels={
            str(x_column): str(x_column),
            str(y_column): str(y_column),
        },
    )

    figure.update_traces(
        marker={
            "size": 9,
            "opacity": 0.75,
        },
        hovertemplate=(
            f"{x_column}: %{{x}}"
            f"<br>{y_column}: %{{y}}"
            "<extra></extra>"
        ),
        name=f"{y_column} vs {x_column}",
    )

    figure.update_xaxes(
        title_text=str(x_column),
    )

    figure.update_yaxes(
        title_text=str(y_column),
    )

    return _apply_common_layout(
        figure=figure,
        title=f"{y_column} vs {x_column}",
        description=(
            f"Relación entre {x_column} y {y_column}."
        ),
    )



def _create_grouped_bar_chart(
    result: dict[str, Any],
) -> go.Figure:
    """
    Construye un gráfico de barras agrupadas desde evidencia SQL.

    Args:
        result: Resultado generado por la herramienta SQL de visualización.

    Returns:
        Figura de Plotly.

    Raises:
        VisualizationError:
            Si el resultado no contiene la estructura esperada.
    """
    x_column = result.get("x_column")
    series_column = result.get("series_column")
    metric_column = result.get("metric_column")
    data = result.get("data")

    if not x_column or not series_column or not metric_column:
        raise VisualizationError(
            "El resultado agrupado no contiene las columnas esperadas."
        )

    if not isinstance(data, list) or not data:
        raise VisualizationError(
            "El resultado agrupado no contiene datos válidos."
        )

    if not all(isinstance(row, dict) for row in data):
        raise VisualizationError(
            "Los datos del resultado agrupado deben ser diccionarios."
        )

    required_columns = {
        str(x_column),
        str(series_column),
        str(metric_column),
    }

    if not all(required_columns.issubset(row.keys()) for row in data):
        raise VisualizationError(
            "Los datos del resultado agrupado no contienen las columnas requeridas."
        )

    dataframe = pd.DataFrame(data)

    if not pd.api.types.is_numeric_dtype(dataframe[metric_column]):
        raise VisualizationError(
            f"La métrica '{metric_column}' debe ser numérica."
        )

    figure = px.bar(
        dataframe,
        x=str(x_column),
        y=str(metric_column),
        color=str(series_column),
        barmode="group",
        labels={
            str(x_column): str(x_column),
            str(series_column): str(series_column),
            str(metric_column): str(metric_column),
        },
    )

    figure.update_traces(
        hovertemplate=(
            f"{x_column}: %{{x}}"
            f"<br>{series_column}: %{{fullData.name}}"
            f"<br>{metric_column}: %{{y}}"
            "<extra></extra>"
        ),
    )

    figure.update_xaxes(
        title_text=str(x_column),
    )

    figure.update_yaxes(
        title_text=str(metric_column),
        rangemode="tozero",
    )

    return _apply_common_layout(
        figure=figure,
        title=f"{metric_column} por {x_column} y {series_column}",
        description=(
            f"Comparación de {metric_column} para cada combinación "
            f"de {x_column} y {series_column}."
        ),
    )

def create_chart(
    result: dict[str, Any],
) -> go.Figure:
    """
    Construye una visualización según el tipo de resultado recibido.

    Args:
        result: Resultado almacenado en chart_results.

    Returns:
        Figura de Plotly.

    Raises:
        VisualizationError:
            Si el resultado no es válido o su tipo no está soportado.
    """
    if not isinstance(result, dict):
        raise VisualizationError(
            "El resultado de visualización debe ser un diccionario."
        )

    chart_type = result.get("type")

    if chart_type == "categorical_distribution":
        return _create_categorical_chart(result)

    if chart_type == "numeric_distribution":
        return _create_numeric_distribution_chart(result)

    if chart_type == "correlation_matrix":
        return _create_correlation_chart(result)

    if chart_type == "scatter":
        return _create_scatter_chart(result)

    if chart_type == "grouped_bar":
        return _create_grouped_bar_chart(result)

    raise VisualizationError(
        f"Tipo de visualización no soportado: {chart_type!r}."
    )


def create_charts(
    results: list[dict[str, Any]],
) -> list[go.Figure]:
    """
    Construye las visualizaciones correspondientes a una colección.

    Los resultados inválidos se registran y se omiten para permitir
    que las demás visualizaciones continúen mostrándose.

    Args:
        results: Resultados almacenados en chart_results.

    Returns:
        Lista de figuras de Plotly válidas.
    """
    figures: list[go.Figure] = []

    for index, result in enumerate(results, start=1):
        try:
            figure = create_chart(result)
            figures.append(figure)

        except VisualizationError as error:
            logger.warning(
                "No fue posible construir el gráfico %s: %s",
                index,
                error,
            )

    logger.info(
        "Se construyeron %s visualización(es) a partir de %s resultado(s).",
        len(figures),
        len(results),
    )

    return figures
