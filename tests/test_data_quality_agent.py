"""Pruebas para el agente de calidad y EDA."""

from unittest.mock import Mock

import pandas as pd

from app.agents import data_quality


def create_test_dataframe() -> pd.DataFrame:
    """Crea un dataset pequeño para las pruebas."""
    return pd.DataFrame(
        {
            "edad": [20, 25, 30, 35, 100],
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


def test_run_data_quality_agent(
    monkeypatch,
) -> None:
    """Verifica la ejecución completa del agente sin llamar a la API."""

    dataframe = create_test_dataframe()

    response = Mock()
    response.content = (
        "El dataset presenta una estructura correcta "
        "y no se observan problemas críticos."
    )

    llm = Mock()
    llm.invoke.return_value = response

    monkeypatch.setattr(
        data_quality,
        "get_llm",
        lambda: llm,
    )

    state = {
        "dataset": dataframe,
        "dataset_name": "test.csv",
        "user_question": "Analiza la calidad de los datos.",
    }

    result = data_quality.run_data_quality_agent(state)

    assert "dataset_schema" in result
    assert "missing_values" in result
    assert "duplicate_info" in result
    assert "numeric_summary" in result
    assert "categorical_summary" in result
    assert "outliers" in result
    assert "correlations" in result
    assert "eda_analysis" in result

    assert (
        result["eda_analysis"]
        == response.content
    )

    llm.invoke.assert_called_once()


def test_run_data_quality_agent_requires_dataframe() -> None:
    """Verifica que el agente rechace un dataset inválido."""

    state = {
        "dataset": "no-es-un-dataframe",
        "dataset_name": "test.csv",
        "user_question": "Analiza los datos.",
    }

    try:
        data_quality.run_data_quality_agent(state)

    except ValueError as error:
        assert str(error) == (
            "El estado del análisis no contiene "
            "un DataFrame válido."
        )

    else:
        raise AssertionError(
            "Se esperaba un ValueError."
        )