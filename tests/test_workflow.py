"""Pruebas para el workflow principal de análisis."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from app.graph.workflow import (
    WorkflowError,
    create_initial_state,
    run_analysis,
)


def test_create_initial_state() -> None:
    """Verifica la creación correcta del estado inicial."""

    dataframe = pd.DataFrame(
        {
            "producto": ["A", "B", "C"],
            "ventas": [100, 200, 150],
        }
    )

    state = create_initial_state(
        dataframe=dataframe,
        dataset_name="ventas.csv",
        user_question="¿Cuál es el producto con mayores ventas?",
    )

    assert state["dataset"] is dataframe
    assert state["dataset_name"] == "ventas.csv"
    assert (
        state["user_question"]
        == "¿Cuál es el producto con mayores ventas?"
    )

    assert state["dataset_info"] == {
        "rows": 3,
        "columns": 2,
    }

    assert state["dataset_schema"] == {}
    assert state["missing_values"] == {}
    assert state["duplicate_info"] == {}
    assert state["numeric_summary"] == {}
    assert state["categorical_summary"] == {}
    assert state["outliers"] == {}
    assert state["correlations"] == {}

    assert state["eda_charts"] == []
    assert state["chart_results"] == []

    assert state["sql_query"] == ""
    assert state["sql_results"] == {}

    assert state["eda_analysis"] == ""
    assert state["narrative"] == ""
    assert state["final_report"] == {}

    assert state["errors"] == []
    assert state["messages"] == []


def test_create_initial_state_rejects_empty_dataset() -> None:
    """Verifica que no se permita iniciar con un dataset vacío."""

    dataframe = pd.DataFrame()

    with pytest.raises(WorkflowError):
        create_initial_state(
            dataframe=dataframe,
            dataset_name="datos.csv",
            user_question="Analiza los datos.",
        )


def test_create_initial_state_rejects_invalid_dataframe() -> None:
    """Verifica que el dataset sea un DataFrame."""

    with pytest.raises(WorkflowError):
        create_initial_state(
            dataframe="dataset inválido",
            dataset_name="datos.csv",
            user_question="Analiza los datos.",
        )


def test_create_initial_state_rejects_empty_dataset_name() -> None:
    """Verifica que el nombre del dataset sea obligatorio."""

    dataframe = pd.DataFrame(
        {
            "valor": [1, 2, 3],
        }
    )

    with pytest.raises(WorkflowError):
        create_initial_state(
            dataframe=dataframe,
            dataset_name="   ",
            user_question="Analiza los datos.",
        )


def test_create_initial_state_rejects_empty_question() -> None:
    """Verifica que la pregunta del usuario sea obligatoria."""

    dataframe = pd.DataFrame(
        {
            "valor": [1, 2, 3],
        }
    )

    with pytest.raises(WorkflowError):
        create_initial_state(
            dataframe=dataframe,
            dataset_name="datos.csv",
            user_question="   ",
        )


def test_run_analysis_returns_workflow_result() -> None:
    """Verifica la ejecución del workflow sin llamar a DeepSeek."""

    dataframe = pd.DataFrame(
        {
            "producto": ["A", "B"],
            "ventas": [100, 200],
        }
    )

    expected_result = {
        "dataset_name": "ventas.csv",
        "narrative": "Análisis completado.",
    }

    mock_workflow = Mock()
    mock_workflow.invoke.return_value = expected_result

    with patch(
        "app.graph.workflow.create_analysis_workflow",
        return_value=mock_workflow,
    ):
        result = run_analysis(
            dataframe=dataframe,
            dataset_name="ventas.csv",
            user_question="¿Cuál vende más?",
        )

    assert result == expected_result

    mock_workflow.invoke.assert_called_once()


def test_run_analysis_converts_execution_error() -> None:
    """Verifica que los errores internos se conviertan en WorkflowError."""

    dataframe = pd.DataFrame(
        {
            "producto": ["A", "B"],
            "ventas": [100, 200],
        }
    )

    mock_workflow = Mock()

    mock_workflow.invoke.side_effect = RuntimeError(
        "Error simulado."
    )

    with patch(
        "app.graph.workflow.create_analysis_workflow",
        return_value=mock_workflow,
    ):
        with pytest.raises(WorkflowError) as error:
            run_analysis(
                dataframe=dataframe,
                dataset_name="ventas.csv",
                user_question="¿Cuál vende más?",
            )

    assert (
        str(error.value)
        == "No fue posible completar el análisis del dataset."
    )