"""Pruebas del contexto proporcionado al SQL Analyst."""

from unittest.mock import Mock

import pandas as pd
from langchain_core.messages import HumanMessage

from app.tools.agent_tools import (
    _build_sql_context,
    _build_sql_state,
    _run_specialist,
)


def _build_test_state() -> dict:
    """Construye un estado mínimo y genérico para las pruebas."""
    dataset = pd.DataFrame(
        {
            "segmento": [
                "A",
                "B",
                "A",
            ],
            "ingresos": [
                1000.0,
                2000.0,
                1500.0,
            ],
        }
    )

    return {
        "dataset": dataset,
        "dataset_name": "dataset_generico.csv",
        "user_question": (
            "¿Cuál es la distribución de ingresos por segmento?"
        ),
        "dataset_info": {
            "rows": 3,
            "columns": 2,
        },
        "dataset_schema": {
            "columns": [
                {
                    "name": "segmento",
                    "dtype": "object",
                },
                {
                    "name": "ingresos",
                    "dtype": "float64",
                },
            ]
        },
        "missing_values": {},
        "duplicate_info": {
            "duplicate_rows": 0,
        },
        "numeric_summary": {
            "ingresos": {
                "mean": 1500.0,
            }
        },
        "categorical_summary": {
            "segmento": {
                "unique": 2,
            }
        },
        "outliers": {},
        "correlations": {},
        "messages": [],
    }


def test_build_sql_context_includes_user_question() -> None:
    """Verifica que el contexto incluya la pregunta original."""
    state = _build_test_state()

    context = _build_sql_context(state)

    assert state["user_question"] in context


def test_build_sql_context_includes_real_schema() -> None:
    """Verifica que el contexto incluya el schema del dataset."""
    state = _build_test_state()

    context = _build_sql_context(state)

    assert "segmento" in context
    assert "ingresos" in context


def test_build_sql_context_does_not_serialize_dataframe_rows() -> None:
    """
    Verifica que los registros originales no se envíen al contexto.

    El DataFrame debe permanecer disponible para las herramientas
    mediante el estado compartido, pero sus registros no deben formar
    parte del mensaje enviado al modelo.
    """
    state = _build_test_state()

    context = _build_sql_context(state)

    dataframe = state["dataset"]

    for row in dataframe.itertuples(index=False, name=None):
        serialized_row = str(row)

        assert serialized_row not in context


def test_build_sql_state_creates_human_message() -> None:
    """Verifica que el SQL Analyst reciba contexto como mensaje."""
    state = _build_test_state()

    sql_state = _build_sql_state(state)

    assert len(sql_state["messages"]) == 1
    assert isinstance(
        sql_state["messages"][0],
        HumanMessage,
    )

    message = sql_state["messages"][0]

    assert state["user_question"] in message.content
    assert "segmento" in message.content
    assert "ingresos" in message.content


def test_run_specialist_uses_sql_context() -> None:
    """Verifica que SQL Analyst reciba el estado contextualizado."""
    state = _build_test_state()

    agent = Mock()
    agent.invoke.return_value = {
        "sql_query": [
            (
                "SELECT segmento, SUM(ingresos) "
                "FROM dataset "
                "GROUP BY segmento"
            )
        ],
        "sql_results": [
            {
                "segmento": "A",
                "ingresos": 5000,
            }
        ],
    }

    agents = {
        "sql_analyst": agent,
    }

    _run_specialist(
        agent_name="sql_analyst",
        agents=agents,
        state=state,
        tool_call_id="tool-123",
    )

    agent.invoke.assert_called_once()

    specialist_state = agent.invoke.call_args.args[0]

    assert len(specialist_state["messages"]) == 1

    message = specialist_state["messages"][0]

    assert isinstance(message, HumanMessage)
    assert state["user_question"] in message.content
    assert "segmento" in message.content
    assert "ingresos" in message.content

    dataframe = state["dataset"]

    for row in dataframe.itertuples(index=False, name=None):
        serialized_row = str(row)

        assert serialized_row not in message.content