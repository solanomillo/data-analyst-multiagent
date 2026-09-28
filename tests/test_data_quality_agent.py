from unittest.mock import Mock

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents import data_quality
from app.prompts.prompts import DATA_QUALITY_SYSTEM_PROMPT
from app.tools.data_tools import inspect_dataset_schema


def create_test_dataframe() -> pd.DataFrame:
    """Crea un DataFrame representativo para las pruebas."""
    return pd.DataFrame(
        {
            "edad": [20, 25, 30, 35, 100],
            "ingresos": [1000, 1500, 2000, 2500, 3000],
            "ciudad": ["Salta", "Tartagal", "Salta", "Orán", "Salta"],
        }
    )


def test_create_data_quality_agent(monkeypatch) -> None:
    """Verifica que el Data Quality Agent pueda crearse correctamente."""
    llm = Mock()
    llm.bind_tools.return_value = llm

    monkeypatch.setattr(
        data_quality,
        "get_llm",
        lambda: llm,
    )

    agent = data_quality.create_data_quality_agent()

    assert agent is not None


def test_run_data_quality_agent(monkeypatch) -> None:
    """Verifica la ejecución básica del Data Quality Agent."""
    dataframe = create_test_dataframe()

    response = AIMessage(
        content=(
            "El dataset presenta una estructura correcta "
            "y no se observan problemas críticos."
        )
    )

    llm = Mock()
    llm.invoke.return_value = response
    llm.bind_tools.return_value = llm

    monkeypatch.setattr(
        data_quality,
        "get_llm",
        lambda: llm,
    )

    agent = data_quality.create_data_quality_agent()

    state = {
        "messages": [
            HumanMessage(
                content="Analiza la calidad de los datos."
            )
        ],
        "dataset": dataframe,
        "dataset_name": "test.csv",
        "user_question": "Analiza la calidad de los datos.",
    }

    result = agent.invoke(state)

    assert "messages" in result
    assert response.content in result["messages"][-1].content
    assert llm.invoke.call_count >= 1


def test_requires_dataframe_is_validated_in_tool() -> None:
    """Verifica que la herramienta valide que el dataset sea un DataFrame."""
    runtime = Mock()
    runtime.state = {
        "dataset": "no-es-un-dataframe",
    }

    with pytest.raises(ValueError, match="DataFrame"):
        inspect_dataset_schema.func(runtime)


def test_data_quality_prompt_defines_conditional_tool_usage() -> None:
    """
    Verifica que el prompt defina cuándo corresponde utilizar
    cada grupo de herramientas de análisis.
    """
    prompt = DATA_QUALITY_SYSTEM_PROMPT

    required_rules = [
        "schema",
        "nulos",
        "duplicados",
        "columnas numéricas",
        "columnas categóricas",
        "al menos 2",
        "No ejecutes herramientas",
    ]

    prompt_lower = prompt.lower()

    for rule in required_rules:
        assert rule.lower() in prompt_lower