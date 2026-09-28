"""Pruebas del SQL Analyst Agent."""

from unittest.mock import Mock, patch

from app.agents.sql_analyst import create_sql_analyst_agent
from app.prompts.prompts import SQL_ANALYST_SYSTEM_PROMPT


def test_sql_analyst_prompt_defines_required_process() -> None:
    """Verifica que el prompt defina el proceso obligatorio de análisis."""
    prompt = SQL_ANALYST_SYSTEM_PROMPT.lower()

    required_rules = [
        "inspecciona primero el esquema",
        "analiza la intención de la pregunta",
        "identifica las columnas reales",
        "determina la operación analítica",
        "construye una consulta sql",
        "ejecuta la consulta",
        "comprueba que el resultado obtenido realmente responde",
    ]

    for rule in required_rules:
        assert rule in prompt


def test_sql_analyst_prompt_defines_aggregation_rules() -> None:
    """Verifica las reglas para agregaciones y agrupaciones."""
    prompt = SQL_ANALYST_SYSTEM_PROMPT.lower()

    required_rules = [
        "count",
        "sum",
        "avg",
        "group by",
        "agrupaciones por varias dimensiones",
        "where",
        "order by",
        "having",
    ]

    for rule in required_rules:
        assert rule in prompt


def test_sql_analyst_prompt_handles_multidimensional_questions() -> None:
    """Verifica el tratamiento de preguntas con varias dimensiones."""
    prompt = SQL_ANALYST_SYSTEM_PROMPT.lower()

    required_rules = [
        'distribución "por a y b"',
        "ambas dimensiones conjuntamente",
        "agregación multidimensional",
        "comparación entre grupos",
    ]

    for rule in required_rules:
        assert rule in prompt


def test_sql_analyst_prompt_prevents_unsupported_assumptions() -> None:
    """Verifica que el agente no invente métricas ni columnas."""
    prompt = SQL_ANALYST_SYSTEM_PROMPT.lower()

    required_rules = [
        "no inventes columnas",
        "no inventes valores",
        "no inventes resultados",
        "no asumas que una columna representa una métrica",
        "si una métrica solicitada no existe",
        "no conviertas automáticamente varias columnas en una métrica",
    ]

    for rule in required_rules:
        assert rule in prompt


def test_sql_analyst_prompt_rejects_generic_sample_as_final_analysis() -> None:
    """Verifica que una muestra no sustituya una consulta analítica."""
    prompt = SQL_ANALYST_SYSTEM_PROMPT.lower()

    assert (
        "una consulta que devuelve una muestra de registros puede servir"
        in prompt
    )

    assert (
        "no sustituye una agregación necesaria"
        in prompt
    )


def test_create_sql_analyst_agent_configures_expected_agent() -> None:
    """Verifica la configuración básica del SQL Analyst Agent."""
    fake_model = Mock()

    with patch(
        "app.agents.sql_analyst.get_llm",
        return_value=fake_model,
    ):
        agent = create_sql_analyst_agent()

    assert agent is not None