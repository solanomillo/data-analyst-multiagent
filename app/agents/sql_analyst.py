"""Agente especializado en análisis mediante SQL."""

from __future__ import annotations

import logging

from langchain.agents import create_agent

from app.graph.state import AnalysisState
from app.prompts.prompts import SQL_ANALYST_SYSTEM_PROMPT
from app.services.llm import get_llm
from app.tools.sql_tools import (
    inspect_sql_schema,
    run_sql_query,
)


logger = logging.getLogger(__name__)


def create_sql_analyst_agent():
    """
    Crea el agente especializado en consultas SQL.

    Returns:
        Agente LangChain configurado para análisis SQL.
    """
    logger.info(
        "Creando SQL Analyst Agent."
    )

    return create_agent(
        model=get_llm(),
        tools=[
            inspect_sql_schema,
            run_sql_query,
        ],
        system_prompt=SQL_ANALYST_SYSTEM_PROMPT,
        state_schema=AnalysisState,
        name="sql_analyst",
    )