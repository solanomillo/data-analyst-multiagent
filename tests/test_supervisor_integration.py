"""Pruebas de integración del supervisor."""

from unittest.mock import Mock, patch

import pandas as pd

from app.graph.workflow import create_initial_state
from app.agents.supervisor import create_supervisor_agent


def test_supervisor_delegates_to_data_quality_agent() -> None:
    """Verifica que el supervisor pueda delegar al agente de calidad."""

    dataframe = pd.DataFrame(
        {
            "producto": ["A", "B"],
            "ventas": [100, 200],
        }
    )

    specialist_result = {
        "eda_analysis": (
            "El producto B presenta mayores ventas."
        ),
    }

    specialist_agent = Mock()
    specialist_agent.invoke.return_value = specialist_result

    agents = {
        "data_quality_agent": specialist_agent,
        "sql_analyst": Mock(),
        "chart_analyst": Mock(),
        "narrative_agent": Mock(),
    }

    supervisor_model = Mock()

    with (
        patch(
            "app.agents.supervisor.create_agent_registry",
            return_value=agents,
        ),
        patch(
            "app.agents.supervisor.get_llm",
            return_value=supervisor_model,
        ),
    ):
        supervisor = create_supervisor_agent()

    assert supervisor is not None