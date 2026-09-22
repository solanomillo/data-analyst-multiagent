"""Pruebas para el estado compartido del grafo."""

import pandas as pd

from app.graph.state import AnalysisState 


def test_analysis_state_initialization() -> None:
    """Verifica la creación del estado inicial."""
    dataframe = pd.DataFrame(
        {
            "producto": ["A", "B"],
            "ventas": [100, 200],
        }
    )

    state: AnalysisState = {
        "dataset": dataframe,
        "dataset_name": "ventas.csv",
        "user_question": "¿Cuál producto tuvo más ventas?",
    }

    assert state["dataset_name"] == "ventas.csv"
    assert state["user_question"] == (
        "¿Cuál producto tuvo más ventas?"
    )
    assert len(state["dataset"]) == 2