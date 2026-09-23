"""Pruebas para el Narrative Agent."""

from unittest.mock import patch

from app.agents.narrative import create_narrative_agent


def test_create_narrative_agent() -> None:
    """Verifica la creación del Narrative Agent."""

    with patch(
        "app.agents.narrative.get_llm"
    ) as mock_get_llm:
        mock_model = object()
        mock_get_llm.return_value = mock_model

        agent = create_narrative_agent()

        assert agent is not None
        mock_get_llm.assert_called_once()