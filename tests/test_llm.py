"""Pruebas para el servicio de modelos de lenguaje."""

import pytest

from app.services import llm


def test_get_llm_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifica que la API Key sea obligatoria."""

    monkeypatch.setattr(
        llm,
        "DEEPSEEK_API_KEY",
        None,
    )

    with pytest.raises(llm.LLMConfigurationError):
        llm.get_llm()


def test_get_llm_creates_deepseek_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifica la creación del modelo DeepSeek sin realizar una llamada."""

    monkeypatch.setattr(
        llm,
        "DEEPSEEK_API_KEY",
        "test-api-key",
    )

    monkeypatch.setattr(
        llm,
        "DEEPSEEK_MODEL",
        "deepseek-flash",
    )

    model = llm.get_llm()

    assert isinstance(
        model,
        llm.ChatDeepSeek,
    )
    assert model.model_name == "deepseek-flash"