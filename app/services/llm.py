"""Servicio centralizado para la interacción con el modelo de lenguaje."""

from __future__ import annotations

import logging

from langchain_deepseek import ChatDeepSeek

from app.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL


logger = logging.getLogger(__name__)


class LLMConfigurationError(Exception):
    """Error relacionado con la configuración del modelo de lenguaje."""


def get_llm() -> ChatDeepSeek:
    """
    Crea y devuelve la instancia configurada del modelo DeepSeek.

    Returns:
        Instancia de ChatDeepSeek lista para utilizarse en los agentes.

    Raises:
        LLMConfigurationError:
            Si no se encuentra configurada la API Key de DeepSeek.
    """
    if not DEEPSEEK_API_KEY:
        logger.error(
            "No se encontró la API Key de DeepSeek."
        )

        raise LLMConfigurationError(
            "La API Key de DeepSeek no está configurada."
        )

    logger.info(
        "Inicializando modelo DeepSeek: %s",
        DEEPSEEK_MODEL,
    )

    return ChatDeepSeek(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        temperature=0,
    )