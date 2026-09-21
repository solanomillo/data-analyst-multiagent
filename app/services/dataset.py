"""Servicio para cargar y validar datasets CSV."""

from __future__ import annotations

import logging
from typing import BinaryIO

import pandas as pd


logger = logging.getLogger(__name__)


class DatasetError(Exception):
    """Error controlado relacionado con la carga de un dataset."""


def load_csv(file: BinaryIO) -> pd.DataFrame:
    """
    Carga un archivo CSV y devuelve un DataFrame.

    Args:
        file: Archivo CSV compatible con pandas.read_csv().

    Returns:
        DataFrame con los datos del archivo.

    Raises:
        DatasetError: Si el archivo no puede leerse o está vacío.
    """
    try:
        dataframe = pd.read_csv(file)

    except pd.errors.EmptyDataError as error:
        logger.error("El archivo CSV está vacío.")
        raise DatasetError(
            "El archivo CSV está vacío."
        ) from error

    except pd.errors.ParserError as error:
        logger.error("No fue posible interpretar el archivo CSV.")
        raise DatasetError(
            "No fue posible interpretar el archivo CSV."
        ) from error

    except UnicodeDecodeError as error:
        logger.error("No fue posible decodificar el archivo CSV.")
        raise DatasetError(
            "El archivo CSV utiliza una codificación no compatible."
        ) from error

    except Exception as error:
        logger.exception(
            "Ocurrió un error inesperado al cargar el CSV."
        )
        raise DatasetError(
            "Ocurrió un error inesperado al cargar el archivo."
        ) from error

    validate_dataset(dataframe)

    logger.info(
        "Dataset cargado correctamente: %s filas, %s columnas.",
        dataframe.shape[0],
        dataframe.shape[1],
    )

    return dataframe


def validate_dataset(dataframe: pd.DataFrame) -> None:
    """
    Valida las condiciones básicas de un dataset.

    Args:
        dataframe: DataFrame que se desea validar.

    Raises:
        DatasetError: Si el dataset no contiene datos o columnas.
    """
    if dataframe.empty:
        logger.warning("El dataset no contiene registros.")
        raise DatasetError(
            "El dataset no contiene registros."
        )

    if dataframe.shape[1] == 0:
        logger.warning("El dataset no contiene columnas.")
        raise DatasetError(
            "El dataset no contiene columnas."
        )


def get_dataset_info(dataframe: pd.DataFrame) -> dict[str, int]:
    """
    Obtiene información básica sobre las dimensiones del dataset.

    Args:
        dataframe: DataFrame del que se desea obtener información.

    Returns:
        Diccionario con cantidad de filas y columnas.
    """
    return {
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
    }