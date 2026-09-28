"""Servicio para cargar y validar datasets CSV."""

from __future__ import annotations

import csv
import logging
from io import StringIO
from typing import BinaryIO

import pandas as pd


logger = logging.getLogger(__name__)


class DatasetError(Exception):
    """Error controlado relacionado con la carga de un dataset."""


_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "latin-1",
)

_SEPARATORS = (
    ",",
    ";",
    "\t",
    "|",
)


def load_csv(file: BinaryIO) -> pd.DataFrame:
    """
    Carga un archivo CSV y devuelve un DataFrame.

    La función intenta detectar automáticamente una codificación
    y un separador habituales en archivos CSV.

    Args:
        file: Archivo CSV compatible con lectura binaria.

    Returns:
        DataFrame con los datos del archivo.

    Raises:
        DatasetError:
            Si el archivo no puede leerse, está vacío o no contiene
            una estructura válida.
    """
    try:
        content = file.read()

    except Exception as error:
        logger.exception(
            "No fue posible leer el archivo CSV."
        )
        raise DatasetError(
            "No fue posible leer el archivo CSV."
        ) from error

    if not content:
        logger.error("El archivo CSV está vacío.")
        raise DatasetError(
            "El archivo CSV está vacío."
        )

    last_error: Exception | None = None

    for encoding in _ENCODINGS:
        try:
            text = content.decode(encoding)

        except UnicodeDecodeError as error:
            last_error = error
            continue

        try:
            separator = _detect_separator(text)
            dataframe = pd.read_csv(
                StringIO(text),
                sep=separator,
            )

        except pd.errors.EmptyDataError as error:
            logger.error("El archivo CSV está vacío.")
            raise DatasetError(
                "El archivo CSV está vacío."
            ) from error

        except pd.errors.ParserError as error:
            last_error = error
            continue

        validate_dataset(dataframe)

        logger.info(
            "CSV cargado correctamente usando encoding=%s "
            "y separator=%r.",
            encoding,
            separator,
        )

        logger.info(
            "Dataset cargado correctamente: %s filas, %s columnas.",
            dataframe.shape[0],
            dataframe.shape[1],
        )

        return dataframe

    if isinstance(last_error, UnicodeDecodeError):
        logger.error(
            "No fue posible decodificar el archivo CSV "
            "con las codificaciones compatibles."
        )
        raise DatasetError(
            "El archivo CSV utiliza una codificación no compatible."
        ) from last_error

    logger.error(
        "No fue posible interpretar la estructura del archivo CSV."
    )
    raise DatasetError(
        "No fue posible interpretar el archivo CSV."
    ) from last_error


def _detect_separator(text: str) -> str:
    """
    Detecta el separador utilizado por el archivo CSV.

    Args:
        text: Contenido del archivo decodificado.

    Returns:
        Separador detectado.

    Raises:
        DatasetError:
            Si no es posible determinar un separador válido.
    """
    sample = text[:10_000]

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=_SEPARATORS,
        )

    except csv.Error as error:
        logger.error(
            "No fue posible detectar el separador del CSV."
        )
        raise DatasetError(
            "No fue posible determinar el separador del archivo CSV."
        ) from error

    return dialect.delimiter


def validate_dataset(dataframe: pd.DataFrame) -> None:
    """
    Valida las condiciones básicas de un dataset.

    Args:
        dataframe: DataFrame que se desea validar.

    Raises:
        DatasetError:
            Si el dataset no contiene registros o columnas.
    """
    if dataframe.shape[0] == 0:
        logger.warning(
            "El dataset no contiene registros."
        )
        raise DatasetError(
            "El dataset no contiene registros."
        )

    if dataframe.shape[1] == 0:
        logger.warning(
            "El dataset no contiene columnas."
        )
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