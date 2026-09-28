"""Pruebas del servicio de carga y validación de datasets."""

from io import BytesIO

import pandas as pd
import pytest

from app.services.dataset import (
    DatasetError,
    get_dataset_info,
    load_csv,
    validate_dataset,
)


def test_load_csv_comma_separator():
    """Debe cargar correctamente un CSV separado por comas."""
    content = b"nombre,edad\nJuan,30\nMaria,25\n"

    dataframe = load_csv(BytesIO(content))

    assert dataframe.shape == (2, 2)
    assert list(dataframe.columns) == ["nombre", "edad"]


def test_load_csv_semicolon_separator():
    """Debe cargar correctamente un CSV separado por punto y coma."""
    content = b"nombre;edad\nJuan;30\nMaria;25\n"

    dataframe = load_csv(BytesIO(content))

    assert dataframe.shape == (2, 2)
    assert list(dataframe.columns) == ["nombre", "edad"]


def test_load_csv_tab_separator():
    """Debe cargar correctamente un CSV separado por tabulaciones."""
    content = b"nombre\tedad\nJuan\t30\nMaria\t25\n"

    dataframe = load_csv(BytesIO(content))

    assert dataframe.shape == (2, 2)
    assert list(dataframe.columns) == ["nombre", "edad"]


def test_load_csv_pipe_separator():
    """Debe cargar correctamente un CSV separado por barras verticales."""
    content = b"nombre|edad\nJuan|30\nMaria|25\n"

    dataframe = load_csv(BytesIO(content))

    assert dataframe.shape == (2, 2)
    assert list(dataframe.columns) == ["nombre", "edad"]


def test_load_csv_utf8():
    """Debe cargar correctamente un CSV codificado en UTF-8."""
    content = "nombre,ciudad\nJosé,Tartagal\n".encode("utf-8")

    dataframe = load_csv(BytesIO(content))

    assert dataframe.shape == (1, 2)
    assert dataframe.iloc[0]["nombre"] == "José"
    assert dataframe.iloc[0]["ciudad"] == "Tartagal"


def test_load_csv_utf8_bom():
    """Debe cargar correctamente un CSV UTF-8 con BOM."""
    content = "\ufeffnombre,edad\nJuan,30\n".encode("utf-8-sig")

    dataframe = load_csv(BytesIO(content))

    assert dataframe.shape == (1, 2)
    assert list(dataframe.columns) == ["nombre", "edad"]


def test_load_csv_cp1252():
    """Debe cargar correctamente un CSV codificado en CP1252."""
    content = "nombre,ciudad\nJosé,Tartagál\n".encode("cp1252")

    dataframe = load_csv(BytesIO(content))

    assert dataframe.shape == (1, 2)
    assert dataframe.iloc[0]["nombre"] == "José"


def test_load_csv_latin1():
    """Debe cargar correctamente un CSV codificado en Latin-1."""
    content = "nombre,ciudad\nJosé,Tartagál\n".encode("latin-1")

    dataframe = load_csv(BytesIO(content))

    assert dataframe.shape == (1, 2)
    assert dataframe.iloc[0]["nombre"] == "José"


def test_load_csv_empty_file():
    """Debe rechazar un archivo CSV completamente vacío."""
    with pytest.raises(DatasetError, match="CSV está vacío"):
        load_csv(BytesIO(b""))


def test_validate_dataset_rejects_empty_dataframe():
    """Debe rechazar un DataFrame sin registros."""
    dataframe = pd.DataFrame()

    with pytest.raises(
        DatasetError,
        match="no contiene registros",
    ):
        validate_dataset(dataframe)


def test_validate_dataset_rejects_dataframe_without_columns():
    """Debe rechazar un DataFrame sin columnas."""
    dataframe = pd.DataFrame(index=[0, 1])

    with pytest.raises(
        DatasetError,
        match="no contiene columnas",
    ):
        validate_dataset(dataframe)


def test_get_dataset_info():
    """Debe devolver correctamente las dimensiones del dataset."""
    dataframe = pd.DataFrame(
        {
            "nombre": ["Juan", "Maria", "Pedro"],
            "edad": [30, 25, 40],
        }
    )

    result = get_dataset_info(dataframe)

    assert result == {
        "rows": 3,
        "columns": 2,
    }