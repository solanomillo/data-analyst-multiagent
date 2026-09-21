"""Pruebas para las herramientas de análisis de datos."""

import pandas as pd

from app.tools.data_tools import (
    calculate_correlations,
    check_duplicates,
    check_missing_values,
    detect_outliers,
    get_categorical_summary,
    get_dataset_schema,
    get_numeric_summary,
)


def create_test_dataframe() -> pd.DataFrame:
    """Crea un dataset pequeño para las pruebas."""
    return pd.DataFrame(
        {
            "edad": [20, 25, 30, 35, 100],
            "ingresos": [1000, 1500, 2000, 2500, 3000],
            "ciudad": [
                "Salta",
                "Tartagal",
                "Salta",
                "Orán",
                "Salta",
            ],
        }
    )


def test_get_dataset_schema() -> None:
    """Verifica que el esquema del dataset sea correcto."""
    dataframe = create_test_dataframe()

    result = get_dataset_schema(dataframe)

    assert result["rows"] == 5
    assert result["columns"] == 3
    assert len(result["column_details"]) == 3


def test_check_missing_values() -> None:
    """Verifica la detección de valores nulos."""
    dataframe = create_test_dataframe()

    dataframe.loc[0, "edad"] = None

    result = check_missing_values(dataframe)

    assert result["total_null_values"] == 1
    assert result["columns_with_nulls"]["edad"]["count"] == 1


def test_check_duplicates() -> None:
    """Verifica la detección de registros duplicados."""
    dataframe = create_test_dataframe()

    dataframe = pd.concat(
        [dataframe, dataframe.iloc[[0]]],
        ignore_index=True,
    )

    result = check_duplicates(dataframe)

    assert result["duplicate_rows"] == 1


def test_get_numeric_summary() -> None:
    """Verifica las estadísticas de las columnas numéricas."""
    dataframe = create_test_dataframe()

    result = get_numeric_summary(dataframe)

    assert "edad" in result
    assert "ingresos" in result
    assert result["edad"]["count"] == 5.0


def test_get_categorical_summary() -> None:
    """Verifica el análisis de variables categóricas."""
    dataframe = create_test_dataframe()

    result = get_categorical_summary(dataframe)

    assert "ciudad" in result
    assert result["ciudad"]["unique_values"] == 3


def test_detect_outliers() -> None:
    """Verifica la detección de posibles valores atípicos."""
    dataframe = create_test_dataframe()

    result = detect_outliers(dataframe)

    assert "edad" in result
    assert result["edad"]["outlier_count"] >= 1


def test_calculate_correlations() -> None:
    """Verifica el cálculo de correlaciones."""
    dataframe = create_test_dataframe()

    result = calculate_correlations(dataframe)

    assert "edad" in result
    assert "ingresos" in result
    assert result["edad"]["ingresos"] > 0