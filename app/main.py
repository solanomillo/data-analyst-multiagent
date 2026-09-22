"""Punto de entrada de la aplicación Streamlit."""

import logging

import streamlit as st

from config import APP_NAME, APP_VERSION
from services.dataset import DatasetError, get_dataset_info, load_csv
from tools.data_tools import (
    calculate_correlations,
    check_duplicates,
    check_missing_values,
    detect_outliers,
    get_categorical_summary,
    get_dataset_schema,
    get_numeric_summary,
)
from tools.chart_tools import (
    get_categorical_counts,
    get_correlation_matrix,
    get_numeric_histograms,
)
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def configure_page() -> None:
    """Configura los parámetros principales de la página."""
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🤖",
        layout="wide",
    )


def render_header() -> None:
    """Muestra el encabezado principal de la aplicación."""
    st.title("🤖 Data Analyst Multi-Agent")
    st.caption(
        f"Analista de datos multi-agente · versión {APP_VERSION}"
    )


def render_dataset_uploader() -> None:
    """Muestra el componente para cargar archivos CSV."""
    st.subheader("📁 Cargar dataset")

    uploaded_file = st.file_uploader(
        "Selecciona un archivo CSV",
        type=["csv"],
        help="Carga un archivo CSV para comenzar el análisis.",
    )

    if uploaded_file is None:
        st.info(
            "Carga un archivo CSV para comenzar a trabajar "
            "con el dataset."
        )
        return

    try:
        dataframe = load_csv(uploaded_file)
        dataset_info = get_dataset_info(dataframe)

        st.success(
            f"Dataset '{uploaded_file.name}' cargado correctamente."
        )

        render_dataset_metrics(dataset_info)
        render_dataset_preview(dataframe)
        render_eda(dataframe)

    except DatasetError as error:
        logger.warning(
            "No fue posible cargar el dataset '%s': %s",
            uploaded_file.name,
            error,
        )

        st.error(str(error))

    except Exception:
        logger.exception(
            "Error inesperado procesando el dataset '%s'.",
            uploaded_file.name,
        )

        st.error(
            "Ocurrió un error inesperado al procesar el archivo."
        )


def render_dataset_metrics(dataset_info: dict[str, int]) -> None:
    """
    Muestra las métricas básicas del dataset.

    Args:
        dataset_info: Información de dimensiones del dataset.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Filas",
            f"{dataset_info['rows']:,}",
        )

    with col2:
        st.metric(
            "Columnas",
            f"{dataset_info['columns']:,}",
        )


def render_dataset_preview(dataframe: pd.DataFrame) -> None:
    """
    Muestra una vista previa del dataset.

    Args:
        dataframe: DataFrame cargado.
    """
    st.subheader("👀 Vista previa")

    st.dataframe(
        dataframe.head(10),
        use_container_width=True,
    )


def render_eda(dataframe: pd.DataFrame) -> None:
    """
    Muestra los resultados del análisis exploratorio inicial.

    Args:
        dataframe: DataFrame que se desea analizar.
    """
    st.divider()
    st.header("🔎 Exploración del dataset")

    render_schema(dataframe)
    render_missing_values(dataframe)
    render_duplicates(dataframe)
    render_numeric_statistics(dataframe)
    render_categorical_statistics(dataframe)
    render_outliers(dataframe)
    render_correlations(dataframe)
    render_eda_charts(dataframe)
    

def render_schema(dataframe: pd.DataFrame) -> None:
    """Muestra el esquema del dataset."""
    schema = get_dataset_schema(dataframe)

    st.subheader("📋 Esquema")

    schema_rows = schema["column_details"]

    st.dataframe(
        schema_rows,
        use_container_width=True,
    )


def render_missing_values(dataframe: pd.DataFrame) -> None:
    """Muestra información sobre valores nulos."""
    missing_values = check_missing_values(dataframe)

    st.subheader("⚠️ Valores nulos")

    total_nulls = missing_values["total_null_values"]

    if total_nulls == 0:
        st.success("No se encontraron valores nulos.")
        return

    st.warning(
        f"Se encontraron {total_nulls:,} valores nulos."
    )

    st.dataframe(
        missing_values["columns_with_nulls"],
        use_container_width=True,
    )


def render_duplicates(dataframe: pd.DataFrame) -> None:
    """Muestra información sobre registros duplicados."""
    duplicates = check_duplicates(dataframe)

    st.subheader("🔁 Registros duplicados")

    duplicate_count = duplicates["duplicate_rows"]
    percentage = duplicates["percentage"]

    if duplicate_count == 0:
        st.success("No se encontraron registros duplicados.")
        return

    st.warning(
        f"Se encontraron {duplicate_count:,} registros duplicados "
        f"({percentage:.2f}%)."
    )


def render_numeric_statistics(dataframe: pd.DataFrame) -> None:
    """Muestra estadísticas descriptivas numéricas."""
    statistics = get_numeric_summary(dataframe)

    st.subheader("📊 Estadísticas descriptivas")

    if not statistics:
        st.info(
            "El dataset no contiene columnas numéricas."
        )
        return

    statistics_dataframe = (
        pd.DataFrame(statistics)
        .transpose()
    )

    st.dataframe(
        statistics_dataframe,
        use_container_width=True,
    )


def render_categorical_statistics(dataframe: pd.DataFrame) -> None:
    """Muestra información de variables categóricas."""
    statistics = get_categorical_summary(dataframe)

    st.subheader("🏷️ Variables categóricas")

    if not statistics:
        st.info(
            "El dataset no contiene variables categóricas."
        )
        return

    for column, data in statistics.items():
        with st.expander(column):
            st.write(
                f"Valores únicos: {data['unique_values']}"
            )

            st.dataframe(
                data["top_values"],
                use_container_width=True,
            )


def render_outliers(dataframe: pd.DataFrame) -> None:
    """Muestra información sobre posibles valores atípicos."""
    outliers = detect_outliers(dataframe)

    st.subheader("🚨 Posibles outliers")

    if not outliers:
        st.info(
            "No hay columnas numéricas disponibles para analizar."
        )
        return

    outlier_rows = []

    for column, data in outliers.items():
        outlier_rows.append(
            {
                "Columna": column,
                "Outliers": data["outlier_count"],
                "Porcentaje": data["outlier_percentage"],
                "Límite inferior": data["lower_bound"],
                "Límite superior": data["upper_bound"],
            }
        )

    st.dataframe(
        outlier_rows,
        use_container_width=True,
    )

    st.caption(
        "Los outliers representan posibles valores atípicos "
        "según el método IQR. No implican necesariamente errores."
    )


def render_correlations(dataframe: pd.DataFrame) -> None:
    """Muestra la matriz de correlaciones."""
    correlations = calculate_correlations(dataframe)

    st.subheader("🔗 Correlaciones")

    if not correlations:
        st.info(
            "Se necesitan al menos dos columnas numéricas "
            "para calcular correlaciones."
        )
        return


    correlation_dataframe = pd.DataFrame(correlations)

    st.dataframe(
        correlation_dataframe,
        use_container_width=True,
    )

def render_eda_charts(dataframe: pd.DataFrame) -> None:
    """Muestra las visualizaciones básicas del análisis exploratorio."""
    st.subheader("📈 Visualizaciones exploratorias")

    render_numeric_histograms(dataframe)
    render_categorical_charts(dataframe)
    render_correlation_chart(dataframe)


def render_numeric_histograms(
    dataframe: pd.DataFrame,
) -> None:
    """Muestra histogramas de las variables numéricas."""
    histograms = get_numeric_histograms(dataframe)

    if not histograms:
        st.info(
            "No existen variables numéricas disponibles "
            "para generar histogramas."
        )
        return

    st.markdown("#### Distribución de variables numéricas")

    for column, series in histograms.items():
        st.write(f"**{column}**")
        st.bar_chart(
            series.value_counts().sort_index(),
        )


def render_categorical_charts(
    dataframe: pd.DataFrame,
) -> None:
    """Muestra gráficos de frecuencia de variables categóricas."""
    categorical_counts = get_categorical_counts(dataframe)

    if not categorical_counts:
        st.info(
            "No existen variables categóricas disponibles "
            "para generar gráficos."
        )
        return

    st.markdown("#### Distribución de variables categóricas")

    for column, counts in categorical_counts.items():
        st.write(f"**{column}**")
        st.bar_chart(counts)


def render_correlation_chart(
    dataframe: pd.DataFrame,
) -> None:
    """Muestra la matriz de correlaciones."""
    try:
        correlation_matrix = get_correlation_matrix(dataframe)

    except ValueError:
        st.info(
            "Se necesitan al menos dos variables numéricas "
            "para generar la matriz de correlaciones."
        )
        return

    st.markdown("#### Matriz de correlaciones")

    st.dataframe(
        correlation_matrix,
        use_container_width=True,
    )


def main() -> None:
    """Ejecuta la aplicación principal."""
    configure_page()
    render_header()
    render_dataset_uploader()


if __name__ == "__main__":
    main()