"""Punto de entrada de la aplicación Streamlit."""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from app.config import APP_NAME, APP_VERSION
from app.graph.workflow import WorkflowError, run_analysis
from app.services.dataset import DatasetError, get_dataset_info, load_csv


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - %(name)s - "
        "%(levelname)s - %(message)s"
    ),
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
    """Muestra el componente para cargar un dataset CSV."""
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

    except DatasetError as error:
        logger.warning(
            "No fue posible cargar el dataset '%s': %s",
            uploaded_file.name,
            error,
        )
        st.error(str(error))
        return

    except Exception:
        logger.exception(
            "Error inesperado procesando el dataset '%s'.",
            uploaded_file.name,
        )
        st.error(
            "Ocurrió un error inesperado al procesar el archivo."
        )
        return

    st.success(
        f"Dataset '{uploaded_file.name}' cargado correctamente."
    )

    render_dataset_information(
        dataframe=dataframe,
        dataset_info=dataset_info,
    )

    render_analysis_form(
        dataframe=dataframe,
        dataset_name=uploaded_file.name,
    )


def render_dataset_information(
    dataframe: Any,
    dataset_info: dict[str, int],
) -> None:
    """
    Muestra información básica del dataset cargado.

    Args:
        dataframe: DataFrame cargado desde el archivo CSV.
        dataset_info: Información dimensional del dataset.
    """
    st.subheader("📊 Dataset")

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

    with st.expander("👀 Vista previa del dataset"):
        st.dataframe(
            dataframe.head(10),
            use_container_width=True,
        )


def render_analysis_form(
    dataframe: Any,
    dataset_name: str,
) -> None:
    """
    Muestra el formulario para iniciar el análisis multi-agente.

    Args:
        dataframe: Dataset que será analizado.
        dataset_name: Nombre del archivo cargado.
    """
    st.divider()
    st.subheader("🔎 Análisis multi-agente")

    user_question = st.text_area(
        "¿Qué quieres analizar?",
        placeholder=(
            "Ejemplo: ¿Cuáles son las principales tendencias "
            "del dataset?"
        ),
        help=(
            "Escribe una pregunta concreta sobre los datos. "
            "El supervisor decidirá qué agentes especializados "
            "deben intervenir."
        ),
    )

    analyze_clicked = st.button(
        "🚀 Ejecutar análisis",
        type="primary",
        use_container_width=True,
    )

    if not analyze_clicked:
        return

    question = user_question.strip()

    if not question:
        st.warning(
            "Debes escribir una pregunta antes de ejecutar "
            "el análisis."
        )
        return

    execute_analysis(
        dataframe=dataframe,
        dataset_name=dataset_name,
        user_question=question,
    )


def execute_analysis(
    dataframe: Any,
    dataset_name: str,
    user_question: str,
) -> None:
    """
    Ejecuta el workflow multi-agente y muestra sus resultados.

    Args:
        dataframe: Dataset que será analizado.
        dataset_name: Nombre del archivo cargado.
        user_question: Pregunta realizada por el usuario.
    """
    logger.info(
        "Iniciando análisis desde Streamlit para '%s'.",
        dataset_name,
    )

    with st.spinner(
        "🤖 Los agentes están analizando el dataset..."
    ):
        try:
            result = run_analysis(
                dataframe=dataframe,
                dataset_name=dataset_name,
                user_question=user_question,
            )

        except WorkflowError as error:
            logger.error(
                "El workflow no pudo completar el análisis: %s",
                error,
            )
            st.error(str(error))
            return

        except Exception:
            logger.exception(
                "Error inesperado durante el análisis multi-agente."
            )
            st.error(
                "Ocurrió un error inesperado durante el análisis."
            )
            return

    logger.info(
        "Análisis completado correctamente para '%s'.",
        dataset_name,
    )

    render_analysis_result(result)


def render_analysis_result(
    result: dict[str, Any],
) -> None:
    """
    Muestra el resultado final producido por el workflow.

    Args:
        result: Estado final devuelto por el workflow.
    """
    st.divider()
    st.header("📄 Resultado del análisis")

    narrative = result.get("narrative", "")
    final_report = result.get("final_report", {})

    if narrative:
        st.subheader("📝 Informe")
        st.markdown(narrative)
    else:
        st.info(
            "El workflow no produjo contenido narrativo."
        )

    if final_report:
        st.subheader("📊 Resultados estructurados")
        st.json(final_report)

    render_analysis_metadata(result)


def render_analysis_metadata(
    result: dict[str, Any],
) -> None:
    """
    Muestra información técnica resumida del análisis.

    Args:
        result: Estado final producido por el workflow.
    """
    with st.expander("🔧 Información del análisis"):
        dataset_info = result.get("dataset_info", {})

        if dataset_info:
            st.write("**Dataset:**")
            st.json(dataset_info)

        sql_query = result.get("sql_query", "")

        if sql_query:
            st.write("**Consulta SQL utilizada:**")
            st.code(
                sql_query,
                language="sql",
            )

        errors = result.get("errors", [])

        if errors:
            st.warning("Se produjeron las siguientes incidencias:")
            for error in errors:
                st.write(f"- {error}")


def main() -> None:
    """Ejecuta la aplicación principal."""
    configure_page()
    render_header()
    render_dataset_uploader()


if __name__ == "__main__":
    main()