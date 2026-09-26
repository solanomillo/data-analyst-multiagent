"""Punto de entrada de la aplicación Streamlit."""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from app.config import APP_NAME, APP_VERSION
from app.graph.workflow import WorkflowError, run_analysis
from app.services.dataset import DatasetError, get_dataset_info, load_csv
from app.services.visualization import create_charts


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
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main {
            padding-top: 1.5rem;
        }

        .app-hero {
            padding: 2.2rem 2.4rem;
            border-radius: 18px;
            margin-bottom: 1.8rem;
            background: linear-gradient(
                135deg,
                rgba(49, 51, 63, 0.95),
                rgba(35, 38, 52, 0.98)
            );
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .app-hero h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.03em;
        }

        .app-hero p {
            margin: 0.7rem 0 0;
            font-size: 1.08rem;
            line-height: 1.6;
            color: rgba(255, 255, 255, 0.72);
            max-width: 850px;
        }

        .feature-card {
            min-height: 190px;
            padding: 1.25rem;
            border-radius: 14px;
            border: 1px solid rgba(128, 128, 128, 0.22);
            background: rgba(128, 128, 128, 0.06);
            margin-bottom: 1rem;
        }

        .feature-icon {
            font-size: 1.7rem;
            margin-bottom: 0.5rem;
        }

        .feature-title {
            font-size: 1.05rem;
            font-weight: 650;
            margin-bottom: 0.45rem;
        }

        .feature-description {
            font-size: 0.9rem;
            line-height: 1.5;
            opacity: 0.72;
        }

        .workflow-step {
            text-align: center;
            padding: 1rem 0.5rem;
        }

        .workflow-icon {
            font-size: 1.7rem;
            margin-bottom: 0.35rem;
        }

        .workflow-title {
            font-weight: 650;
            font-size: 0.95rem;
        }

        .workflow-description {
            font-size: 0.78rem;
            opacity: 0.65;
            margin-top: 0.25rem;
        }

        .dataset-banner {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.22);
            background: rgba(128, 128, 128, 0.06);
            margin-bottom: 1rem;
        }

        .dataset-name {
            font-size: 1.05rem;
            font-weight: 650;
        }

        .dataset-status {
            font-size: 0.85rem;
            opacity: 0.68;
            margin-top: 0.2rem;
        }

        .question-hint {
            font-size: 0.86rem;
            opacity: 0.68;
            margin-top: -0.4rem;
            margin-bottom: 0.8rem;
        }

        .section-description {
            opacity: 0.68;
            margin-top: -0.5rem;
            margin-bottom: 1.2rem;
        }

        .footer {
            text-align: center;
            padding: 2rem 0 1rem;
            opacity: 0.5;
            font-size: 0.78rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Muestra el encabezado principal y la propuesta de valor."""
    st.markdown(
        f"""
        <div class="app-hero">
            <h1>📊 {APP_NAME}</h1>
            <p>
                Analiza tus datasets con un sistema multi-agente que
                combina control de calidad, análisis estadístico,
                consultas SQL, visualizaciones e interpretación
                automática de resultados.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Analista de datos multi-agente · versión {APP_VERSION}"
    )


def render_sidebar() -> None:
    """Muestra información contextual y ejemplos en la barra lateral."""
    with st.sidebar:
        st.header("💡 ¿Qué puedes hacer?")

        st.markdown(
            """
            **Sube un CSV y formula una pregunta en lenguaje natural.**

            El sistema puede ayudarte a:

            - 🔎 Revisar la calidad de los datos.
            - 📋 Analizar estructura y variables.
            - 📊 Obtener estadísticas descriptivas.
            - 🧮 Realizar consultas SQL.
            - 📈 Generar visualizaciones relevantes.
            - 🔗 Analizar correlaciones y relaciones.
            - ⚠️ Detectar posibles valores atípicos.
            - 📝 Generar un informe explicativo.
            """
        )

        st.divider()

        st.subheader("💬 Ejemplos")

        st.markdown(
            """
            **Exploración**
            
            > ¿Cuáles son las principales tendencias?

            **Calidad**
            
            > ¿Existen valores nulos o duplicados?

            **Relaciones**
            
            > ¿Qué variables están relacionadas?

            **Segmentación**
            
            > ¿Qué categoría tiene más registros?

            **Visualización**
            
            > Genera gráficos para entender los datos.
            """
        )

        st.divider()

        st.caption(
            "Los agentes especializados coordinan el análisis "
            "según la pregunta realizada."
        )


def render_capabilities() -> None:
    """Muestra las principales capacidades de la aplicación."""
    st.subheader("🚀 ¿Qué puede hacer esta aplicación?")

    st.markdown(
        """
        <p class="section-description">
            El sistema combina herramientas determinísticas con
            agentes especializados para transformar un CSV en
            información útil y comprensible.
        </p>
        """,
        unsafe_allow_html=True,
    )

    capabilities = [
        (
            "🔎",
            "Calidad de datos",
            (
                "Revisa estructura, valores nulos, duplicados, "
                "cardinalidad y posibles valores atípicos."
            ),
        ),
        (
            "📊",
            "Análisis estadístico",
            (
                "Obtiene estadísticas descriptivas, distribuciones "
                "y resúmenes de variables numéricas y categóricas."
            ),
        ),
        (
            "🧮",
            "Consultas SQL",
            (
                "Utiliza consultas SQL de lectura para responder "
                "preguntas específicas sobre los datos."
            ),
        ),
        (
            "📈",
            "Visualizaciones",
            (
                "Selecciona y genera gráficos según la pregunta "
                "y la información disponible en el dataset."
            ),
        ),
        (
            "🤖",
            "Multi-Agent",
            (
                "Coordina agentes especializados para dividir "
                "el análisis en tareas independientes."
            ),
        ),
        (
            "📝",
            "Informe automático",
            (
                "Convierte los resultados obtenidos en un informe "
                "narrativo basado en la evidencia disponible."
            ),
        ),
    ]

    for start in range(0, len(capabilities), 3):
        columns = st.columns(3)

        for column, capability in zip(
            columns,
            capabilities[start:start + 3],
        ):
            icon, title, description = capability

            with column:
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <div class="feature-icon">{icon}</div>
                        <div class="feature-title">{title}</div>
                        <div class="feature-description">
                            {description}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_workflow_overview() -> None:
    """Muestra visualmente el flujo general del sistema."""
    st.subheader("⚙️ ¿Cómo funciona?")

    st.markdown(
        """
        <p class="section-description">
            Una pregunta puede activar diferentes agentes
            especializados. El supervisor coordina el proceso
            y reúne los resultados para generar el informe final.
        </p>
        """,
        unsafe_allow_html=True,
    )

    steps = [
        (
            "📁",
            "Dataset",
            "Carga tu archivo CSV",
        ),
        (
            "💬",
            "Pregunta",
            "Formula una pregunta",
        ),
        (
            "🤖",
            "Supervisor",
            "Coordina los agentes",
        ),
        (
            "🔎",
            "Análisis",
            "Procesa la evidencia",
        ),
        (
            "📈",
            "Visualización",
            "Genera gráficos",
        ),
        (
            "📝",
            "Informe",
            "Explica los resultados",
        ),
    ]

    columns = st.columns(len(steps))

    for column, step in zip(columns, steps):
        icon, title, description = step

        with column:
            st.markdown(
                f"""
                <div class="workflow-step">
                    <div class="workflow-icon">{icon}</div>
                    <div class="workflow-title">{title}</div>
                    <div class="workflow-description">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_dataset_uploader() -> None:
    """Muestra el componente para cargar un dataset CSV."""
    st.divider()
    st.subheader("📁 Comienza tu análisis")

    st.markdown(
        """
        <p class="section-description">
            Carga un archivo CSV y luego formula una pregunta
            sobre los datos. El sistema se encargará de seleccionar
            los análisis necesarios.
        </p>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Selecciona un archivo CSV",
        type=["csv"],
        help=(
            "Carga un archivo CSV para comenzar el análisis. "
            "El archivo debe contener al menos una fila y una columna."
        ),
    )

    if uploaded_file is None:
        st.info(
            "👆 Carga un archivo CSV para habilitar el análisis."
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
        dataset_name=uploaded_file.name,
    )

    render_analysis_form(
        dataframe=dataframe,
        dataset_name=uploaded_file.name,
    )


def render_dataset_information(
    dataframe: Any,
    dataset_info: dict[str, int],
    dataset_name: str,
) -> None:
    """
    Muestra información básica del dataset cargado.

    Args:
        dataframe: DataFrame cargado desde el archivo CSV.
        dataset_info: Información dimensional del dataset.
        dataset_name: Nombre del dataset.
    """
    st.divider()
    st.subheader("📊 Dataset cargado")

    st.markdown(
        f"""
        <div class="dataset-banner">
            <div class="dataset-name">📄 {dataset_name}</div>
            <div class="dataset-status">
                Dataset disponible para análisis multi-agente
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
            width="stretch",
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
    st.subheader("💬 ¿Qué quieres descubrir?")

    st.markdown(
        """
        <p class="question-hint">
            Puedes preguntar sobre calidad, tendencias,
            relaciones, categorías, estadísticas o visualizaciones.
            No necesitas escribir una consulta SQL.
        </p>
        """,
        unsafe_allow_html=True,
    )

    user_question = st.text_area(
        "Pregunta de análisis",
        placeholder=(
            "Ejemplo: ¿Cuáles son las principales tendencias "
            "del dataset?"
        ),
        help=(
            "Escribe una pregunta en lenguaje natural. "
            "El supervisor decidirá qué agentes especializados "
            "deben intervenir."
        ),
    )

    st.caption(
        "Ejemplos: "
        "¿Hay valores atípicos? · "
        "¿Qué variables están relacionadas? · "
        "¿Qué categoría concentra más registros?"
    )

    analyze_clicked = st.button(
        "🚀 Analizar dataset",
        type="primary",
        width="stretch",
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
        dataset_name: Nombre del archivo.
        user_question: Pregunta realizada por el usuario.
    """
    logger.info(
        "Iniciando análisis desde Streamlit para '%s'.",
        dataset_name,
    )

    with st.status(
        "🤖 Analizando tu dataset...",
        expanded=True,
    ) as status:
        st.write("🔎 Revisando la calidad y estructura de los datos.")
        st.write("🧮 Determinando qué análisis son necesarios.")
        st.write("📈 Preparando visualizaciones relevantes.")
        st.write("📝 Generando el informe final.")

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
            status.update(
                label="❌ No fue posible completar el análisis.",
                state="error",
            )
            st.error(str(error))
            return

        except Exception:
            logger.exception(
                "Error inesperado durante el análisis multi-agente."
            )
            status.update(
                label="❌ Ocurrió un error durante el análisis.",
                state="error",
            )
            st.error(
                "Ocurrió un error inesperado durante el análisis."
            )
            return

        status.update(
            label="✅ Análisis completado correctamente.",
            state="complete",
        )

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
    st.header("📊 Resultado del análisis")

    narrative = result.get("narrative", "")

    if narrative:
        st.subheader("📝 Informe")
        st.markdown(narrative)
    else:
        st.info(
            "El workflow no produjo contenido narrativo."
        )

    render_visualizations(result)
    render_analysis_metadata(result)


def render_visualizations(
    result: dict[str, Any],
) -> None:
    """
    Renderiza las visualizaciones producidas por el Chart Analyst.

    Args:
        result: Estado final producido por el workflow.
    """
    chart_results = result.get("chart_results", [])

    if not chart_results:
        return

    st.divider()
    st.subheader("📈 Visualizaciones")

    st.markdown(
        """
        <p class="section-description">
            Estas visualizaciones fueron seleccionadas por el
            Chart Analyst según la pregunta y la información
            disponible en el dataset.
        </p>
        """,
        unsafe_allow_html=True,
    )

    figures = create_charts(chart_results)

    if not figures:
        st.info(
            "Se obtuvieron resultados de visualización, "
            "pero no fue posible construir gráficos para mostrarlos."
        )
        return

    for index, figure in enumerate(figures, start=1):
        st.plotly_chart(
            figure,
            width="stretch",
            key=f"analysis-chart-{index}",
        )


def render_analysis_metadata(
    result: dict[str, Any],
) -> None:
    """
    Muestra información técnica resumida del análisis.

    Args:
        result: Estado final producido por el workflow.
    """
    with st.expander("🔧 Detalles técnicos del análisis"):
        dataset_info = result.get("dataset_info", {})

        if dataset_info:
            st.write("**Información del dataset:**")
            st.json(dataset_info)

        sql_query = result.get("sql_query", [])

        if sql_query:
            st.write("**Consultas SQL utilizadas:**")

            for query in sql_query:
                st.code(
                    query,
                    language="sql",
                )

        errors = result.get("errors", [])

        if errors:
            st.warning(
                "Se produjeron las siguientes incidencias:"
            )

            for error in errors:
                st.write(f"- {error}")

        if not dataset_info and not sql_query and not errors:
            st.caption(
                "No hay información técnica adicional disponible."
            )


def render_footer() -> None:
    """Muestra el pie de página de la aplicación."""
    st.markdown(
        f"""
        <div class="footer">
            Data Analyst Multi-Agent · v{APP_VERSION}
            <br>
            Análisis de datos mediante agentes especializados,
            herramientas Python y consultas SQL.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Ejecuta la aplicación principal."""
    configure_page()
    render_sidebar()
    render_header()
    render_capabilities()
    render_workflow_overview()
    render_dataset_uploader()
    render_footer()


if __name__ == "__main__":
    main()