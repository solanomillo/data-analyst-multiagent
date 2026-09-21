"""Punto de entrada de la aplicación Streamlit."""

import streamlit as st

from config import APP_NAME, APP_VERSION


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
        import pandas as pd

        dataframe = pd.read_csv(uploaded_file)

        st.success(
            f"Dataset '{uploaded_file.name}' cargado correctamente."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Filas",
                f"{dataframe.shape[0]:,}",
            )

        with col2:
            st.metric(
                "Columnas",
                f"{dataframe.shape[1]:,}",
            )

        st.subheader("👀 Vista previa")

        st.dataframe(
            dataframe.head(10),
            use_container_width=True,
        )

    except Exception as error:
        st.error(
            "No fue posible cargar el archivo CSV. "
            "Verifica que el archivo tenga un formato válido."
        )

        st.exception(error)


def main() -> None:
    """Ejecuta la aplicación principal."""
    configure_page()
    render_header()
    render_dataset_uploader()


if __name__ == "__main__":
    main()