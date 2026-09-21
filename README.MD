# 🤖 Data Analyst Multi-Agent

Sistema de análisis de datos basado en una arquitectura multi-agente
con LangGraph y DeepSeek.

El usuario puede cargar un archivo CSV y solicitar análisis sobre sus
datos. El sistema contará con diferentes agentes especializados para
exploración, consultas, visualizaciones y generación de informes.

## Estado del proyecto

🚧 En desarrollo.

Actualmente se encuentra implementada la primera etapa:

- Interfaz Streamlit.
- Carga de archivos CSV.
- Visualización de información básica.
- Vista previa del dataset.

## Arquitectura

La aplicación evolucionará hacia una arquitectura desacoplada basada
en agentes especializados:

- Supervisor.
- EDA / Data Quality Agent.
- SQL Analyst.
- Chart Analyst.
- Narrative Analyst.

## Tecnologías

- Python
- Streamlit
- Pandas
- LangGraph
- LangChain
- DeepSeek API

## Instalación

Crear y activar un entorno virtual:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Ejecución
```bash
streamlit run app/main.py
```
La aplicación permitirá cargar un archivo CSV y visualizar una primera
información sobre el dataset.
 
---

## Roadmap

- [x] Base inicial del proyecto
- [x] Carga de CSV
- [ ] Servicio de datasets
- [ ] EDA / Data Quality
- [ ] Estado compartido con LangGraph
- [ ] SQL Analyst
- [ ] Chart Analyst
- [ ] Narrative Analyst
- [ ] Supervisor
- [ ] Handoffs y routing
- [ ] Tests
- [ ] LangSmith
- [ ] Docker
- [ ] Deploy


