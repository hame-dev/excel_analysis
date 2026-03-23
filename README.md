# Excel Analysis Agent

A Django-based AI-powered data analysis application that lets users upload Excel/CSV files and ask natural language questions about their data. Built with local LLMs via Ollama, LangChain agents, and vector search.

## Features

- **File Upload** — Upload Excel (.xlsx, .xls) and CSV files with configurable null-value handling strategies (forward fill, backward fill, mean fill, zero fill, etc.)
- **Natural Language Chat** — Ask questions about your data in plain English; an AI agent translates them into SQL queries, Python computations, or semantic lookups
- **Visualization** — Auto-generated charts and plots via Matplotlib/Seaborn, plus a dedicated data exploration page with filtering and pagination
- **ReAct Agent** — Multi-tool reasoning agent with SQL, Python execution, and vector search capabilities
- **Streaming Responses** — Real-time Server-Sent Events for agent reasoning traces
- **Vector Search** — Semantic search over your data using ChromaDB embeddings
- **Configurable Models** — Choose Ollama models for QA, chat, and embeddings via the settings page

## Tech Stack

- **Backend:** Django, LangChain, Pandas, NumPy
- **LLM:** Ollama (local models, default: qwen3.5:9b)
- **Vector Store:** ChromaDB with nomic-embed-text embeddings
- **Database:** SQLite
- **Visualization:** Matplotlib, Seaborn

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally

## Setup

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd excel_analysis
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Pull the required Ollama models**

   ```bash
   ollama pull qwen3.5:9b
   ollama pull nomic-embed-text
   ```

4. **Run migrations**

   ```bash
   python manage.py migrate
   ```

5. **Start the server**

   ```bash
   python manage.py runserver
   ```

6. Open `http://127.0.0.1:8000/` in your browser.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_QA_MODEL` | `qwen3.5:9b` | Model for SQL/data queries |
| `OLLAMA_CHAT_MODEL` | `qwen3.5:9b` | Model for conversational responses |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model for vector search |

These can also be configured from the in-app Settings page.

## Usage

1. **Upload** — Go to `/upload` and upload an Excel or CSV file. Choose a null-handling strategy if your data has missing values.
2. **Chat** — Navigate to `/chat`, create a session, and ask questions about your data (e.g., "What is the average unemployment rate by region?").
3. **Visualize** — Use `/visualization` to filter, sort, and paginate through your data.
4. **Settings** — Configure Ollama models and connection at `/settings`.

## Project Structure

```
excel_analysis/
├── ai_analysis/            # Main Django app
│   ├── services/           # Business logic
│   │   ├── agent.py        # ReAct agent engine
│   │   ├── data_ingestion.py   # File upload & processing
│   │   ├── data_query.py       # Filtering & visualization
│   │   ├── prompting.py        # Dynamic prompt construction
│   │   ├── app_settings.py     # Ollama config management
│   │   └── runtime.py          # In-memory state management
│   ├── models.py           # Database models
│   ├── views.py            # API endpoints
│   ├── templates/          # HTML templates
│   └── tests/              # Test suite
├── excel_agent/            # Django project settings
├── data/                   # Vector store index
├── media/                  # Uploaded files & generated plots
├── requirements.txt
└── manage.py
```

## Running Tests

```bash
pytest
```

## License

This project is for personal/educational use.
