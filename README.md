# Agentic AI with LangChain and Ollama

An intelligent agent system built with LangChain and Ollama for local LLM inference.

## Features

- Agentic AI with custom tools
- RAG (Retrieval Augmented Generation) capabilities
- Vector-based memory storage
- Local LLM inference with Ollama

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install and run Ollama locally
3. Run the application:
```bash
python app.py
```

## API Endpoint

Expose agent as REST API:
```bash
python api.py
```

**Endpoints:**
- `POST /query` - Execute agent query
  ```bash
  curl -X POST "http://localhost:8000/query" -F "query=list all pods"
  ```

- `POST /analyze` - Analyze screenshot
  ```bash
  curl -X POST "http://localhost:8000/analyze" -F "image=@error.png" -F "query=What error?"
  ```

- `GET /health` - Health check