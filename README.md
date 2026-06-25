# ⚡ FastAPI LLM Service

A production-ready **AI inference API** built with FastAPI, supporting OpenAI and Anthropic Claude.
Includes streaming, API key auth, health checks, and Docker deployment.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.112+-green?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-ready-blue?style=flat-square)

## Quick Start
```bash
git clone https://github.com/OsamaBinShah1/fastapi-llm-service.git
cd fastapi-llm-service
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
# → Docs at http://localhost:8000/docs
```

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat` | Chat completion |
| POST | `/v1/chat/stream` | Streaming chat (SSE) |
| POST | `/v1/summarize` | Summarise text |
| POST | `/v1/embed` | Generate embeddings |
| GET  | `/health` | Health check |

## Docker
```bash
docker-compose up --build
```

## Author
**Muhammad Osama Bin Shah** — AI Engineer, Frankfurt, Germany
[LinkedIn](https://www.linkedin.com/in/muhammad-osama-bin-shah/)
