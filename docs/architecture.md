# Architecture

## Folder Structure

| Folder | Responsibility |
| ----------- | --------------------------------------------------------------------------- |
| `api/` | FastAPI endpoints only (e.g. `POST /chat`) |
| `services/` | Chatbot logic — LangGraph / LangChain workflows |
| `clients/` | Connectors to external systems: Ollama/OpenRouter, ClickHouse, MongoDB, Redis, MCP |
| `models/` | Pydantic request/response models and agent state |
| `prompts/` | System prompts and templates — kept outside Python code |
| `utils/` | Small generic helpers: logging, token validation, ID generation |
| `tests/` | Automated tests |
| `scripts/` | One-off operational commands, evaluation runs, seed scripts |
| `docs/` | Architecture and deployment notes |

---

## Request Flow

```
User request
    ↓
api/chat.py
    ↓
services/chat_service.py
    ↓
services/agent_service.py
    ↓
clients/llm_client.py + database clients + MCP tools
    ↓
Return response
```

---

## Example: "Which sneaker colours are trending?"

```
api/chat.py
  → receives the user request
  → calls chat_service.py

chat_service.py
  → fetches past conversation from Redis
  → calls agent_service.py

agent_service.py
  → identifies intent
  → calls ClickHouse / MongoDB tool
  → uses LLM to generate a business-friendly response

chat_service.py
  → saves the conversation in Redis
  → returns the final answer
```