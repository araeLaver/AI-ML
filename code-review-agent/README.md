# Code Review Agent

AI-powered multi-agent code review system using LangChain and LangGraph.

## Overview

This project implements an automated code review system with specialized AI agents:

- **🔐 Security Agent**: Detects vulnerabilities (SQL injection, XSS, hardcoded secrets)
- **⚡ Performance Agent**: Identifies optimization opportunities (N+1 queries, complexity)
- **🎨 Style Agent**: Reviews code quality (naming, SOLID principles, readability)

## Architecture

```
User → Streamlit/API → FastAPI → LangGraph Workflow
                                      ↓
                        ┌─────────────┼─────────────┐
                        ↓             ↓             ↓
                   Security      Performance     Style
                    Agent          Agent         Agent
                        ↓             ↓             ↓
                        └─────────────┼─────────────┘
                                      ↓
                               Synthesizer
                                      ↓
                              Final Report
```

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key or Ollama (local LLM)

### Installation

```bash
# Clone and setup
cd code-review-agent
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys
```

### Run

```bash
# Streamlit Demo
streamlit run app/streamlit_app.py

# API Server
uvicorn main:app --reload --port 8080
```

## Usage

### Streamlit Demo

1. Open http://localhost:8501
2. Enter your API key
3. Paste code to review
4. Click "Run Review"

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/api/webhook/github` | GitHub webhook |

### GitHub Integration

1. Create a GitHub webhook pointing to `/api/webhook/github`
2. Set content type to `application/json`
3. Select "Pull requests" events
4. Add webhook secret to `.env`

## Project Structure

```
code-review-agent/
├── agents/                 # AI review agents
│   ├── base.py            # Base agent class
│   ├── security_agent.py  # Security vulnerability detection
│   ├── performance_agent.py  # Performance analysis
│   ├── style_agent.py     # Code quality review
│   └── orchestrator.py    # Multi-agent coordination
├── tools/                  # Utility tools
│   ├── github_tools.py    # GitHub API integration
│   └── code_analyzer.py   # Static code analysis
├── workflows/              # LangGraph workflows
│   └── review_workflow.py # Review pipeline
├── api/                    # FastAPI routes
│   └── webhook.py         # GitHub webhook handler
├── app/                    # Frontend
│   └── streamlit_app.py   # Demo UI
├── tests/                  # Test suite
├── main.py                # Application entry
└── requirements.txt       # Dependencies
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM Framework | LangChain, LangGraph |
| LLM Providers | OpenAI, Ollama |
| Backend | FastAPI |
| Frontend | Streamlit |
| Testing | pytest |

## Development

```bash
# Run tests
pytest tests/ -v

# Format code
black .
ruff check --fix .
```

## License

MIT License
