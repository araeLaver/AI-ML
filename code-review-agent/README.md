# Code Review Agent

AI-powered multi-agent code review system using LangChain and LangGraph.

## Features

- **Multi-Agent Architecture**: Specialized agents for security, performance, code style, and OWASP
- **OWASP Top 10 Analysis**: Comprehensive vulnerability detection with LLM and static analysis
- **Custom YAML Rules**: Define your own organization-specific rules
- **Multi-Language Support**: Enhanced analysis for Python, Java, Go, TypeScript, Rust
- **Parallel Execution**: Agents run concurrently for faster reviews
- **GitHub Integration**: Automatic PR reviews via webhooks
- **REST API**: Direct code review endpoint for integration
- **Streamlit UI**: Interactive web interface for manual reviews
- **Export**: Download reports as Markdown or JSON

## Agent Overview

| Agent | Focus Areas |
|-------|-------------|
| **Security** | SQL/Command Injection, XSS, Auth issues, Data exposure, Cryptographic weaknesses |
| **Performance** | Algorithm complexity, N+1 queries, Caching opportunities, Async optimization |
| **Style** | Naming conventions, SOLID principles, DRY, Documentation, Code complexity |
| **OWASP** | OWASP Top 10 (2021) vulnerability detection with both LLM and static analysis |

## Architecture

```
                           Code Review Agent
                                  |
            +---------------------+---------------------+
            |                     |                     |
       Streamlit UI          FastAPI Server       GitHub Webhook
            |                     |                     |
            +---------------------+---------------------+
                                  |
                         Review Orchestrator
                                  |
            +----------+----------+----------+---------+
            |          |          |          |         |
        Security  Performance   Style     OWASP    Custom
         Agent      Agent       Agent     Agent    Rules
            |          |          |          |         |
            +----------+----------+----------+---------+
                                  |
                            Synthesizer
                                  |
                         Final Report + Verdict
```

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key or Ollama (local LLM)

### Installation

```bash
# Clone repository
cd code-review-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

```env
# Required (one of)
OPENAI_API_KEY=sk-...
OLLAMA_URL=http://localhost:11434

# Optional
OPENAI_MODEL=gpt-4o-mini
OLLAMA_MODEL=llama3.2
GITHUB_TOKEN=ghp_...
GITHUB_WEBHOOK_SECRET=your-secret
CORS_ORIGINS=*
APP_ENV=development
```

### Run

```bash
# Streamlit Demo (recommended for testing)
streamlit run app/streamlit_app.py

# API Server
uvicorn main:app --reload --port 8080

# Or with Python
python main.py
```

## Usage

### Streamlit Demo

1. Open http://localhost:8501
2. Enter your OpenAI API key (or select Ollama)
3. Paste code or upload a file
4. Click "Run AI Review"
5. View results by category (Security/Performance/Style)
6. Export report as Markdown or JSON

### REST API

#### Review Code

```bash
# Synchronous review
curl -X POST http://localhost:8080/api/review \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def hello(): print(\"Hello World\")",
    "language": "python"
  }'

# Async review (for large files)
curl -X POST http://localhost:8080/api/review/async \
  -H "Content-Type: application/json" \
  -d '{
    "code": "...",
    "language": "python"
  }'

# Check async status
curl http://localhost:8080/api/review/{review_id}
```

#### Quick Check (No LLM)

```bash
curl -X POST http://localhost:8080/api/quick-check \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def hello(): pass",
    "language": "python"
  }'
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/docs` | OpenAPI documentation |
| POST | `/api/review` | Synchronous code review |
| POST | `/api/review/async` | Async code review |
| GET | `/api/review/{id}` | Get async review status |
| POST | `/api/quick-check` | Static analysis only |
| GET | `/api/supported-languages` | List supported languages |
| POST | `/api/webhook/github` | GitHub webhook handler |
| GET | `/api/webhook/jobs` | List webhook jobs |
| GET | `/api/webhook/jobs/{id}` | Get job status |

### GitHub Integration

1. Go to your repository Settings > Webhooks
2. Add webhook:
   - Payload URL: `https://your-server/api/webhook/github`
   - Content type: `application/json`
   - Secret: (set in your `.env`)
   - Events: Select "Pull requests"
3. The agent will automatically review PRs on open/sync/reopen

## Project Structure

```
code-review-agent/
├── agents/                    # AI review agents
│   ├── __init__.py
│   ├── base.py               # Abstract base agent
│   ├── security_agent.py     # Security vulnerability detection
│   ├── performance_agent.py  # Performance analysis
│   ├── style_agent.py        # Code quality review
│   ├── owasp_agent.py        # OWASP Top 10 security analysis
│   └── orchestrator.py       # Multi-agent coordination (parallel)
├── workflows/                 # LangGraph workflows
│   ├── __init__.py
│   └── review_workflow.py    # Review pipeline with error handling
├── tools/                     # Utility tools
│   ├── __init__.py
│   ├── code_analyzer.py      # Static code analysis (multi-language)
│   ├── github_tools.py       # GitHub API integration
│   └── rules_loader.py       # Custom YAML rules system
├── rules/                     # Custom review rules (YAML)
│   ├── security.yaml         # Security-focused rules
│   └── quality.yaml          # Code quality rules
├── api/                       # FastAPI routes
│   ├── __init__.py
│   └── webhook.py            # GitHub webhook with retry logic
├── app/                       # Frontend
│   └── streamlit_app.py      # Interactive demo UI
├── tests/                     # Test suite (148 tests)
│   ├── test_agents.py        # Agent tests
│   ├── test_api.py           # API endpoint tests
│   ├── test_code_analyzer.py # Analyzer tests
│   ├── test_owasp.py         # OWASP agent tests
│   └── test_rules_loader.py  # YAML rules tests
├── main.py                   # FastAPI application
├── requirements.txt          # Dependencies
├── Dockerfile                # Docker image
├── docker-compose.yml        # Container orchestration
├── .env.example              # Environment template
└── README.md                 # This file
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM Framework | LangChain 0.3+, LangGraph 0.2+ |
| LLM Providers | OpenAI GPT-4, Ollama (local) |
| Backend | FastAPI, uvicorn |
| Frontend | Streamlit |
| Code Analysis | tree-sitter, Pygments |
| Testing | pytest, pytest-asyncio |

## Development

### Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# Specific test file
pytest tests/test_agents.py -v
```

### Code Quality

```bash
# Format
black .

# Lint
ruff check --fix .
```

### Docker

```bash
# Build
docker build -t code-review-agent .

# Run
docker run -p 8080:8080 -e OPENAI_API_KEY=sk-... code-review-agent

# Or with docker-compose
docker-compose up
```

## Response Format

### Review Response

```json
{
  "review_id": "review_1234567890",
  "status": "completed",
  "security": {
    "findings": [
      {
        "severity": "HIGH",
        "location": "line 15",
        "title": "SQL Injection",
        "description": "User input concatenated in query",
        "recommendation": "Use parameterized queries"
      }
    ],
    "summary": "Found 1 security issue"
  },
  "performance": {
    "findings": [...],
    "complexity_analysis": {
      "time": "O(n^2)",
      "space": "O(n)"
    },
    "summary": "..."
  },
  "style": {
    "findings": [...],
    "metrics": {
      "readability": "7/10",
      "maintainability": "6/10"
    },
    "summary": "..."
  },
  "synthesis": {
    "verdict": "REQUEST_CHANGES",
    "health_score": 6,
    "critical_issues": ["SQL Injection vulnerability"],
    "important_improvements": [...],
    "minor_suggestions": [...],
    "issue_counts": {
      "critical": 0,
      "high": 1,
      "medium": 2,
      "total": 3
    },
    "summary": "Code has security concerns that need attention."
  },
  "final_report": "# Code Review Report\n...",
  "duration_ms": 2500
}
```

### Verdict Types

| Verdict | Meaning |
|---------|---------|
| `APPROVE` | Code is good to merge |
| `COMMENT` | Minor issues, okay to merge |
| `REQUEST_CHANGES` | Critical/high issues, needs fixes |

## Supported Languages

| Language | Features |
|----------|----------|
| Python | Functions, classes, imports, security patterns |
| JavaScript | Functions, classes, imports, XSS/eval detection |
| TypeScript | Functions, classes, interfaces, enums, type safety |
| Java | Functions, classes, interfaces, enums, security patterns |
| Go | Functions, interfaces, structs, security patterns |
| Rust | Functions, structs, enums, unsafe block detection |
| C / C++ | Basic analysis |
| Ruby | Basic analysis |
| PHP | Basic analysis |
| Swift | Basic analysis |
| Kotlin | Basic analysis |

## OWASP Top 10 Support

The OWASP agent provides comprehensive vulnerability detection based on the [OWASP Top 10 (2021)](https://owasp.org/www-project-top-ten/):

| Category | Description |
|----------|-------------|
| A01:2021 | Broken Access Control |
| A02:2021 | Cryptographic Failures |
| A03:2021 | Injection (SQL, Command, XSS) |
| A04:2021 | Insecure Design |
| A05:2021 | Security Misconfiguration |
| A06:2021 | Vulnerable Components |
| A07:2021 | Auth/Session Failures |
| A08:2021 | Data Integrity Failures |
| A09:2021 | Logging/Monitoring Failures |
| A10:2021 | SSRF |

### Static Analysis

OWASP static analyzer provides pattern-based detection without LLM:

```python
from agents import OWASPStaticAnalyzer

analyzer = OWASPStaticAnalyzer("python")
result = analyzer.analyze(code)
# Returns findings with OWASP category, CWE ID, severity
```

## Custom YAML Rules

Define your own code review rules in YAML format:

```yaml
# rules/my-rules.yaml
name: my-custom-rules
version: "1.0.0"
description: My organization's rules

rules:
  - id: CUSTOM-001
    name: AWS Key Detection
    pattern: 'AKIA[0-9A-Z]{16}'
    severity: critical
    languages: [all]
    owasp_id: "A02:2021"
    cwe_id: "CWE-798"
    recommendation: Use environment variables

  - id: CUSTOM-002
    name: Debug Print
    pattern: 'print\s*\('
    severity: info
    languages: [python]
    category: code-quality
```

### Using Custom Rules

```python
from tools import RulesLoader, CustomRulesAnalyzer

# Load rules
loader = RulesLoader()
ruleset = loader.load_from_file("rules/my-rules.yaml")

# Analyze code
analyzer = CustomRulesAnalyzer([ruleset])
result = analyzer.analyze(code, "python")
```

### Rule Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique rule identifier |
| `name` | string | Yes | Human-readable name |
| `pattern` | regex | Yes | Detection pattern |
| `severity` | enum | Yes | critical/high/medium/low/info |
| `languages` | list | No | Target languages (default: all) |
| `category` | string | No | Rule category |
| `owasp_id` | string | No | OWASP category (e.g., "A03:2021") |
| `cwe_id` | string | No | CWE identifier (e.g., "CWE-89") |
| `recommendation` | string | No | Fix suggestion |
| `enabled` | bool | No | Enable/disable (default: true) |
| `tags` | list | No | Searchable tags |

## License

MIT License
