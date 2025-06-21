# A2A-MCP: Agent-to-Agent Model Control Protocol

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](License.md)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/managed%20with-uv-purple.svg)](https://github.com/astral-sh/uv)

A sophisticated multi-agent system that combines the Model Context Protocol (MCP) with Agent-to-Agent (A2A) communication for intelligent text processing, web interaction, and automated workflows.

## 🌟 Features

### 🤖 Intelligent Multi-Agent System
- **Smart Agent Coordination**: Automatic detection of user intent and routing to appropriate agents
- **A2A Communication**: Direct agent-to-agent communication for complex workflows
- **MCP Integration**: Standards-compliant Model Context Protocol implementation

### 🔧 Core Services (MCP Tools)
- **Web Search & Weather**: DuckDuckGo integration with weather-specific queries
- **Website Content Extraction**: Headless browser-based text extraction
- **Time & Date Services**: NTP-synchronized accurate time information
- **File Processing**: Multi-format document conversion to PDF
- **Data Anonymization**: Intelligent PII detection and removal

### 🎯 Specialized Agents (A2A)
- **Text Optimizer**: Professional email generation and tone adjustment
- **Grammar Corrector (Lektor)**: German/English grammar and spelling correction
- **Sentiment Analysis**: Emotion detection and sentiment scoring
- **Query Refactoring**: LLM-optimized query reformulation
- **User Interface Agent**: Intelligent request interpretation and routing
- **Prompt Engineer**: Advanced prompt optimization and intent detection with A2A orchestration

### 🌐 User Interfaces
- **Gradio Web Interface**: User-friendly browser-based interaction
- **RESTful API**: Programmatic access to all services
- **CLI Integration**: Command-line tool compatibility

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    A2A-MCP System                          │
├─────────────────────────────────────────────────────────────┤
│  Gradio Interface (Port 7860)                              │
│  ├── File Upload & Processing                              │
│  ├── Natural Language Input                                │
│  └── Tonality Selection                                    │
├─────────────────────────────────────────────────────────────┤
│  User Interface Agent (Intelligent Router)                 │
│  ├── Intent Detection                                      │
│  ├── Agent Selection                                       │
│  └── Response Coordination                                 │
├─────────────────────────────────────────────────────────────┤
│  MCP Server (Port 8000)                     A2A Registry   │
│  ├── Web Search & Weather                   ├── Optimizer  │
│  ├── Website Extraction                     ├── Lektor     │
│  ├── Time/Date Services                     ├── Sentiment  │
│  ├── File Conversion                        ├── Query Ref  │
│  └── Anonymization                          └── UI Agent   │
├─────────────────────────────────────────────────────────────┤
│  Backend Services                                          │
│  ├── Selenium WebDriver                                    │
│  ├── DuckDuckGo Search                                     │
│  ├── NTP Time Sync                                         │
│  ├── PDF Conversion                                        │
│  └── LLM Integration (Ollama)                              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) with `qwen2.5:latest` model
- Modern web browser (for Gradio interface)

### Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

1. **Install uv** (if not already installed):
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Or via pip
   pip install uv
   ```

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Agent_System
   ```

3. **Install dependencies with uv**:
   ```bash
   # Create virtual environment and install dependencies
   uv sync
   
   # Activate the environment
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

4. **Configure environment**:
   ```bash
   # Copy and edit configuration files
   cp .env.example .env
   cp .env.a2ap.example .env.a2ap              # If needed
   cp .env.mcp.example .env.mcp                # If needed
   cp .env.prompt_engineer.example .env.prompt_engineer  # If needed
   ```

5. **Start Ollama** (if not running):
   ```bash
   ollama serve
   ollama pull qwen2.5:latest
   ollama pull google/gemma3:latest  # For sentiment analysis
   ```

### Quick Launch

Start all services with the integrated launcher:

```bash
python launcher.py
```

This will automatically start:
- **MCP Server** on `http://localhost:8000`
- **Gradio Interface** on `http://localhost:7860`
- **A2A Agent Registry** (embedded)

## 🎯 Usage Examples

### Web Interface

1. **Open your browser** to `http://localhost:7860`
2. **Try these example requests**:

```
"Wie wird das Wetter morgen in Berlin?"
→ Automatic web search with weather optimization

"Korrigiere diesen Text: Das ist ein sehr schlechte Satz mit viele Fehler."
→ Grammar correction via Lektor agent

"Optimiere diesen Text für eine professionelle E-Mail: Das Produkt ist Schrott!"
→ Professional email generation via Optimizer agent

"Analysiere das Sentiment: Ich bin so glücklich über dieses großartige Produkt!"
→ Sentiment analysis with emotion detection

"Wie spät ist es jetzt?"
→ NTP-synchronized time retrieval
```

### API Usage

```python
import httpx

# Direct MCP tool call
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/mcp/call-tool",
        json={
            "name": "duckduckgo_search",
            "arguments": {"query": "weather Berlin", "max_results": 5}
        }
    )
    print(response.json())
```

### Agent Integration

```python
from agent_server.user_interface import process_input

# Intelligent request processing
result = await process_input("Mache diesen Text freundlicher: Ihre Anfrage wurde abgelehnt.")
print(result.final_result)
```

## 🔧 Configuration

### Modular Configuration System

The system uses a modular configuration approach with separate files for different components:

- **`.env`** - Main system configuration (server, models, debugging)
- **`.env.a2ap`** - Agent-to-Agent Protocol settings (communication, discovery, delegation)
- **`.env.mcp`** - Model Context Protocol settings (context management, optimization)
- **`.env.prompt_engineer`** - Prompt Engineering agent settings (optimization strategies, templates, quality metrics)

The `core/config_manager.py` module provides centralized access to all configuration values with type-safe getters and automatic loading of all configuration files.

### Environment Variables

```bash
# LLM Configuration
BASE_URL=http://localhost:11434/v1
API_KEY=ollama
USER_INTERFACE_MODEL=qwen2.5:latest
OPTIMIZER_MODEL=qwen2.5:latest
LEKTOR_MODEL=qwen2.5:latest
SENTIMENT_MODEL=qwen2.5:latest
QUERY_REF_MODEL=qwen2.5:latest

# Server Configuration
SERVER_HOST=localhost
SERVER_PORT=8000
SERVER_SCHEME=http
GRADIO_HOST=127.0.0.1
GRADIO_PORT=7860

# Debug Options
DEBUG_AGENT_RESPONSES=false
DEBUG_A2A_CALLS=false

# Service Configuration
ANONYMIZER_USE_LLM=false
ANONYMIZER_LLM_ENDPOINT=
ANONYMIZER_LLM_API_KEY=
ANONYMIZER_LLM_MODEL=
```

### Model Requirements

Ensure these models are available in Ollama:
```bash
ollama pull qwen2.5:latest  # Primary model for all agents
# Or configure different models per agent in .env
```

## 📚 Available Tools & Agents

### MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_current_time` | NTP-synchronized UTC time | None |
| `duckduckgo_search` | Web search with weather optimization | `query`, `max_results` |
| `extract_website_text` | Extract main content from URLs | `url` |
| `anonymize_text` | Remove PII from text | `text` |
| `convert_to_pdf` | Convert files to PDF format | `input_filepath`, `output_directory` |

### A2A Agents

| Agent | Purpose | Input | Output |
|-------|---------|-------|--------|
| **Optimizer** | Professional text optimization | Raw text + tonality | Polished professional text |
| **Lektor** | Grammar & spelling correction | Text with errors | Corrected text |
| **Sentiment** | Emotion & sentiment analysis | Any text | Sentiment score + emotions |
| **Query Ref** | LLM query optimization | User query | Optimized query |
| **User Interface** | Intelligent request routing | Natural language | Coordinated response |
| **Prompt Engineer** | Advanced prompt optimization | User query | Optimized prompt + execution plan |

### Supported File Types

#### Text Processing
- `.txt`, `.md`, `.py`, `.csv`, `.log`, `.json`, `.xml`, `.html`

#### PDF Conversion
- **Text files**: All above formats
- **Images**: `.jpg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp`
- **Office docs**: `.docx`, `.xlsx`, `.pptx` (requires LibreOffice)

## 🔄 Workflows

### Intelligent Text Processing
```
User Input → Intent Detection → Agent Selection → Processing → Response
     ↓              ↓              ↓              ↓          ↓
"Fix grammar" → Text Processing → Lektor Agent → Correction → Clean Text
```

### Professional Email Generation
```
Complaint Text → Optimizer Agent → Professional Email → Lektor Check → Final Email
```

### Multi-Step Analysis
```
Raw Text → Query Refactor → Optimization → Grammar Check → Sentiment Analysis
```

## 🛠️ Development

### Project Structure

```
Agent_System/
├── core/                   # Core system components
│   └── config_manager.py   # Modular configuration manager
├── agent_server/           # A2A agents
│   ├── user_interface.py   # Main coordination agent
│   ├── optimizer.py        # Text optimization
│   ├── lektor.py          # Grammar correction
│   ├── sentiment.py       # Sentiment analysis
│   └── query_ref.py       # Query refactoring
├── mcp_services/          # MCP service implementations
│   ├── mcp_search/        # DuckDuckGo integration
│   ├── mcp_website/       # Web scraping
│   ├── mcp_time/         # NTP time services
│   ├── mcp_anonymizer/   # Data anonymization
│   └── mcp_fileconverter/ # PDF conversion
├── .env                   # Main configuration
├── .env.a2ap             # A2A Protocol configuration
├── .env.mcp              # Model Context Protocol configuration
├── .env.prompt_engineer  # Prompt Engineering configuration
├── mcp_main.py           # MCP server
├── launcher.py           # Service orchestrator
├── gradio_interface.py   # Web UI
├── a2a_server.py        # A2A registry
├── pyproject.toml        # uv project configuration
└── uploaded_files/      # File upload storage
```

### Development with uv

```bash
# Install development dependencies
uv sync --dev

# Add new dependency
uv add package-name

# Remove dependency
uv remove package-name

# Update dependencies
uv lock --upgrade

# Run specific scripts
uv run python launcher.py
uv run python mcp_main.py
```

### Adding New Agents

1. **Create agent file** in `agent_server/`:
```python
from pydantic_ai import Agent
from pydantic import BaseModel

class MyAgentResponse(BaseModel):
    result: str

async def my_agent_a2a_function(messages: list) -> MyAgentResponse:
    # Implementation
    pass
```

2. **Register in A2A server**:
```python
# In a2a_server.py
registry.register_a2a_agent("my_agent", my_agent_a2a_function)
```

3. **Add to user interface** agent routing logic

### Adding MCP Tools

1. **Implement service** in `mcp_services/`
2. **Add endpoint** in `mcp_main.py`
3. **Register tool** in MCP configuration

## 🧪 Testing

### Manual Testing
```bash
# Test individual agents
python agent_server/sentiment.py
python agent_server/optimizer.py

# Test MCP server
curl http://localhost:8000/health

# Test full workflow
python a2a_server.py
```

### Example Test Cases
```python
# Sentiment analysis
await sentiment_agent("I love this amazing product!")

# Text optimization
await optimizer_agent("Das Produkt ist Schrott!", tonality="professionell")

# Grammar correction
await lektor_agent("Das ist ein sehr schlechte Satz.")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)** - see the [License.md](License.md) file for details.

### AGPLv3 Summary

- ✅ **Commercial use** - You can use this software commercially
- ✅ **Modification** - You can modify the source code
- ✅ **Distribution** - You can distribute the software
- ✅ **Patent use** - Express grant of patent rights from contributors
- ✅ **Private use** - You can use the software privately

**Requirements:**
- 📋 **License and copyright notice** - Include the license and copyright notice
- 📋 **State changes** - Document significant changes made to the software
- 📋 **Disclose source** - Provide source code when distributing
- 📋 **Network use is distribution** - **Users interacting with the software over a network must be able to download the source code**

The AGPLv3 ensures that any network service using this code must provide the source code to its users, maintaining the open-source nature even in SaaS deployments.

## 🙏 Acknowledgments

- [Pydantic AI](https://github.com/pydantic/pydantic-ai) for the agent framework
- [Model Context Protocol](https://modelcontextprotocol.io/) for the standards
- [Gradio](https://gradio.app/) for the web interface
- [Ollama](https://ollama.ai/) for local LLM support

---

**Built with ❤️ for intelligent multi-agent workflows**
