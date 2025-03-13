# Conversify

A flexible framework for building autonomous conversational agents with LangChain.

## Project Structure

The project has a simplified structure with all core functionality in a single directory:

```
.
├── conversify/            # Main package directory
│   ├── __init__.py       # Package initialization and exports
│   ├── config.py         # Configuration loading and logging setup
│   ├── executor.py       # AgentExecutor for running agents
│   ├── memory.py         # Various memory implementations
│   ├── streaming.py      # Streaming functionality for async responses
│   └── tools.py          # Tool implementations for agent use
├── config.yaml           # Configuration settings
├── example.py            # Example script showing basic usage
├── requirements.txt      # Project dependencies
└── logs/                 # Directory for log files
```

## Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables by creating a `.env` file in the project root:

```bash
# Edit the .env file with your API keys
nano .env
```

4. Add your API keys to the `.env` file:

```
# Google Gemini API Key for LLM
GOOGLE_API_KEY=your_google_api_key_here

# SerpAPI Key for web search
SERPAPI_KEY=your_serpapi_key_here
```


## Configuration

The project uses `config.yaml` for configuration settings. You can customize:

- LLM settings (model, temperature, max tokens)
- Memory type and parameters
- Logging configuration
- Agent settings (max iterations, async mode)
- Tool-specific settings
- Streaming settings

See the `config.yaml` file for all available options.

## Running Examples

The project includes an example script to demonstrate its capabilities:

```bash
python example.py
```

The example script will:
1. Load the configuration from `config.yaml`
2. Initialize the language model and tools
3. Create an agent with the specified memory type
4. Run a series of test queries to demonstrate different features
5. Show the memory state at the end

## Features

- Multiple memory implementations (Buffer, Window, Summary)
- Streaming responses for real-time output
- Various tools (calculator, datetime, web search, etc.)
- Configurable via YAML
- Comprehensive logging 
- Async operation mode for improved performance 