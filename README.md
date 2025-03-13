# Agent Project

A flexible framework for building autonomous agents with LangChain.

## Project Structure

The project has a simplified structure with all core functionality in a single directory:

```
agent_project/
├── __init__.py        # Package initialization and exports
├── config.py          # Configuration loading and logging setup
├── executor.py        # AgentExecutor for running agents
├── memory.py          # Various memory implementations
├── requirements.txt   # Project dependencies
├── streaming.py       # Streaming functionality for async responses
├── tools.py           # Tool implementations for agent use
└── example.py         # Example script showing basic usage
```

Logs are stored in the `logs/` directory.

## Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables by creating a `.env` file in the project root:

```bash
# Create .env file from the template
cp .env.example .env

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

> Note: The `.env` file is included in `.gitignore` to prevent accidentally committing your API keys to version control.

## Running Examples

The project includes several example scripts to demonstrate its capabilities:

- `quickstart.py` - Simple script to verify your .env setup and run a basic query
- `debug_example.py` - Tests the agent with basic tools
- `streaming_debug.py` - Tests streaming capabilities with different memory types
- `agent_project/example.py` - Simple example of basic usage

Try the quickstart script first to verify your environment setup:

```bash
python quickstart.py
```

Then run any of the other example scripts:

```bash
python debug_example.py
```

## Usage

Here's a basic example of how to use the Agent:

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from agent_project import AgentExecutor, get_all_tools

# Create a language model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    google_api_key="your_api_key"
)

# Create an agent with tools
agent = AgentExecutor(
    llm=llm,
    system_prompt="You are a helpful AI assistant.",
    tools=get_all_tools(),
    memory_type="buffer"
)

# Run the agent
answer = agent.run("What is 42 squared?")
print(answer)
```

## Features

- Multiple memory implementations (Buffer, Window, Summary)
- Streaming responses for real-time output
- Various tools (calculator, datetime, weather, web search, etc.)
- Configurable via YAML
- Comprehensive logging 