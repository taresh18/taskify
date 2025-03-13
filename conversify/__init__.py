"""
Agent Project - A flexible agent framework built with LangChain.

This package provides tools for creating autonomous agents with memory and tools.
"""

__version__ = "0.1.0"

from dotenv import load_dotenv
from agent_project.config import load_config, setup_logging
from agent_project.executor import AgentExecutor
from agent_project.memory import (
    ConversationalBufferMemory, 
    ConversationalBufferWindowMemory,
    ConversationalSummaryMemory
)
from agent_project.streaming import QueueCallbackHandler
from agent_project.tools import get_all_tools

__all__ = [
    "load_dotenv",
    "load_config",
    "setup_logging",
    "AgentExecutor",
    "ConversationalBufferMemory",
    "ConversationalBufferWindowMemory",
    "ConversationalSummaryMemory",
    "QueueCallbackHandler",
    "get_all_tools",   
] 