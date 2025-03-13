"""
Example script demonstrating the Agent project.

This script shows how to use the AgentExecutor, memory, and tools.
"""
import asyncio
import os
from typing import Dict, Any, Optional
import logging
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(".."))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseLanguageModel

from agent_project.executor import AgentExecutor
from agent_project.config import load_config, setup_logging
from agent_project.tools import get_all_tools


async def process_token(token: Any) -> None:
    """
    Process streaming tokens.
    
    Args:
        token (Any): Token to process
    """
    try:
        if token is None:
            # Skip None tokens
            return
            
        if hasattr(token, "content"):
            # AIMessageChunk content
            if token.content:
                print(token.content, end="", flush=True)
        elif isinstance(token, str):
            # Plain string
            print(token, end="", flush=True)
        else:
            # Any other type - print its type for debugging
            token_type = type(token).__name__
            print(f"[{token_type}]", end="", flush=True)
    except Exception as e:
        print(f"\nError processing token: {str(e)}")


async def process_streaming_queue(queue) -> None:
    """
    Process all tokens from the streaming queue.
    
    Args:
        queue: The queue containing streaming tokens
    """
    try:
        token_count = 0
        token_start_time = asyncio.get_event_loop().time()
        max_wait_time = 30.0  # Maximum time to wait for new tokens in seconds
        last_token_time = token_start_time
        
        print("Processing streaming response...", flush=True)
        
        while True:
            # Check if we've been waiting too long for a new token
            current_time = asyncio.get_event_loop().time()
            if current_time - last_token_time > max_wait_time:
                print("\n[Token stream timeout]")
                break
                
            try:
                # Try to get a token with a short timeout
                token = await asyncio.wait_for(queue.get(), timeout=1.0)
                last_token_time = asyncio.get_event_loop().time()  # Update the last token time
                token_count += 1
                
                if token == "<<DONE>>":
                    print()  # Add final newline
                    break
                    
                elif token == "<<STEP_END>>":
                    print("\n")
                    
                else:
                    await process_token(token)
                    
            except asyncio.TimeoutError:
                # Just continue looping, the wait time check will handle actual timeouts
                pass
                
        token_end_time = asyncio.get_event_loop().time()
        processing_time = token_end_time - token_start_time
        print(f"\nProcessed {token_count} tokens in {processing_time:.2f} seconds")
            
    except Exception as e:
        print(f"\nError in stream processing: {str(e)}")


async def run_streaming_example(
    agent: AgentExecutor,
    query: str,
    logger: logging.Logger
) -> None:
    """
    Run a streaming example.
    
    Args:
        agent (AgentExecutor): Agent executor instance
        query (str): Query to ask the agent
        logger (logging.Logger): Logger instance
    """
    print(f"\n🔄 Running streaming example with query: '{query}'")
    logger.info(f"Running streaming example with query: {query}")
    
    # Get the streaming queue
    queue = agent.stream(query)
    
    # Process streaming tokens
    print("\nAnswer: ", end="", flush=True)
    start_time = asyncio.get_event_loop().time()
    await process_streaming_queue(queue)
    end_time = asyncio.get_event_loop().time()
    
    logger.info(f"Query processing took {end_time - start_time:.2f} seconds")
    print("\n✅ Streaming complete\n")


async def main() -> None:
    """Run the example script."""
    # Set up logging
    logger = setup_logging("INFO")
    logger.info("Starting example script")
    
    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration: {config}")
    
    # Check for Google API key
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        logger.error("Google API key not found in environment variables")
        print("Error: Google API key is required.")
        print("Please set up your .env file with your API keys as described in the README.")
        print("Example .env file format:")
        print("GOOGLE_API_KEY=your_google_api_key_here")
        print("SERPAPI_KEY=your_serpapi_key_here")
        return
    
    # Configure the language model
    llm_config = config["llm"]
    llm = ChatGoogleGenerativeAI(
        model=llm_config["model"],
        temperature=llm_config["temperature"],
        max_tokens=llm_config["max_tokens"],
        google_api_key=google_api_key,
        streaming=True
    )
    
    # Get tools based on configuration
    tools = get_all_tools()
    
    # Configure memory
    memory_config = config["memory"]
    memory_type = memory_config["type"]
    
    # Create the agent executor
    agent = AgentExecutor(
        llm=llm,
        system_prompt="""You are a helpful AI assistant with access to various tools that allow you to perform tasks that would normally be outside your capabilities.

Your job is to answer the user's questions as accurately as possible using the tools provided to you.

IMPORTANT INSTRUCTIONS FOR TOOL USAGE:
1. For questions about the current date or time, ALWAYS use the 'current_datetime' tool, not your internal knowledge.
2. For questions about weather, recent events, or any real-world information that might be changing, ALWAYS use the 'serpapi_search' tool to search for up-to-date information.
3. For calculations, ALWAYS use the 'calculator' tool rather than calculating internally.
4. You have access to web search through the 'serpapi_search' tool - use it whenever you need to find information that might not be in your training data or could be outdated.
5. Only provide a final answer after you've gathered all necessary information using the appropriate tools.

TOOL USAGE EXAMPLES:

Example 1: Getting current date/time
User: What's today's date and time?
Thought: I need to use the current_datetime tool to get the current date and time.
Action: current_datetime
Action Input: {"format": "%Y-%m-%d %H:%M:%S", "timezone": "local"}
[Tool Response: 2023-03-15 14:30:45]
Final Answer: The current date and time is March 15, 2023, 2:30:45 PM local time.

Example 2: Searching for weather information
User: What's the weather like in New York?
Thought: I need to search for current weather information in New York.
Action: serpapi_search
Action Input: {"query": "current weather in New York", "num_results": 3}
[Tool Response: Search results about weather in New York]
Final Answer: According to current information, the weather in New York is...

Example 3: Performing a calculation
User: What is the square root of 144 plus 25?
Thought: I need to calculate the square root of 144 and then add 25.
Action: calculator
Action Input: {"expression": "sqrt(144) + 25"}
[Tool Response: Result: 37.0]
Final Answer: The square root of 144 plus 25 is 37.

Remember: Do not try to answer questions about current time, date, weather, or recent events from your internal knowledge - ALWAYS use the provided tools instead.""",
        tools=tools,
        memory_type=memory_type,
        memory_window_k=memory_config.get("k", 5),
        memory_token_limit=memory_config.get("max_token_limit", 4000),
        log_level=config["logging"]["level"]
    )
    
    # Example 1: Non-streaming basic query
    query1 = "What's 25 squared plus 10?"
    logger.info(f"Running example 1 with query: {query1}")
    answer1 = agent.run(query1)
    print(f"\n❓ Query: {query1}")
    print(f"🤖 Answer: {answer1}")
    
    # Example 2: Streaming query with tool usage
    query2 = "What's the current date and time?"
    logger.info(f"Running example 2 with query: {query2}")
    await run_streaming_example(agent, query2, logger)
    
    # Example 3: Follow-up question demonstrating memory
    query3 = "What time would it be 3 hours from now?"
    logger.info(f"Running example 3 with query: {query3}")
    await run_streaming_example(agent, query3, logger)
    
    logger.info("Example script completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
