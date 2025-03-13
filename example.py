import asyncio
import os
from getpass import getpass

from langchain_google_genai import ChatGoogleGenerativeAI

from conversify.config import load_config, setup_logging
from conversify.executor import AgentExecutor
from conversify.tools import get_all_tools

# Configure logging
logger = setup_logging()

async def process_streaming_tokens(queue):
    """Process tokens from the streaming queue."""
    try:
        token_count = 0
        token_start_time = asyncio.get_event_loop().time()
        
        # Get streaming settings from config
        config = load_config()
        streaming_config = config.get("streaming", {})
        max_wait_time = streaming_config.get("max_wait_time", 5.0)  # Increased default
        wait_time = streaming_config.get("wait_time", 0.5)  # Increased default
        
        logger.debug("Starting to process streaming tokens")
        last_token_time = token_start_time
        content_received = False
        
        while True:
            try:
                # Use a shorter timeout for queue.get() to allow for more frequent checks
                token = await asyncio.wait_for(queue.get(), timeout=wait_time)
                last_token_time = asyncio.get_event_loop().time()
                token_count += 1
                
                if token == "<<DONE>>":
                    logger.debug("Received DONE signal")
                    if not content_received:
                        print("\n[No content received]")
                    else:
                        print()  # Add newline at the end
                    break
                    
                elif token == "<<STEP_END>>":
                    logger.debug("Received STEP_END signal")
                    print("\n")
                    
                else:
                    # Process the token based on its type
                    if hasattr(token, "message") and hasattr(token.message, "content"):
                        # AIMessageChunk with content
                        content = token.message.content
                        if content:
                            content_received = True
                            print(content, end="", flush=True)
                            
                    elif hasattr(token, "content"):
                        # AIMessageChunk or similar
                        if token.content:
                            content_received = True
                            print(token.content, end="", flush=True)
                            
                    elif isinstance(token, str):
                        # Plain string token
                        content_received = True
                        print(token, end="", flush=True)
                        
                    else:
                        # Unknown token type - log it for debugging
                        logger.debug(f"Unknown token type: {type(token)}")
                
                # Reset wait time after successful token processing
                wait_time = streaming_config.get("wait_time", 0.5)
                
            except asyncio.TimeoutError:
                current_time = asyncio.get_event_loop().time()
                time_since_last = current_time - last_token_time
                
                if time_since_last > max_wait_time:
                    if not content_received:
                        logger.warning("No content received before timeout")
                        print("\n[No response received - please try again]")
                    break
                
                # Gradually increase wait time to reduce CPU usage
                wait_time = min(wait_time * 1.5, streaming_config.get("max_wait_time", 5.0))
                continue
                
            except Exception as e:
                logger.error(f"Error processing token: {str(e)}")
                if not content_received:
                    print(f"\n[Error: {str(e)}]")
                break
        
        # Log completion statistics
        token_end_time = asyncio.get_event_loop().time()
        processing_time = token_end_time - token_start_time
        logger.info(f"Processed {token_count} tokens in {processing_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in token processing loop: {str(e)}")
        print(f"\n[Error: {str(e)}]")

async def process_query(agent, query, query_num, total_queries):
    """Process a single query and handle all streaming output."""
    print(f"\n[Query {query_num}/{total_queries}]: {query}")
    logger.info(f"Running agent with query: {query}")
    
    # Get the streaming queue
    print("\nResponse: ", end="", flush=True)
    queue = agent.stream(query)
    
    # Process streaming tokens
    await process_streaming_tokens(queue)
    
    # Add a clear divider after the response
    print("\n" + "-" * 80)
    
    # Return True to indicate successful completion
    return True

async def main():
    """Run a comprehensive test of the Conversify agent."""
    print("\n=== Conversify Agent Test ===\n")
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()
    
    # Get API key
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Configure the language model
    logger.info("Initializing language model...")
    llm_config = config.get("llm", {})
    llm = ChatGoogleGenerativeAI(
        model=llm_config.get("model", "gemini-1.0-pro"),
        temperature=llm_config.get("temperature", 0.1),
        max_tokens=llm_config.get("max_tokens", 1000),
        google_api_key=google_api_key,
        streaming=True,
        verbose=True  # Add verbose mode to help with debugging
    )
    
    # Create a more concise system prompt
    system_prompt = """You are a helpful AI assistant that uses tools to provide accurate information. For current time/date, use current_datetime tool. For calculations, use calculator tool. For web searches or current info, use serpapi_search tool. Always format tool usage as:

Thought: [reasoning]
Action: [tool_name]
Action Input: {"param": "value"}

After getting tool response, provide:
Final Answer: [your answer]"""

    logger.info("Creating agent executor with all tools...")
    # Create agent executor with ALL tools
    agent = AgentExecutor(
        llm=llm,
        system_prompt=system_prompt,
        tools=get_all_tools(),
    )
    
    try:
        # Series of queries to test different tools and conversational memory
        queries = [
            # Start with a math query
            "What is the square root of 144 plus 25?",
            
            # Ask for current date/time
            "What's today's date and time?",
            
            # Ask for weather that requires web search
            "What's the weather like in New York?",
            
            # Ask for information that requires web search
            "Tell me about quantum computing",
            
            # Refer back to previous answers to test conversational memory
            "Given the information above, what's the sum of today's day of the month and the square root of 144?",
            
            # Test more conversational memory with self-reference
            "When did you tell me about the weather earlier?",
            
            # Test combining multiple pieces of information
            "Compare quantum computing with classical computing based on what you told me earlier"
        ]
        
        # Process each query in sequence
        for i, query in enumerate(queries):
            query_num = i+1
            
            # Process the current query
            success = await process_query(agent, query, query_num, len(queries))
            
            # Only prompt for continue if this isn't the last query
            if i < len(queries) - 1:
                print("\nPress Enter to continue to the next query... ", end="", flush=True)
                try:
                    user_input = input()
                    logger.debug(f"User pressed Enter to continue to query {query_num+1}")
                except Exception as e:
                    logger.error(f"Error getting user input: {str(e)}")
                    print(f"Error: {str(e)}, continuing automatically")
        
        # Check the memory state
        print("\n=== Memory State ===")
        memory_variables = agent.memory.load_memory_variables()
        memory_messages = memory_variables.get("chat_history", [])
        print(f"Number of messages in memory: {len(memory_messages)}")
        
        if len(memory_messages) > 0:
            print("\nLast message:")
            last_message = memory_messages[-1]
            preview = last_message.content[:100] + "..." if len(last_message.content) > 100 else last_message.content
            print(f"Type: {last_message.__class__.__name__}, Content: {preview}")
            
        print("\n=== Test Completed Successfully ===")
        
    except KeyboardInterrupt:
        print("\n\nTest stopped by user.")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error("Exception details:", exc_info=True)
        print(f"\nError: {str(e)}")
    

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 