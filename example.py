import asyncio
import json

from conversify.config import load_config, setup_logging
from conversify.executor import AgentExecutor

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
        max_wait_time = streaming_config.get("max_wait_time", 5.0)
        wait_time = streaming_config.get("wait_time", 0.5)
        
        logger.debug("Starting to process streaming tokens")
        last_token_time = token_start_time
        content_received = False
        tool_output_displayed = False
        last_displayed_tool = None
        
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
                    tool_output_displayed = False  # Reset for next step
                    
                else:
                    # Process the token based on its type
                    if isinstance(token, str):
                        # Plain string token
                        content_received = True
                        print(token, end="", flush=True)
                    elif hasattr(token, "content") and token.content:
                        # Message with content attribute
                        content_received = True
                        print(token.content, end="", flush=True)
                    elif hasattr(token, "message") and hasattr(token.message, "content"):
                        # For ChatGenerationChunk objects
                        content_received = True
                        if token.message.content:
                            print(token.message.content, end="", flush=True)
                            
                        # Check for tool calls in message
                        if hasattr(token.message, "tool_calls") and token.message.tool_calls:
                            tool_call = token.message.tool_calls[0]
                            tool_name = tool_call.get("name", "unknown_tool")
                            
                            # Only display tool call once per tool
                            if last_displayed_tool != tool_name:
                                print(f"\n[Using tool: {tool_name}]", end="", flush=True)
                                
                                # For calculator, also show the expression
                                if tool_name == "calculator" and "expression" in tool_call.get("args", {}):
                                    expression = tool_call["args"]["expression"]
                                    print(f" Calculating: {expression}", end="", flush=True)
                                    
                                last_displayed_tool = tool_name
                            
                        # Check for function calls in additional_kwargs
                        elif hasattr(token.message, "additional_kwargs") and token.message.additional_kwargs.get("function_call"):
                            function_call = token.message.additional_kwargs.get("function_call")
                            tool_name = function_call.get("name", "unknown_function")
                            
                            # Only display tool call once per tool
                            if last_displayed_tool != tool_name:
                                print(f"\n[Using tool: {tool_name}]", end="", flush=True)
                                
                                # For calculator, also show the expression
                                if tool_name == "calculator" and function_call.get("arguments"):
                                    try:
                                        args = json.loads(function_call.get("arguments", "{}"))
                                        if "expression" in args:
                                            print(f" Calculating: {args['expression']}", end="", flush=True)
                                    except:
                                        pass
                                    
                                last_displayed_tool = tool_name
                    elif hasattr(token, "additional_kwargs") and token.additional_kwargs.get("function_call"):
                        # Handle function call
                        function_call = token.additional_kwargs.get("function_call")
                        tool_name = function_call.get("name", "unknown_function")
                        content_received = True
                        
                        # Only display tool call once per tool
                        if last_displayed_tool != tool_name:
                            print(f"\n[Using tool: {tool_name}]", end="", flush=True)
                            last_displayed_tool = tool_name
                    elif hasattr(token, "tool_calls") and token.tool_calls:
                        # Handle tool calls
                        content_received = True
                        tool_name = token.tool_calls[0].get("name", "unknown_tool")
                        
                        # Only display tool call once per tool
                        if last_displayed_tool != tool_name:
                            print(f"\n[Using tool: {tool_name}]", end="", flush=True)
                            last_displayed_tool = tool_name
                    # Special handling for ToolMessage content
                    elif hasattr(token, "type") and token.type == "tool":
                        content_received = True
                        if not tool_output_displayed:
                            print(f"\n[Tool Result: {token.content}]", end="\n", flush=True)
                            tool_output_displayed = True
                    else:
                        # Unknown token type - log it for debugging
                        logger.debug(f"Unknown token type: {type(token)}")
                        # Try to extract any useful information from the token
                        if hasattr(token, "__dict__"):
                            logger.debug(f"Token attributes: {str(token.__dict__)[:200]}")
                
                # Reset wait time after successful token processing
                wait_time = streaming_config.get("wait_time", 0.5)
                
                # Mark queue task as done
                queue.task_done()
                
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
    
    # Reset variables for each query
    global last_processed_token
    last_displayed_tool = None
    
    # Get the streaming queue
    print("\nResponse: ", end="", flush=True)
    
    # Create a fresh queue for each query
    queue = agent.stream(query)
    
    # Process streaming tokens
    await process_streaming_tokens(queue)
    
    # Add a clear divider after the response
    print("\n" + "-" * 80)
    
    # Return True to indicate successful completion
    return True

async def main():
    """Run a demonstration of the Conversify agent."""
    print("\n=== Conversify Agent Demonstration ===\n")
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()

    logger.info("Creating agent executor...")
    # Create agent executor
    agent = AgentExecutor()
    
    try:
        # Define test queries to demonstrate different capabilities
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
        
        print("\n=== Demonstration Completed Successfully ===")
        
    except KeyboardInterrupt:
        print("\n\nDemonstration stopped by user.")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error("Exception details:", exc_info=True)
        print(f"\nError: {str(e)}")
    

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 