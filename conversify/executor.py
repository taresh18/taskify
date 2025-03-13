import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Union, Callable
import sys

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate

from conversify.config import setup_logging, load_config
from conversify.memory import (
    ConversationalBufferMemory,
    ConversationalBufferWindowMemory,
    ConversationalSummaryMemory,
)
from conversify.tools import get_all_tools
from conversify.streaming import QueueCallbackHandler

# Load configuration
config = load_config()
agent_config = config.get("agent", {})
memory_config = config.get("memory", {})


class AgentExecutor:
    """
    Executor for running an agent with tools.
    
    This class handles the execution of an agent, including tool selection,
    tool execution, memory management, and streaming capabilities.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        system_prompt: str,
        tools: Optional[List[Callable]] = None,
        **kwargs
    ):
        """
        Initialize the agent executor.
        
        Args:
            llm (BaseLanguageModel): The language model to use
            system_prompt (str): The system prompt for the agent
            tools (Optional[List[Callable]]): The tools available to the agent
            **kwargs: Additional keyword arguments
        """
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or get_all_tools()
        
        # Use provided values or get from config
        self.max_iterations = agent_config.get("max_iterations", 10)
        self.async_mode = agent_config.get("async_mode", True)
        self.logger = setup_logging()
        
        # Get memory configuration
        memory_type = memory_config.get("type", "buffer")
        memory_window_k = memory_config.get("window_k", 5)
        memory_summary_k = memory_config.get("summary_k", 5)
        
        # Initialize memory based on type
        if memory_type == "window":
            self.memory = ConversationalBufferWindowMemory(k=memory_window_k)
        elif memory_type == "summary":
            self.memory = ConversationalSummaryMemory(
                llm=llm,
                k=memory_summary_k
            )
        else:  
            self.memory = ConversationalBufferMemory()
        
        # Set up tool configurations
        self.tool_names = [getattr(tool, "name", tool.__class__.__name__) for tool in self.tools]
        self.tool_descriptions = {
            name: getattr(tool, "description", "") for name, tool in zip(self.tool_names, self.tools)
        }
        self.tool_schemas = {
            name: json.dumps(getattr(tool, "args_schema", {}).schema()) 
            for name, tool in zip(self.tool_names, self.tools) 
            if hasattr(tool, "args_schema")
        }
        
        # Initialize callback handlers
        self.callback_handlers = kwargs.get("callbacks", [])
        
        self.logger.info(f"Initialized AgentExecutor with {len(self.tools)} tools and {memory_type} memory")
    
    def _parse_tool_input(self, agent_action: AgentAction) -> Dict[str, Any]:
        """
        Parse the input for a tool from the agent action.
        
        Args:
            agent_action (AgentAction): The agent action
        
        Returns:
            Dict[str, Any]: The parsed tool input
        """
        tool_input = agent_action.tool_input
        self.logger.debug(f"Parsing tool input: {tool_input}")
        
        if isinstance(tool_input, dict):
            self.logger.debug("Tool input is already a dictionary")
            return tool_input
        elif isinstance(tool_input, str):
            # Clean up the input string in case it has markdown code blocks or other artifacts
            clean_input = tool_input.strip()
            
            # Remove markdown code blocks if present
            if clean_input.startswith('```') and '```' in clean_input[3:]:
                self.logger.debug("Removing markdown code blocks from tool input")
                parts = clean_input.split('```')
                # Take the content between first pair of ```
                clean_input = parts[1].strip()
            
            # If input contains multiple lines, try to find a JSON object
            if '\n' in clean_input:
                self.logger.debug("Input contains multiple lines, trying to extract JSON")
                # Look for lines that look like valid JSON
                for line in clean_input.split('\n'):
                    line = line.strip()
                    if line and (line.startswith('{') and line.endswith('}')):
                        self.logger.debug(f"Found potential JSON line: {line}")
                        try:
                            return json.loads(line)
                        except json.JSONDecodeError:
                            # Continue to next line if this one isn't valid JSON
                            continue
            
            # Try to parse as JSON
            if (clean_input.startswith('{') and clean_input.endswith('}')) or \
               (clean_input.startswith('[') and clean_input.endswith(']')):
                try:
                    self.logger.debug(f"Attempting to parse as JSON: {clean_input}")
                    return json.loads(clean_input)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse tool input as JSON: {str(e)}")
                    
                    # If we have a specific format like {"expression": "..."}, try to extract it manually
                    if "expression" in clean_input:
                        match = re.search(r'"expression"\s*:\s*"([^"]+)"', clean_input)
                        if match:
                            expression = match.group(1)
                            self.logger.debug(f"Manually extracted expression: {expression}")
                            return {"expression": expression}
            
            # For calculator tool, try to extract the expression from common formats
            if agent_action.tool == "calculator" and "expression" in clean_input:
                self.logger.debug("Trying to extract calculator expression")
                # Look for patterns like {"expression": "..."} or expression: ...
                match = re.search(r'(?:"expression"|expression)\s*:?\s*"?([^",}]+)"?', clean_input)
                if match:
                    expression = match.group(1).strip()
                    self.logger.debug(f"Extracted calculator expression: {expression}")
                    return {"expression": expression}
            
            # Return as is if it's not valid JSON
            return {"input": tool_input}
        else:
            self.logger.debug(f"Tool input is of type {type(tool_input)}, converting to string")
            return {"input": str(tool_input)}
    
    def _get_prompt(self) -> ChatPromptTemplate:
        """
        Get the prompt for the agent.
        
        Returns:
            ChatPromptTemplate: The chat prompt template
        """
        # Prepare tool descriptions
        tool_descriptions = []
        for name in self.tool_names:
            desc = self.tool_descriptions.get(name, "")
            schema = self.tool_schemas.get(name, "")
            tool_descriptions.append(f"Tool: {name}\nDescription: {desc}\nSchema: {schema}")
        
        tools_str = "\n\n".join(tool_descriptions)
        
        # Create a system message without template variables
        system_message = SystemMessage(content=f"{self.system_prompt}\n\n"
            "You have access to the following tools:\n"
            f"{tools_str}\n\n"
            "To use a tool, please use the following format:\n"
            "```\n"
            "Thought: I need to use a tool\n"
            "Action: tool_name\n"
            "Action Input: {\"param\": \"value\"}\n"
            "```\n\n"
            "Once you have the final answer, respond with:\n"
            "```\n"
            "Thought: I have the final answer\n"
            "Final Answer: the final answer\n"
            "```\n")
        
        return ChatPromptTemplate.from_messages([system_message])
    
    async def _arun_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Run a tool asynchronously.
        
        Args:
            tool_name (str): The name of the tool to run
            tool_input (Dict[str, Any]): The input for the tool
        
        Returns:
            str: The tool output
        """
        self.logger.info(f"Running tool {tool_name} with input {tool_input}")
        
        tool = next((t for t in self.tools if getattr(t, "name", t.__class__.__name__) == tool_name), None)
        
        if not tool:
            error_msg = f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(self.tool_names)}"
            self.logger.error(error_msg)
            return error_msg
        
        self.logger.debug(f"Found tool {tool_name}, preparing to execute")
        
        # Clean up tool input if needed
        cleaned_input = tool_input.copy()
        
        # If there's a nested "input" key with JSON or code blocks, extract it
        if "input" in cleaned_input and isinstance(cleaned_input["input"], str):
            input_str = cleaned_input["input"]
            if input_str.startswith("{") and input_str.endswith("}"):
                try:
                    # Try to parse JSON from the input string
                    self.logger.debug("Attempting to parse nested JSON from input string")
                    parsed_input = json.loads(input_str)
                    cleaned_input = parsed_input
                except json.JSONDecodeError:
                    # Keep as is if it's not valid JSON
                    self.logger.debug("Failed to parse nested JSON, keeping original input")
                    pass
            
            # Remove code block markers and backticks if present
            elif "```" in input_str:
                self.logger.debug("Removing code block markers from input")
                code_content = input_str.split("```")[1] if len(input_str.split("```")) > 1 else input_str
                cleaned_input["input"] = code_content.strip()
        
        self.logger.debug(f"Cleaned tool input: {cleaned_input}")
        
        try:
            # Log tool execution start
            self.logger.debug(f"Starting execution of tool {tool_name}")
            
            # Run the tool with the provided input
            if hasattr(tool, "invoke"):
                # Handle both sync and async invoke methods
                if asyncio.iscoroutinefunction(tool.invoke):
                    self.logger.debug(f"Executing {tool_name} with async invoke method")
                    result = await tool.invoke(cleaned_input)
                else:
                    self.logger.debug(f"Executing {tool_name} with sync invoke method via executor")
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: tool.invoke(cleaned_input))
            elif asyncio.iscoroutinefunction(tool):
                self.logger.debug(f"Executing {tool_name} as async function")
                result = await tool(**cleaned_input)
            else:
                self.logger.debug(f"Executing {tool_name} as sync function via executor")
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: tool(**cleaned_input))
            
            # Log successful execution and result
            self.logger.info(f"Tool {tool_name} executed successfully")
            self.logger.debug(f"Tool {tool_name} result: {str(result)[:100]}...")
            
            return str(result)
        except Exception as e:
            error_msg = f"Error running tool {tool_name}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"Exception details:", exc_info=True)
            return error_msg
    
    def _format_intermediate_steps(self, intermediate_steps: List[tuple]) -> List[BaseMessage]:
        """
        Format intermediate steps into messages.
        
        Args:
            intermediate_steps (List[tuple]): List of (AgentAction, tool_output) tuples
        
        Returns:
            List[BaseMessage]: List of formatted messages
        """
        messages = []
        
        for agent_action, tool_output in intermediate_steps:
            # Format the tool input to ensure it's a valid JSON string
            tool_input_str = agent_action.tool_input
            if isinstance(tool_input_str, dict):
                tool_input_str = json.dumps(tool_input_str)
            
            # For Gemini, we need to combine both the agent action and tool output in a single turn
            # Instead of separate AIMessage and FunctionMessage
            self.logger.debug(f"Formatting intermediate step for tool: {agent_action.tool}")
            
            # Add a human message with the tool output
            # This is more compatible with Gemini's expected format
            messages.append(
                HumanMessage(
                    content=f"I used the tool {agent_action.tool} with input: {tool_input_str}\nThe tool returned: {str(tool_output)}"
                )
            )
        
        return messages
    
    async def _agenerate_agent_step(
        self,
        input_message: str,
        intermediate_steps: List[tuple]
    ) -> Union[AgentAction, AgentFinish]:
        """
        Generate the next agent step.
        
        Args:
            input_message (str): The input message from the user
            intermediate_steps (List[tuple]): List of (AgentAction, tool_output) tuples
        
        Returns:
            Union[AgentAction, AgentFinish]: An agent action or agent finish
        """
        # Get the prompt
        prompt_template = self._get_prompt()
        
        # Format intermediate steps
        intermediate_messages = self._format_intermediate_steps(intermediate_steps)
        
        # Get memory variables
        memory_variables = self.memory.load_memory_variables({})
        memory_messages = memory_variables.get("chat_history", [])
        
        # Log memory and intermediate state
        self.logger.debug(f"Processing with {len(memory_messages)} memory messages and {len(intermediate_steps)} intermediate steps")
        
        # Start with system message
        messages = [prompt_template.messages[0]]
        
        # Add memory messages
        if memory_messages:
            self.logger.debug(f"Including {len(memory_messages)} messages from memory")
            messages.extend(memory_messages)
        
        # Add the human message
        messages.append(HumanMessage(content=input_message))
        
        # Add intermediate messages
        messages.extend(intermediate_messages)
        
        # Log that we're about to call the LLM
        self.logger.debug(f"Calling LLM with {len(messages)} total messages")
        
        # Generate a response
        response = await self.llm.agenerate(
            [messages],
            callbacks=self.callback_handlers,
        )
        ai_message = response.generations[0][0].message
        
        # Parse the response
        content = ai_message.content
        self.logger.debug(f"Received response from LLM: {content[:100]}...")
        
        # Check for "Final Answer"
        if "Final Answer:" in content:
            # Extract the final answer
            final_answer = content.split("Final Answer:")[1].strip()
            self.logger.info(f"Agent generated final answer: {final_answer}")
            return AgentFinish(return_values={"output": final_answer}, log=content)
        
        # Parse the action and input
        try:
            # First, check if "Action:" is present but not properly followed by a tool input
            if "Action:" in content:
                # Safely extract the action name
                action_parts = content.split("Action:")
                if len(action_parts) < 2:
                    raise ValueError("Malformed Action section")
                
                action_line = action_parts[1].split("\n")[0] if "\n" in action_parts[1] else action_parts[1]
                action_match = action_line.strip()
                
                # Check if Action Input is missing
                if "Action Input:" not in content:
                    self.logger.error("Agent provided 'Action:' but did not provide 'Action Input:' - this is a formatting error")
                    self.logger.info(f"Agent attempted to use tool: {action_match} but input was missing")
                    
                    # For common tools, try to provide a default input
                    if action_match.lower() == "current_datetime":
                        self.logger.info("Using default input for current_datetime tool")
                        tool_input = {"format": "%Y-%m-%d %H:%M:%S", "timezone": "local"}
                        self.logger.info(f"Using default tool input: {tool_input}")
                        return AgentAction(
                            tool=action_match,
                            tool_input=tool_input,
                            log=content
                        )
                    elif action_match.lower() == "calculator" and "expression" in content:
                        # Try to extract an expression if it's mentioned in the content
                        expression_match = re.search(r'expression[:\s]+([^\n]+)', content, re.IGNORECASE)
                        if expression_match:
                            expression = expression_match.group(1).strip()
                            self.logger.info(f"Extracted calculator expression: {expression}")
                            tool_input = {"expression": expression}
                            return AgentAction(
                                tool=action_match,
                                tool_input=tool_input,
                                log=content
                            )
                    elif action_match.lower() == "serpapi_search" and "query" in content:
                        # Try to extract a query if it's mentioned in the content
                        query_match = re.search(r'query[:\s]+([^\n]+)', content, re.IGNORECASE)
                        if query_match:
                            query = query_match.group(1).strip()
                            self.logger.info(f"Extracted search query: {query}")
                            tool_input = {"query": query}
                            return AgentAction(
                                tool=action_match,
                                tool_input=tool_input,
                                log=content
                            )
                            
                    # If we couldn't provide a default input, raise an error
                    raise ValueError(f"Missing Action Input for tool {action_match}")
                
                # Normal parsing - safely extract the action input
                input_parts = content.split("Action Input:")
                if len(input_parts) < 2:
                    raise ValueError("Malformed Action Input section")
                
                action_input_match = input_parts[1].strip()
                
                # Handle JSON format
                try:
                    tool_input = json.loads(action_input_match)
                    self.logger.debug(f"Successfully parsed tool input as JSON: {tool_input}")
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse tool input as JSON, using raw string: {str(e)}")
                    tool_input = action_input_match
                
                self.logger.info(f"Agent decided to use tool: {action_match} with input: {tool_input}")
                return AgentAction(
                    tool=action_match,
                    tool_input=tool_input,
                    log=content
                )
            else:
                # No Action directive found, treat as a final answer
                self.logger.info("No Action directive found, treating as final answer")
                return AgentFinish(return_values={"output": content}, log=content)
                
        except Exception as e:
            self.logger.error(f"Error parsing agent response: {str(e)}")
            self.logger.error("Full response content that caused the error:")
            self.logger.error(content)
            
            # Attempt to salvage meaningful content for the user
            clean_content = content
            
            # Try to remove any "Thought:" sections which aren't meant for the user
            if "Thought:" in clean_content:
                try:
                    thought_parts = clean_content.split("Thought:")
                    # Keep only the parts that might be intended as responses
                    relevant_parts = []
                    for part in thought_parts:
                        # Skip parts that contain Action: or Action Input:
                        if "Action:" not in part and "Action Input:" not in part:
                            # Clean up the part
                            clean_part = re.sub(r'^\s*I need to.*?\.', '', part)
                            if clean_part.strip():
                                relevant_parts.append(clean_part.strip())
                    
                    if relevant_parts:
                        clean_content = " ".join(relevant_parts)
                except Exception as inner_e:
                    self.logger.error(f"Error cleaning up response: {str(inner_e)}")
            
            # If response looks like it could be a direct answer, present it to the user
            return AgentFinish(return_values={"output": clean_content}, log=content)
    
    async def arun(self, input_message: str) -> str:
        """
        Run the agent asynchronously.
        
        Args:
            input_message (str): The input message from the user
        
        Returns:
            str: The output from the agent
        """
        self.logger.info(f"Running agent with input: {input_message}")
        
        # Add the user message to memory
        self.memory.add_user_message(input_message)
        
        # Track intermediate steps
        intermediate_steps = []
        iterations = 0
        
        # Run the agent loop
        while iterations < self.max_iterations:
            agent_step = await self._agenerate_agent_step(
                input_message, intermediate_steps
            )
            
            if isinstance(agent_step, AgentFinish):
                self.logger.info(f"Agent finished: {agent_step.return_values['output']}")
                # Add the AI message to memory
                self.memory.add_ai_message(agent_step.return_values["output"])
                return agent_step.return_values["output"]
            
            # Execute tool
            tool_name = agent_step.tool
            tool_input = self._parse_tool_input(agent_step)
            
            tool_output = await self._arun_tool(tool_name, tool_input)
            
            # Record the step
            intermediate_steps.append((agent_step, tool_output))
            
            iterations += 1
        
        # If reached max iterations
        self.logger.warning(f"Agent exceeded maximum iterations ({self.max_iterations})")
        output = f"I apologize, but I've exceeded the maximum number of iterations ({self.max_iterations}). Here's what I've learned so far: "
        
        # Add a summary of what happened
        for agent_action, tool_output in intermediate_steps[-3:]:
            output += f"\n- Used {agent_action.tool} and found: {tool_output[:100]}..."
        
        # Add the AI message to memory
        self.memory.add_ai_message(output)
        
        return output
    
    def run(self, input_message: str) -> str:
        """
        Run the agent synchronously.
        
        Args:
            input_message (str): The input message from the user
        
        Returns:
            str: The output from the agent
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.arun(input_message))
    
    async def astream(
        self,
        input_message: str,
        queue: asyncio.Queue
    ) -> None:
        """
        Stream the agent's response asynchronously.
        
        Args:
            input_message (str): The input message from the user
            queue (asyncio.Queue): Queue to put the streaming tokens into
        """
        self.logger.info(f"Streaming agent with input: {input_message}")
        
        # Add the user message to memory
        self.memory.add_user_message(input_message)
        
        # Track intermediate steps
        intermediate_steps = []
        iterations = 0
        
        # Create a queue for streaming the final output
        final_queue = asyncio.Queue()
        
        # Create a callback handler for streaming
        streaming_handler = QueueCallbackHandler(final_queue)
        
        # Add the streaming handler to the callbacks
        callbacks = self.callback_handlers + [streaming_handler]
        self.callback_handlers = callbacks
        
        try:
            # Run the agent loop
            while iterations < self.max_iterations:
                self.logger.info(f"Starting iteration {iterations+1}/{self.max_iterations}")
                
                try:
                    agent_step = await self._agenerate_agent_step(
                        input_message, intermediate_steps
                    )
                except Exception as e:
                    # Handle Gemini API errors
                    error_msg = str(e)
                    self.logger.error(f"Error in agent step generation: {error_msg}")
                    
                    # If it's a function call/response format error from Gemini
                    if "function response parts" in error_msg and "function call" in error_msg:
                        self.logger.warning("Detected Gemini API function call format error, using fallback approach")
                        
                        # Clear any problematic intermediate steps that may be causing format issues
                        if intermediate_steps:
                            # If we have too many intermediate steps, keep only the most recent one
                            if len(intermediate_steps) > 1:
                                self.logger.info("Simplifying intermediate steps to avoid format errors")
                                intermediate_steps = [intermediate_steps[-1]]
                            else:
                                # If only one step is causing an issue, provide a summarized version
                                last_action, last_output = intermediate_steps[0]
                                self.logger.info(f"Creating simplified representation of tool usage: {last_action.tool}")
                                
                                # Convert the intermediate steps to a simple prompt addition
                                await queue.put(f"\nI used the tool {last_action.tool} and found: {str(last_output)[:100]}...\n")
                                await queue.put("\nI'll now provide the final answer based on this information:\n")
                                await queue.put("<<DONE>>")
                                return
                    
                    # For other errors, provide an error message and end the stream
                    await queue.put(f"\nAn error occurred: {error_msg}")
                    await queue.put("<<DONE>>")
                    return
                
                if isinstance(agent_step, AgentFinish):
                    self.logger.info(f"Agent finished: {agent_step.return_values['output']}")
                    # Add the AI message to memory
                    self.memory.add_ai_message(agent_step.return_values["output"])
                    
                    # Stream the final answer
                    content_streamed = False
                    streaming_timeout = 30.0  # 30 seconds timeout for streaming
                    
                    try:
                        # Set a timeout for the streaming process
                        streaming_start = asyncio.get_event_loop().time()
                        self.logger.debug("Starting to stream final answer tokens")
                        
                        # Use streaming handler as an async iterator
                        tokens_processed = 0
                        async for token in streaming_handler:
                            # Transfer tokens to the output queue
                            await queue.put(token)
                            content_streamed = True
                            tokens_processed += 1
                            
                            # Log token progress periodically
                            if tokens_processed % 10 == 0:
                                self.logger.debug(f"Streamed {tokens_processed} tokens so far")
                            
                            # Check if we've exceeded our timeout
                            if asyncio.get_event_loop().time() - streaming_start > streaming_timeout:
                                self.logger.warning("Streaming timeout exceeded, ending stream")
                                break
                                
                        self.logger.debug(f"Finished streaming with {tokens_processed} total tokens")
                        
                    except Exception as e:
                        self.logger.error(f"Error during token streaming: {str(e)}")
                    
                    # Ensure the content gets streamed if streaming didn't work
                    if not content_streamed:
                        self.logger.warning("No content streamed, sending final answer directly")
                        
                        # Get the final answer
                        final_answer = agent_step.return_values["output"]
                        self.logger.debug(f"Sending final answer directly to queue: {final_answer[:50]}...")
                        
                        # Send the final answer in smaller chunks to simulate streaming
                        chunk_size = 10  # Characters per chunk
                        for i in range(0, len(final_answer), chunk_size):
                            chunk = final_answer[i:i+chunk_size]
                            await queue.put(chunk)
                            await asyncio.sleep(0.01)  # Small delay between chunks
                    
                    # Signal completion
                    await queue.put("<<DONE>>")
                    self.logger.info("Streaming completed")
                    return
                
                # Execute tool
                tool_name = agent_step.tool
                tool_input = self._parse_tool_input(agent_step)
                
                self.logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
                tool_output = await self._arun_tool(tool_name, tool_input)
                self.logger.info(f"Tool output: {tool_output[:100]}...")
                
                # Record the step
                intermediate_steps.append((agent_step, tool_output))
                
                iterations += 1
                self.logger.debug(f"Completed iteration {iterations}")
            
            # If reached max iterations
            self.logger.warning(f"Agent exceeded maximum iterations ({self.max_iterations})")
            output = f"I apologize, but I've exceeded the maximum number of iterations ({self.max_iterations}). Here's what I've learned so far: "
            
            # Add a summary of what happened
            for agent_action, tool_output in intermediate_steps[-3:]:
                output += f"\n- Used {agent_action.tool} and found: {tool_output[:100]}..."
            
            # Add the AI message to memory
            self.memory.add_ai_message(output)
            
            # Put the output in the queue
            await queue.put(output)
            await queue.put("<<DONE>>")
            self.logger.info("Streaming completed with max iterations")
            
        except Exception as e:
            # Handle any exceptions during streaming
            error_msg = f"Error during agent streaming: {str(e)}"
            self.logger.error(error_msg)
            await queue.put(f"\nAn error occurred: {str(e)}")
            await queue.put("<<DONE>>")
            return
    
    def stream(self, input_message: str) -> asyncio.Queue:
        """
        Stream the agent's response synchronously.
        
        Args:
            input_message (str): The input message from the user
        
        Returns:
            asyncio.Queue: Queue with the streaming tokens
        """
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        loop.create_task(self.astream(input_message, queue))
        return queue
