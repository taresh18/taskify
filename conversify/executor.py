import asyncio
import json
import os
from langchain_core.runnables import ConfigurableField
from typing import List, Any, Dict, Optional, Union, Callable

from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

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
llm_config = config.get("llm", {})
google_api_key = os.environ.get("GOOGLE_API_KEY")


class AgentExecutor:
    """
    Executor for running an agent with tools.
    
    This class handles the execution of an agent, including tool selection,
    tool execution, memory management, and streaming capabilities.
    """
    
    def __init__(self):
        """
        Initialize the agent executor.
    
        """
        self.llm = ChatGoogleGenerativeAI(
            model=llm_config.get("model"),
            temperature=llm_config.get("temperature"),
            max_tokens=llm_config.get("max_tokens"),
            google_api_key=google_api_key,
            disable_streaming=False,  # Explicitly enable streaming
            verbose=True  # Add verbose mode to help with debugging
        ).configurable_fields(
            callbacks=ConfigurableField(
                id="callbacks",
                name="callbacks",
                description="A list of callbacks to use for streaming",
            )
        )

        self.system_prompt = (
            "You are a helpful AI assistant with access to various tools that allow you to perform tasks that would normally be outside your capabilities. "
            "Your job is to answer the user's questions as accurately as possible using the tools provided to you.\n"
            "After using a tool, look at the tool output and determine if you now have enough information to answer the user's question. "
            "If you have the answer after using a tool, use the final_answer tool to provide your final answer. "
            "Do not call the same tool with the same input multiple times. "
            "For calculator questions, once you get the numerical result, use the final_answer tool right away to present the result clearly.\n"
            "After using a tool the tool output will be provided back to you. When you have all the information you need, you MUST use the final_answer tool to provide a final answer to the user."
        )
        self.tools = get_all_tools()
        
        # Use provided values or get from config
        self.max_iterations = agent_config.get("max_iterations")
        self.async_mode = agent_config.get("async_mode")
        self.logger = setup_logging()
        
        # Create a tool name to tool mapping
        self.name_to_tool = {
            getattr(tool, "name", tool.__class__.__name__): tool
            for tool in self.tools
        }
        
        # Get memory configuration
        memory_type = memory_config.get("type")
        
        # Initialize memory based on type
        if memory_type == "window":
            self.memory = ConversationalBufferWindowMemory()
        elif memory_type == "summary":
            self.memory = ConversationalSummaryMemory(llm=self.llm)
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
        
        # Create the agent chain
        self._create_agent_chain()
        
        self.logger.info(f"Initialized AgentExecutor with {len(self.tools)} tools and {memory_type} memory")

    def _create_agent_chain(self):
        """Create the agent chain with tools and prompt."""
        # Prepare tool descriptions for the prompt
        tool_descriptions = []
        for name in self.tool_names:
            desc = self.tool_descriptions.get(name, "")
            # Escape JSON schema to prevent it from being interpreted as template variables
            schema_raw = self.tool_schemas.get(name, "")
            # Replace all single curly braces with double curly braces to escape them
            schema = schema_raw.replace("{", "{{").replace("}", "}}")
            
            tool_descriptions.append(f"Tool: {name}\nDescription: {desc}\nSchema: {schema}")
        
        tools_str = "\n\n".join(tool_descriptions)
        
        # Create the system message
        system_message = (
            f"{self.system_prompt}\n"
            "You have access to the following tools:\n"
            f"{tools_str}\n\n"
            "Always format tool usage as:\n\n"
            "Thought: [reasoning]\n"
            "Action: [tool_name]\n" 
            "Action Input: {{\"param\": \"value\"}}\n\n"
        )
        
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent chain with tools
        self.agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            | RunnablePassthrough.assign(
                chat_history=RunnableLambda(lambda x: self._format_chat_history())
            )
            | self.prompt
            | self.llm.bind_tools(self.tools, tool_choice="any")
        )

    def _format_chat_history(self) -> List[BaseMessage]:
        """Format the chat history from memory for the agent."""
        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        self.logger.debug(f"Formatting chat history with {len(chat_history)} messages")
        return chat_history

    async def _execute_tool(self, tool_call) -> ToolMessage:
        """
        Execute a tool based on the tool call.
        
        Args:
            tool_call: The tool call message or description
        
        Returns:
            ToolMessage: The result of the tool execution
        """
        try:
            # Extract tool information based on the format of tool_call
            if hasattr(tool_call, "tool_calls") and tool_call.tool_calls:
                # Modern LangChain format
                tool_name = tool_call.tool_calls[0]["name"]
                tool_args = tool_call.tool_calls[0]["args"]
                tool_id = tool_call.tool_calls[0]["id"]
            elif hasattr(tool_call, "function_call"):
                # Older function_call format
                tool_name = tool_call.function_call["name"]
                tool_args = tool_call.function_call["arguments"]
                tool_id = tool_call.function_call.get("id", "function-call-" + str(id(tool_call)))
            else:
                # Dictionary format
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", "tool-call-" + str(id(tool_call)))
            
            self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
            
            # Get the tool function
            tool_func = self.name_to_tool.get(tool_name)
            if not tool_func:
                error_msg = f"Tool '{tool_name}' not found. Available tools: {', '.join(self.tool_names)}"
                self.logger.error(error_msg)
                return ToolMessage(content=error_msg, tool_call_id=tool_id)
            
            # Execute the tool
            if asyncio.iscoroutinefunction(tool_func):
                # If the tool is already async
                if hasattr(tool_func, "invoke"):
                    tool_output = await tool_func.invoke(tool_args)
                else:
                    tool_output = await tool_func(**tool_args)
            else:
                # Run sync tool in executor
                if hasattr(tool_func, "invoke"):
                    tool_output = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: tool_func.invoke(tool_args)
                    )
                else:
                    tool_output = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: tool_func(**tool_args)
                    )
            
            self.logger.info(f"Tool {tool_name} execution successful")
            
            # Return tool message
            return ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_id
            )
        except Exception as e:
            error_msg = f"Error executing tool: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error("Exception details:", exc_info=True)
            return ToolMessage(
                content=error_msg,
                tool_call_id=tool_id if 'tool_id' in locals() else "error-call"
            )

    async def invoke(self, input_message: str, streamer: QueueCallbackHandler) -> Union[Dict, str]:
        """
        Run the agent asynchronously.
        
        Args:
            input_message (str): The input message from the user
            streamer (QueueCallbackHandler): The streamer to use for streaming
        
        Returns:
            Union[Dict, str]: The output from the agent, either a dict with final_answer or string
        """
        self.logger.info(f"Running agent with input: {input_message}")
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        
        # Add the user message to memory
        self.memory.add_user_message(input_message)
        
        # Initialize scratchpad and iterations
        agent_scratchpad = []
        iterations = 0
        final_answer = None
        
        # Reset streamer state to ensure clean start for this query
        streamer.is_done = False
        streamer.final_answer_seen = False
        
        # Clear any leftover items in streamer queue
        try:
            while not streamer.queue.empty():
                await streamer.queue.get()
                streamer.queue.task_done()
        except Exception as e:
            self.logger.warning(f"Error clearing queue: {str(e)}")

        # Define the streaming function
        async def _stream(query: str) -> List[AIMessage]:
            # Get a fresh response object for each stream call to avoid carrying over state
            response = self.agent.with_config(
                callbacks=[streamer]
            )
            # we initialize the output dictionary that we will be populating with
            # our streamed output
            outputs = []
            
            # Create fresh input state to avoid cached responses
            input_state = {
                "input": query,
                "chat_history": chat_history,
                "agent_scratchpad": agent_scratchpad
            }
            
            # now we begin streaming
            async for token in response.astream(input_state):
                # Log the token type for debugging
                self.logger.debug(f"Processing token type: {type(token).__name__}")
                
                # Extract tool calls from various possible locations
                tool_calls = None
                
                # Handle ChatGenerationChunk objects
                if hasattr(token, "message") and hasattr(token.message, "tool_calls"):
                    tool_calls = token.message.tool_calls
                    self.logger.debug(f"Found tool_calls in ChatGenerationChunk.message: {tool_calls}")
                # Check for function_call in additional_kwargs
                elif hasattr(token, "additional_kwargs") and token.additional_kwargs.get("function_call"):
                    func_call = token.additional_kwargs.get("function_call")
                    tool_calls = [{
                        "name": func_call.get("name"),
                        "args": json.loads(func_call.get("arguments", "{}")) if isinstance(func_call.get("arguments"), str) else func_call.get("arguments", {}),
                        "id": func_call.get("id", f"call-{len(outputs)}"),
                        "type": "tool_call"
                    }]
                    self.logger.debug(f"Found function_call in additional_kwargs: {tool_calls}")
                # Check for tool_calls attribute directly
                elif hasattr(token, "tool_calls") and token.tool_calls:
                    tool_calls = token.tool_calls
                    self.logger.debug(f"Found tool_calls in token: {tool_calls}")
                # Check for function_calls in additional_kwargs
                elif hasattr(token, "additional_kwargs") and token.additional_kwargs.get("function_calls"):
                    tool_calls = token.additional_kwargs.get("function_calls")
                    self.logger.debug(f"Found function_calls in additional_kwargs: {tool_calls}")
                
                # Process the token based on what we found
                if tool_calls:
                    content = getattr(token, "content", "") if hasattr(token, "content") else ""
                    ai_message = AIMessage(
                        content=content,
                        tool_calls=tool_calls,
                        tool_call_id=getattr(token, "tool_call_id", None)
                    )
                    outputs.append(ai_message)
                # For tokens with normal content
                elif hasattr(token, "content") and token.content:
                    outputs.append(token)
            
            if not outputs:
                # If we got no outputs but have a final text, create one
                try:
                    if hasattr(response, "get_final_text"):
                        full_response = await response.get_final_text()
                        self.logger.debug(f"No outputs collected, using final response: {full_response}")
                        return [AIMessage(content=full_response)]
                    self.logger.warning("No outputs collected and no final text available")
                    return []
                except Exception as e:
                    self.logger.error(f"Error getting final text: {str(e)}")
                    return []
                
            self.logger.debug(f"Collected {len(outputs)} outputs from stream")
            return [
                AIMessage(
                    content=getattr(x, "content", "") if hasattr(x, "content") else "",
                    tool_calls=getattr(x, "tool_calls", []) if hasattr(x, "tool_calls") else [],
                    tool_call_id=getattr(x, "tool_call_id", None) if hasattr(x, "tool_call_id") else None
                ) for x in outputs
            ]
        
        # Run the agent loop
        while iterations < self.max_iterations:
            self.logger.info(f"Starting iteration {iterations+1}/{self.max_iterations}")

            # Check for repeated tool calls to break loops
            if iterations >= 2 and len(agent_scratchpad) >= 4:
                # Get last two tool calls
                last_two_tool_calls = []
                for item in agent_scratchpad[-4:]:
                    if hasattr(item, "tool_calls") and item.tool_calls:
                        last_two_tool_calls.append(item)
                
                # Check if we have at least two tool calls and they're identical
                if len(last_two_tool_calls) >= 2 and last_two_tool_calls[0].tool_calls == last_two_tool_calls[1].tool_calls:
                    self.logger.warning("Detected repeated tool call pattern, forcing final answer")
                    # Get the last tool observation to use as a basis for final answer
                    last_observation = None
                    for item in agent_scratchpad[-2:]:
                        if hasattr(item, "content") and isinstance(item.content, str) and "Result:" in item.content:
                            last_observation = item.content
                            break
                            
                    if last_observation:
                        final_answer = f"Based on my calculations, the answer is {last_observation.replace('Result: ', '')}"
                        break
            
            # invoke a step for the agent to generate a tool call
            tool_calls = await _stream(query=input_message)
            self.logger.debug(f"Agent response: {tool_calls}")
            
            # Skip if we didn't get any tool calls
            if not tool_calls:
                self.logger.warning("No tool calls received from agent")
                iterations += 1
                continue
                
            # Check if we have valid tool calls
            valid_tool_calls = []
            for tool_call in tool_calls:
                if hasattr(tool_call, "tool_calls") and tool_call.tool_calls:
                    valid_tool_calls.append(tool_call)
                else:
                    self.logger.warning(f"Invalid tool call received: {tool_call}")
                    
            if not valid_tool_calls:
                self.logger.warning("No valid tool calls received")
                iterations += 1
                continue
                
            # Check for final answer tool first
            found_final_answer = False
            for tool_call in valid_tool_calls:
                if hasattr(tool_call, "tool_calls") and tool_call.tool_calls:
                    for single_tool_call in tool_call.tool_calls:
                        if single_tool_call.get("name") == "final_answer":
                            final_answer = single_tool_call.get("args", {}).get("answer", "")
                            found_final_answer = True
                            self.logger.info(f"Found final answer: {final_answer}")
                            break
                    if found_final_answer:
                        break
            
            # If we found a final answer, break the loop
            if found_final_answer:
                break
                
            # Generate a list of tool execution coroutines only for valid tool calls
            tool_execution_coroutines = []
            for tool_call in valid_tool_calls:
                tool_execution_coroutines.append(self._execute_tool(tool_call))
                
            # Execute all tools concurrently
            if tool_execution_coroutines:
                tool_obs = await asyncio.gather(*tool_execution_coroutines)
            else:
                tool_obs = []
            
            # Skip if we got no tool observations
            if not tool_obs:
                self.logger.warning("No tool observations generated")
                iterations += 1
                continue

            # Add each tool call and its observation to the scratchpad
            for i, (tool_call, tool_ob) in enumerate(zip(valid_tool_calls, tool_obs)):
                self.logger.debug(f"Adding tool call {i} to scratchpad: {tool_call}")
                self.logger.debug(f"Adding tool observation {i} to scratchpad: {tool_ob}")
                agent_scratchpad.append(tool_call)
                agent_scratchpad.append(tool_ob)
            
            iterations += 1
        
        # add the final output to the chat history, we only add the "answer" field
        if final_answer:
            self.memory.add_ai_message(final_answer)
            return {"answer": final_answer, "tools_used": []}
        else:
            self.memory.add_ai_message("I was unable to provide a complete answer.")
            return {"answer": "I was unable to provide a complete answer.", "tools_used": []}

    def stream(self, input_message: str) -> asyncio.Queue:
        """
        Stream the agent's response synchronously.
        
        Args:
            input_message (str): The input message from the user
        
        Returns:
            asyncio.Queue: Queue with the streaming tokens
        """
        queue = asyncio.Queue()
        streamer = QueueCallbackHandler(queue)
        
        # Get the event loop and create tasks
        loop = asyncio.get_event_loop()
        
        # Reset the streamer first to ensure clean state
        async def setup_and_invoke():
            await streamer.reset()
            # Refresh agent to ensure clean state
            self._create_agent_chain()
            # Run invoke
            await self.invoke(input_message, streamer)
        
        # Create and run the full task sequence
        loop.create_task(setup_and_invoke())
        
        return queue


