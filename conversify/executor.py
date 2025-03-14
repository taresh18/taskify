import asyncio
import json
import os
from langchain_core.runnables import ConfigurableField
from typing import List, Dict, Union

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    BaseMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

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
            "Your job is to answer the user's questions as accurately as possible using the tools provided to you.\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "1. Make your searches as SPECIFIC and TARGETED as possible. Prefer ONE comprehensive search over multiple narrow searches.\n"
            "2. After receiving search results, CAREFULLY ANALYZE the information before deciding next steps.\n"
            "3. DO NOT search for the same information repeatedly or make multiple similar searches.\n"
            "4. For most queries, ONE good search should be sufficient. After your first search, evaluate if you truly need more information.\n"
            "5. Once you have sufficient information from search results, YOU MUST use the final_answer tool to provide a conclusion.\n"
            "6. After making TWO searches on the same topic, you MUST use the final_answer tool with your synthesis of the information.\n"
            "7. For calculator questions, once you get the numerical result, use the final_answer tool right away.\n\n"
            "8. Use tools to answer the user's CURRENT question, not previous questions.\n"
            "MEMORY INSTRUCTIONS:\n"
            "9. For questions about previous conversations or what you've told the user before, DO NOT USE TOOLS. Instead, examine the chat history provided to you.\n"
            "10. If asked 'when did you tell me about X' or 'what did you say about X earlier', look in your chat history for that information.\n"
            "IMPORTANT: You MUST use the final_answer tool to provide your conclusion. DO NOT keep searching indefinitely."
            "IMPORTANT: DO NOT USE current_datetime TOOL WHEN ANSWERING USER'S QUESTIONS ABOUT WEATHER"
        )

        # Use provided values or get from config
        self.max_iterations = agent_config.get("max_iterations")
        self.async_mode = agent_config.get("async_mode")
        self.logger = setup_logging()
        
        self.tools = get_all_tools()
        # Create a tool name to tool mapping
        self.name_to_tool = {
            getattr(tool, "name", tool.__class__.__name__): tool
            for tool in self.tools
        }

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

        # Initialize memory based on type
        if memory_config.get("type") == "window":
            self.memory = ConversationalBufferWindowMemory()
        elif memory_config.get("type") == "summary":
            self.memory = ConversationalSummaryMemory(llm=self.llm)
        else:  
            self.memory = ConversationalBufferMemory()

        # Create the agent chain
        self._create_agent_chain()
        
        self.logger.info(f"Initialized AgentExecutor with {len(self.tools)} tools and {memory_config.get('type')} memory")

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
            # "Always format tool usage as:\n\n"
            # "Thought: [reasoning]\n"
            # "Action: [tool_name]\n" 
            # "Action Input: {{\"param\": \"value\"}}\n\n"
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
            | self.prompt
            | self.llm.bind_tools(self.tools, tool_choice="any")
        )


    async def _execute_tool(self, tool_call) -> ToolMessage:
        """
        Execute a tool based on the tool call.
        
        Args:
            tool_call: The tool call message or description
        
        Returns:
            ToolMessage: The result of the tool execution
        """
        try:
            # extract tool information from the tool call
            tool_name = tool_call.tool_calls[0]["name"]
            tool_args = tool_call.tool_calls[0]["args"]
            tool_id = tool_call.tool_calls[0]["id"]
            
            self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
            
            # Get the tool function
            tool_func = self.name_to_tool.get(tool_name)
            if not tool_func:
                error_msg = f"Tool '{tool_name}' not found. Available tools: {', '.join(self.tool_names)}"
                self.logger.error(error_msg)
                return ToolMessage(content=error_msg, tool_call_id=tool_id)
            
            # Execute the tool
            if asyncio.iscoroutinefunction(tool_func):
                tool_output = await tool_func.invoke(tool_args)
            else:
                # Run sync tool in executor
                tool_output = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: tool_func.invoke(tool_args)
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
        
    # Define the streaming function
    async def _stream(self, query: str, chat_history: List[BaseMessage], agent_scratchpad: List[BaseMessage], streamer: QueueCallbackHandler) -> List[AIMessage]:
        response = self.agent.with_config(
            callbacks=[streamer]
        )
        # outputs is a list of AIMessage objects
        outputs = []
        
        # now we begin streaming
        async for token in response.astream({
            "input": query,
            "chat_history": chat_history,
            "agent_scratchpad": agent_scratchpad
        }):                
            # Extract tool calls from various possible locations
            tool_calls = None
            
            # Check for function_call in additional_kwargs
            if hasattr(token, "additional_kwargs") and token.additional_kwargs.get("function_call"):
                func_call = token.additional_kwargs.get("function_call")
                tool_calls = [{
                    "name": func_call.get("name"),
                    "args": json.loads(func_call.get("arguments", "{}")) if isinstance(func_call.get("arguments"), str) else func_call.get("arguments", {}),
                    "id": func_call.get("id", f"call-{len(outputs)}"),
                    "type": "tool_call"
                }]
                self.logger.debug(f"Found function_call in additional_kwargs: {tool_calls}")
            
                # Process the token based on what we found
                content = getattr(token, "content", "") if hasattr(token, "content") else ""
                ai_message = AIMessage(
                    content=content,
                    tool_calls=tool_calls,
                    tool_call_id=getattr(token, "tool_call_id", None)
                )
                outputs.append(ai_message)
            
        self.logger.debug(f"Collected {len(outputs)} outputs from stream")
        return outputs

    async def invoke(self, input_message: str, streamer: QueueCallbackHandler) -> Union[Dict, str]:
        """
        Run the agent asynchronously.
        
        Args:
            input_message (str): The input message from the user
            streamer (QueueCallbackHandler): The streamer to use for streaming
        
        Returns:
            Union[Dict, str]: The output from the agent, either a dict with final_answer or string
        """        
        # Get a clean slate version of chat history
        chat_history = self.memory.get_messages()
        self.logger.debug(f"Chat history: {chat_history}")
        
        # Initialize scratchpad and iterations
        agent_scratchpad = []
        iterations = 0
        final_answer = None
        
        # Run the agent loop
        while iterations < self.max_iterations:
            self.logger.info(f"Starting iteration {iterations+1}/{self.max_iterations}")

            # invoke a step for the agent to generate a tool call
            tool_calls = await self._stream(query=input_message, chat_history=chat_history, agent_scratchpad=agent_scratchpad, streamer=streamer)
            self.logger.debug(f"Agent response: {tool_calls}")
            
            # Skip if we didn't get any tool calls
            if not tool_calls:
                self.logger.warning("No tool calls received from agent")
                iterations += 1
                continue
                
            # Check for final answer tool first
            for tool_call in tool_calls:
                if hasattr(tool_call, "tool_calls") and tool_call.tool_calls:
                    for single_tool_call in tool_call.tool_calls:
                        if single_tool_call.get("name") == "final_answer":
                            final_answer = single_tool_call.get("args", {}).get("answer", "")
                            self.logger.info(f"Found final answer: {final_answer}")
                            break
                    if final_answer:
                        break
            
            # If we found a final answer, break the loop
            if final_answer:
                break
                
            # Generate a list of tool execution coroutines 
            tool_execution_coroutines = []
            for tool_call in tool_calls:
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
            for i, (tool_call, tool_ob) in enumerate(zip(tool_calls, tool_obs)):
                self.logger.debug(f"Adding tool call {i} to scratchpad: {tool_call}")
                self.logger.debug(f"Adding tool observation {i} to scratchpad: {tool_ob}")
                agent_scratchpad.append(tool_call)
                agent_scratchpad.append(tool_ob)
            
            iterations += 1
        
        # Add messages to memory
        self.memory.add_user_message(input_message)
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
        
        self.logger.info(f"User input received: {input_message}")
        async def setup_and_invoke():
            await self.invoke(input_message, streamer)
        
        # Create and run the full task sequence
        loop.create_task(setup_and_invoke())
        
        return queue


