#!/usr/bin/env python3
"""
AI Agent with web search and ArXiv capabilities using LangChain with Google's Gemini model.
This agent can search the web, fetch ArXiv abstracts, and provide final answers to user queries.
"""

import json
import os
import re
import operator
import requests
from dataclasses import asdict, dataclass
from typing import TypedDict, Annotated, Dict, List, Any
import datetime
import math
import logging
from simpleeval import SimpleEval
from tavily import TavilyClient

import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# ----- CONFIGURATION -----

dotenv.load_dotenv()

# Set up logging
LOG_CONFIG = {
    "log_file": "conversify.log",
    "level": logging.INFO,
    "format": "%(asctime)s - %(levelname)s - %(message)s"
}

# Configure logger
logging.basicConfig(
    filename=LOG_CONFIG["log_file"],
    level=LOG_CONFIG["level"],
    format=LOG_CONFIG["format"]
)
logger = logging.getLogger("conversify")

# Add console handler to see logs in console as well
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_CONFIG["format"]))
logger.addHandler(console_handler)

# LLM Configuration
LLM_CONFIG = {
    "model": "gemini-2.0-flash-exp",
    "temperature": 0.1,
    "max_tokens": 1000,
    "google_api_key": os.environ.get("GOOGLE_API_KEY"),
    "disable_streaming": False,
    "verbose": True
}

# Search Configuration
SEARCH_CONFIG = {
    "num_results": 5,
    "engine": "google",
    "language": "en",
    "region": "us",
    "api_key": os.environ.get("TAVILY_API_KEY")
}

# Agent Configuration
AGENT_CONFIG = {
    "max_iterations": 10
}

# ----- DATA MODELS -----

class AgentState(TypedDict):
    """Represents the current state of the agent."""
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


@dataclass
class SearchResult:
    """Dataclass representing a single search result."""
    title: str
    url: str
    content: str


# ----- TOOLS -----

@tool("calculator")
def calculator(expression: str) -> str:
    """
    Calculator Tool: Evaluates mathematical expressions.
    
    Use this tool when you need to perform calculations such as addition, subtraction,
    multiplication, division, or using mathematical functions like sqrt(), sin(), cos(),
    tan(), log(), exp(), etc. This tool can handle complex expressions with parentheses
    and supports mathematical constants like pi and e.
    
    Examples of valid expressions:
    - "2 + 2"
    - "sqrt(144) + 25"
    - "sin(pi/2)"
    - "log(10) * 5"
    """
    try:
        # Clean and parse the expression
        if isinstance(expression, dict) and "expression" in expression:
            expression = expression["expression"]
            
        # Fix: Properly handle JSON escaped characters
        expression = expression.replace("\\", "")
        
        # Validate the expression with a more permissive regex
        if not re.match(r'^[\d+\-*/().^\s_a-zA-Z,]+$', expression):
            return f"Error: Expression contains invalid characters: '{expression}'"
        
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            # Math module functions and constants
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }
        
        evaluator = SimpleEval(functions=allowed_names, names=allowed_names)
        result = evaluator.eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

@tool("current_datetime")
def current_datetime(format: str = "%Y-%m-%d %H:%M:%S", timezone: str = "local") -> str:
    """
    Gets the current date and time.
    
    Use this tool when you need to know the current date, time, or both.
    """
    try:
        format = "%Y-%m-%d %H:%M:%S"        
        now = datetime.datetime.now()
        
        try:
            formatted_datetime = now.strftime(format)
            return formatted_datetime
        except ValueError as e:
            return f"Error: Invalid datetime format string: {str(e)}"

    except Exception as e:
        return f"Error getting current datetime: {str(e)}"

search_client = TavilyClient(SEARCH_CONFIG["api_key"])

@tool("web_search")
def web_search(query: str) -> str:
    """
    Web Search Tool: Performs a web search using Google and returns relevant information.
    
    Use this tool when you need to:
    - Find up-to-date information about current events, people, places, or topics
    - Research facts that might not be in your training data
    - Get links to relevant websites for a query
    - Find specific information that requires an internet search
    
    The results contain titles, URLs, source, and snippets of the most relevant web pages.
    """
    try:
        data = search_client.search(
            query=query,
            search_depth="advanced",
            max_results=SEARCH_CONFIG["num_results"]
        )

        if "error" in data:
            return f"Error from SerpAPI: {data['error']}"

        results_list = []
                
        # Take the first num_results results
        for item in data['results']:
            search_result = SearchResult(
                title=item.get("title", "No title"),
                url=item.get("url", "No URL"),
                content=item.get("content", "No source"),
            )
            results_list.append(search_result)

        # Format results in a more readable form
        formatted_results = "Search results for: " + query + "\n\n"
        
        for i, result in enumerate(results_list, 1):
            formatted_results += f"{i}. {result.title}\n"
            formatted_results += f"   URL: {result.url}\n"
            formatted_results += f"   Content: {result.content}\n"
        
        return formatted_results
    except Exception as e:
        return f"Error performing SerpAPI search: {str(e)}"


@tool("final_answer")
def final_answer(answer: str) -> Dict[str, str]:
    """
    Final answer to a user's query
    
    Use this tool ONLY when you're ready to provide the FINAL ANSWER to the user's question.
    This should be used at the end of your reasoning process, after you've gathered all necessary
    information and are ready to respond directly to the user's query.
    
    IMPORTANT: After performing searches or using other tools, you MUST use this tool to provide 
    your conclusion. Do not continue searching indefinitely. If you've made multiple searches
    on the same topic, use this tool to synthesize what you've learned and provide your best answer.
    
    The answer should be comprehensive and directly address the user's original question.
    """
    # Check if the answer is already in JSON format and try to parse it
    if isinstance(answer, str):
        try:
            # If it's a JSON string, parse it
            if answer.strip().startswith('{'):
                parsed = json.loads(answer)
                if isinstance(parsed, dict) and "answer" in parsed:
                    return parsed
        except json.JSONDecodeError:
            pass
    
    # If not JSON or parsing failed, just wrap it in a dictionary
    return {"answer": answer}


# ----- LLM SETUP -----

def initialize_llm() -> ChatGoogleGenerativeAI:
    """Initialize and return the language model."""
    return ChatGoogleGenerativeAI(
        model=LLM_CONFIG["model"],
        temperature=LLM_CONFIG["temperature"],
        max_tokens=LLM_CONFIG["max_tokens"],
        google_api_key=LLM_CONFIG["google_api_key"],
        disable_streaming=LLM_CONFIG["disable_streaming"],
        verbose=LLM_CONFIG["verbose"]
    )


# ----- AGENT SETUP -----

def create_system_prompt() -> str:
    """Create the system prompt for the agent."""
    return """You are Conversify, a smart chatbot agent with access to multiple tools.
Given the user's query you must decide what to do with it based on the
list of tools provided to you.

IMPORTANT INSTRUCTIONS:
1. If a tool returns a valid and useful response, DO NOT call the same tool with the same input again.
2. For the current_datetime tool, call it ONLY ONCE to get the current time, then use that information.
3. For calculations, use the calculator tool not more than twice with a properly formatted expression.
4. When using web_search:
   - Read the search results carefully
   - Extract relevant information from the snippets and titles
   - Summarize the information in your own words
   - Only search again if you need different information (with a different query)
5. Always use the final_answer tool to provide your conclusion once you have gathered enough information.
6. DO NOT make multiple consecutive calls to the same tool with the same input.
7. When providing your final answer, DO NOT include raw JSON or search results. Instead, provide a concise, well-formatted answer.

When using the calculator tool:
- For multiplication use the "*" symbol (e.g., "2*3")
- For division use the "/" symbol
- For addition use the "+" symbol
- For subtraction use the "-" symbol

You should aim to collect information from a diverse range of sources before
providing the answer to the user. Once you have collected enough information
to answer the user's question (stored in the scratchpad) use the final_answer
tool."""


def create_agent_prompt() -> ChatPromptTemplate:
    """Create and return the agent prompt template."""
    return ChatPromptTemplate.from_messages([
        ("system", create_system_prompt()),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("assistant", "scratchpad: {scratchpad}"),
    ])


def create_scratchpad(intermediate_steps: list[AgentAction]) -> str:
    """
    Create a scratchpad string from intermediate steps.

    Args:
        intermediate_steps: List of (AgentAction, result) tuples

    Returns:
        Formatted scratchpad string
    """
    research_steps = []
    for i, action in enumerate(intermediate_steps):
        if action.log != "TBD":
            # this was the ToolExecution
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n---\n".join(research_steps)


def setup_agent(llm: ChatGoogleGenerativeAI, tools: List) -> Any:
    """Set up and return the agent with the given LLM and tools."""
    prompt = create_agent_prompt()
    
    return (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "scratchpad": lambda x: create_scratchpad(
                intermediate_steps=x["intermediate_steps"]
            ),
        }
        | prompt
        | llm.bind_tools(tools, tool_choice="any")
    )

# ----- GRAPH FUNCTIONS -----

def run_agent(state: AgentState) -> Dict:
    """
    Run the agent to get the next action.
    
    Args:
        state: The current agent state
        
    Returns:
        Updated state with a new action
    """
    logger.info("Running agent")
    logger.info(f"Intermediate steps: {state['intermediate_steps']}")
    
    # Check for repeated tool calls to avoid unnecessary API calls
    if should_skip_repeated_tool_call(state):
        # Get the most recent tool result to include in the final answer
        last_result = "No information available."
        if state["intermediate_steps"] and len(state["intermediate_steps"]) > 0:
            for step in reversed(state["intermediate_steps"]):
                if step.log and step.log != "TBD":
                    last_result = step.log
                    break
        
        # Craft a more informative final answer that processes the search results
        if "Search results for:" in last_result:
            # Process search results to create a better answer
            lines = last_result.split("\n")
            query = lines[0].replace("Search results for: ", "")
            summary = f"Based on my search for '{query}', I found that "
            
            # Extract key information from the results
            for line in lines:
                if line.strip() and not line.startswith("   URL:") and not line.startswith("   Source:"):
                    if line[0].isdigit() and ". " in line:
                        # This is a title
                        title = line.split(". ", 1)[1]
                        summary += f"{title}. "
                    elif line.strip() and not line.startswith("   "):
                        # This is other relevant content
                        summary += f"{line}. "
            
            answer = summary
        else:
            # For other tools, just use the result directly
            answer = f"Based on the information I found: {last_result}"
        
        return {
            "intermediate_steps": [
                AgentAction(
                    tool="final_answer",
                    tool_input=json.dumps({"answer": answer}),
                    log="TBD"
                )
            ]
        }
    
    out = agent.invoke(state)
    
    # Extract tool call from response
    tool_name = out.additional_kwargs.get('function_call', {}).get('name')
    tool_args = out.additional_kwargs.get('function_call', {}).get('arguments')
    
    # If we couldn't extract a proper tool call, default to final_answer
    if not tool_name or not tool_args:
        logger.warning("Invalid tool format detected, defaulting to final_answer")
        return {
            "intermediate_steps": [
                AgentAction(
                    tool="final_answer",
                    tool_input=json.dumps({"answer": "I'll provide my best answer based on what I know."}),
                    log="TBD"
                )
            ]
        }
    
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log="TBD"
    )
    
    return {
        "intermediate_steps": [action_out]
    }

def should_skip_repeated_tool_call(state: AgentState) -> bool:
    """
    Check if we're seeing too many repeated tool calls of the same type and should skip.
    
    Args:
        state: The current agent state
        
    Returns:
        Boolean indicating if we should skip to final answer
    """
    # If less than 3 steps, no need to check for repetition yet
    if len(state["intermediate_steps"]) < 3:
        return False
    
    # Get the last 3 steps
    last_steps = state["intermediate_steps"][-3:]
    
    # Check if all are the same tool
    tools = [step.tool for step in last_steps]
    if len(set(tools)) == 1 and tools[0] != "final_answer":
        # All the same tool, now check inputs
        inputs = [str(step.tool_input) for step in last_steps]
        
        # If we have at least 2 identical inputs in a row, we're in a loop
        if len(set(inputs[-2:])) == 1:
            logger.info(f"Detected repeated tool call pattern: ({tools[0]}, {inputs[0]})")
            return True
    
    return False


def router(state: AgentState) -> str:
    """
    Route to the next tool based on the last action.
    
    Args:
        state: The current agent state
        
    Returns:
        Name of the tool to use next
    """
    # Return the tool name to use
    if isinstance(state["intermediate_steps"], list) and state["intermediate_steps"]:
        return state["intermediate_steps"][-1].tool
    else:
        # If we output bad format go to final answer
        logger.warning("Router invalid format")
        return "final_answer"


def run_tool(state: AgentState) -> Dict:
    """
    Run the tool specified in the last action.
    
    Args:
        state: The current agent state
        
    Returns:
        Updated state with the tool result
    """
    # Use this as helper function so we repeat less code
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input
    
    logger.info(f"Running tool: {tool_name} with input: {tool_args}")
    
    # Fix: Improve tool argument parsing
    if isinstance(tool_args, str):
        try:
            parsed_args = json.loads(tool_args)
            if tool_name == "calculator" and isinstance(parsed_args, dict) and "expression" in parsed_args:
                # Remove escape characters and ensure clean expression
                parsed_args["expression"] = parsed_args["expression"].replace("\\", "")
                tool_args = parsed_args
        except json.JSONDecodeError:
            # If not valid JSON, keep as is
            pass
    
    # Run tool with properly parsed arguments
    out = tool_str_to_func[tool_name].invoke(input=tool_args)
    
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(out)
    )
    
    return {"intermediate_steps": [action_out]}


# ----- RESULT EXTRACTION -----

def extract_final_answer(agent_output: Dict) -> str:
    """
    Extracts the final answer from the agent output.
    
    Args:
        agent_output: The output from the runnable.invoke() call
        
    Returns:
        The extracted final answer as a string
    """
    if not agent_output or "intermediate_steps" not in agent_output:
        return "No answer found in agent output."
    
    # Get the last intermediate step
    last_step = agent_output["intermediate_steps"][-1]
    
    # Check if it's from the final_answer tool
    if last_step.tool == "final_answer":
        # Extract the answer from the tool input or log
        try:
            # First try the tool_input if it's available
            if hasattr(last_step, 'tool_input') and last_step.tool_input:
                if isinstance(last_step.tool_input, str):
                    try:
                        answer_dict = json.loads(last_step.tool_input)
                        if isinstance(answer_dict, dict) and "answer" in answer_dict:
                            return answer_dict["answer"]
                    except json.JSONDecodeError:
                        pass
                elif isinstance(last_step.tool_input, dict) and "answer" in last_step.tool_input:
                    return last_step.tool_input["answer"]
            
            # Then try the log
            log_data = last_step.log
            if isinstance(log_data, str):
                try:
                    answer_dict = json.loads(log_data)
                    if isinstance(answer_dict, dict) and "answer" in answer_dict:
                        return answer_dict["answer"]
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, return the log directly
            return str(log_data)
        except Exception as e:
            logger.error(f"Error extracting final answer: {str(e)}")
            return str(last_step.log)
    elif last_step.log and last_step.log != "TBD":
        # If not a final_answer but we have a non-empty log, return it
        if last_step.tool == "calculator":
            return f"The calculation result is: {last_step.log}"
        elif last_step.tool == "web_search":
            return f"Based on my search, I found: {last_step.log}"
        elif last_step.tool == "current_datetime":
            return f"The current date/time is: {last_step.log}"
        else:
            return f"Tool result: {last_step.log}"
    
    return "No final answer found in agent output."


# ----- MAIN EXECUTION -----

def run_with_query(query: str, chat_history: list[BaseMessage] = None) -> tuple[str, list[BaseMessage]]:
    """
    Run the agent with a specific query and return the final answer.
    
    Args:
        query: The user's query
        chat_history: Optional list of previous chat messages
        
    Returns:
        Tuple of (final_answer, updated_chat_history)
    """
    logger.info(f"Processing query: {query}")
    
    # Use provided chat history or initialize empty if none exists
    if chat_history is None:
        chat_history = []
    
    inputs = {
        "input": query,
        "chat_history": chat_history,
        "intermediate_steps": [],  # Start with empty intermediate steps for each query
    }
    
    # Run the agent graph
    output = runnable.invoke(inputs)
    
    # Extract the final answer
    answer = extract_final_answer(output)
    logger.info(f"Final answer: {answer}")
    
    # Add only the query and final answer to chat history
    updated_chat_history = chat_history.copy()
    updated_chat_history.append(HumanMessage(content=query))
    updated_chat_history.append(AIMessage(content=answer))
    
    return answer, updated_chat_history


# Initialize components
llm = initialize_llm()
tools = [web_search, final_answer, current_datetime, calculator]
tool_str_to_func = {tool.name: tool for tool in tools}
agent = setup_agent(llm, tools)

# Set up the graph
graph = StateGraph(AgentState)
graph.add_node("agent", run_agent)
graph.add_node("web_search", run_tool)
graph.add_node("current_datetime", run_tool)
graph.add_node("calculator", run_tool)
graph.add_node("final_answer", run_tool)
graph.set_entry_point("agent")
graph.add_conditional_edges(source="agent", path=router)

# Create edges from each tool back to the oracle
for tool_obj in tools:
    if tool_obj.name != "final_answer":
        graph.add_edge(tool_obj.name, "agent")

# If anything goes to final answer, it must then move to END
graph.add_edge("final_answer", END)

# Compile the graph
runnable = graph.compile()

# Example usage
if __name__ == "__main__":
    logger.info("Starting Conversify AI Agent")
    
    # Interactive mode
    print("\n🤖 Welcome to Conversify AI Agent 🤖")
    print("Type 'exit', 'quit', or 'q' to end the conversation")
    
    # Initialize chat history - will contain only user queries and agent final answers
    chat_history = []
    
    while True:
        # Get user input
        user_query = input("\nYour question: ")
        
        # Check for exit commands
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("Thank you for using Conversify. Goodbye!")
            break
        
        # Process the query
        try:
            answer, chat_history = run_with_query(user_query, chat_history)
            # print("\n🤖 Conversify: ", answer)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            print(f"\n❌ Sorry, an error occurred: {str(e)}")

