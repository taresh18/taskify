from dataclasses import asdict, dataclass
import os
import re
from typing import Dict, List
import json
import datetime
import requests
import math

from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field
from simpleeval import SimpleEval

from conversify.config import load_config

# Load configuration
config = load_config()
tools_config = config.get("tools", {})


class FinalAnswerInput(BaseModel):
    """Input schema for the final answer tool."""
    answer: str = Field(description="The final answer to the user's question")


@tool("final_answer", args_schema=FinalAnswerInput)
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
    return {"answer": answer}


class CalculatorInput(BaseModel):
    """Input schema for the calculator tool."""
    expression: str = Field(description="The mathematical expression to evaluate")

@tool("calculator", args_schema=CalculatorInput)
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
    # Validate the expression: allow digits, operators, parentheses, letters, commas, spaces, and underscores.
    if not re.match(r"^[\d+\-*/().,\s_a-zA-Z]+$", expression):
        return "Error: Expression contains invalid characters."
    
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
    
    try:
        evaluator = SimpleEval(functions=allowed_names, names=allowed_names)
        
        # Replace ** with ^ for Python compatibility
        if "**" in expression:
            expression = expression.replace("**", "^")  
        
        result = evaluator.eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"


class CurrentDateTimeInput(BaseModel):
    """Input schema for the current datetime tool."""
    format: str = Field(
        default="%Y-%m-%d %H:%M:%S", 
        description="The datetime format to use (e.g. '%Y-%m-%d %H:%M:%S')"
    )
    timezone: str = Field(
        default="local",
        description="Timezone: 'utc' for UTC time, 'local' for local time"
    )

@tool("current_datetime", args_schema=CurrentDateTimeInput)
def current_datetime(format: str = "%Y-%m-%d %H:%M:%S", timezone: str = "local") -> str:
    """
    Gets the current date and time in the specified format and timezone.
    
    Use this tool when you need to know the current date, time, or both.
    IMPORTANT: DO NOT USE THIS TOOL WHEN ANSWERING USER'S QUESTIONS ABOUT WEATHER
    You can specify:
    - format: How the date/time should be displayed (default: "%Y-%m-%d %H:%M:%S")
      Common formats: "%Y-%m-%d" (date only), "%H:%M:%S" (time only), 
      "%b %d, %Y" (e.g., "Jan 01, 2023")
    - timezone: Either "utc" for Universal Coordinated Time or "local" for the
      system's local time (default: "local")
    """
    try:
        if not format:
            format = "%Y-%m-%d %H:%M:%S"
        
        format = format.replace('\\', '')
        
        timezone = timezone.lower() if timezone else "local"
        if timezone == "utc":
            now = datetime.datetime.utcnow()
        else:
            now = datetime.datetime.now()
        
        try:
            formatted_datetime = now.strftime(format)
            return formatted_datetime
        except ValueError as e:
            return f"Error: Invalid datetime format string: {str(e)}"

    except Exception as e:
        return f"Error getting current datetime: {str(e)}"

@dataclass
class SearchResult:
    """Dataclass representing a single search result."""
    title: str
    url: str
    source: str
    snippet: str

class SerpapiSearchInput(BaseModel):
    """Input schema for the SerpAPI search tool."""
    query: str = Field(description="The search query for SerpAPI search.")

@tool("serpapi_search", args_schema=SerpapiSearchInput)
def serpapi_search(query: str) -> str:
    """
    Web Search Tool: Performs a web search using Google and returns relevant information.
    
    Use this tool when you need to:
    - Find up-to-date information about current events, people, places, or topics
    - Research facts that might not be in your training data
    - Get links to relevant websites for a query
    - Find specific information that requires an internet search
    
    The results contain titles, URLs, source, and snippets of the most relevant web pages.
    """
    num_results = tools_config.get("serpapi_num_results", 5)
        
    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        return "Error: SerpAPI key is not set."

    endpoint = "https://serpapi.com/search"
    params = {
        "engine": tools_config.get("serpapi_engine", "google"), 
        "q": query,
        "api_key": api_key,
        "hl": tools_config.get("serpapi_language", "en"),
        "gl": tools_config.get("serpapi_region", "us"),
    }

    try:
        # Log the request being made (without API key)
        safe_params = params.copy()
        safe_params.pop("api_key")
        print(f"Making SerpAPI request with params: {safe_params}")
        
        # Make the request
        response = requests.get(endpoint, params=params, timeout=10) 
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return f"Error from SerpAPI: {data['error']}"

        results_list = []
        if "organic_results" in data:
            results = data["organic_results"]
            
            if not results:
                return f"No search results found for query: '{query}'. Try refining your search terms."
                
            # take the first num_results results
            for item in results[:num_results]:
                search_result = SearchResult(
                    title=item.get("title", "No title"),
                    url=item.get("link", "No URL"),
                    source=item.get("source", "No source"),
                    snippet=item.get("snippet", "No snippet")
                )
                results_list.append(search_result)

            # Convert the list of SearchResult objects to a list of dicts and dump as JSON.
            return json.dumps([asdict(r) for r in results_list], indent=2)
    except Exception as e:
        return f"Error performing SerpAPI search: {str(e)}"
    

def get_all_tools() -> List[BaseTool]:
    """
    Get all available tools.
    
    Returns:
        List[BaseTool]: List of all tools
    """
    return [
        final_answer,
        calculator,
        current_datetime,
        serpapi_search,
    ]
