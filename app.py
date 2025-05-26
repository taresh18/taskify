import asyncio
import json

import uvicorn

from taskify.executor import AgentExecutor
from taskify.streaming import QueueCallbackHandler
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Define request model
class RequestBody(BaseModel):
    content: str

agent_executor = AgentExecutor()

# initilizing our application
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# streaming function
async def token_generator(input_message: str, streamer: QueueCallbackHandler):
    task = asyncio.create_task(agent_executor.invoke(
        input_message=input_message,
        streamer=streamer,
    ))
    
    # For collecting final answer
    final_answer = ""
    tool_usage = []
    
    # initialize various components to stream
    async for token in streamer:
        try:
            # Check for the final answer in the token
            if hasattr(token, "message") and hasattr(token.message, "additional_kwargs"):
                function_call = token.message.additional_kwargs.get("function_call")
                if function_call and function_call.get("name") == "final_answer":
                    if "answer" in function_call.get("arguments", {}):
                        # Extract the final answer
                        try:
                            args = function_call.get("arguments", "{}")
                            if isinstance(args, str):
                                args = json.loads(args)
                            final_answer = args.get("answer", "")
                        except:
                            pass
            
            # Extract content from token
            content = None
            if isinstance(token, str):
                if token == "<<DONE>>":
                    # If we have a final answer, return it
                    if final_answer:
                        yield final_answer
                    break
                elif token == "<<STEP_END>>":
                    continue
                else:
                    content = token
                    yield content
            elif hasattr(token, "content") and token.content:
                content = token.content
                yield content
            elif hasattr(token, "message") and hasattr(token.message, "content") and token.message.content:
                content = token.message.content
                yield content
                
            # Track tool usage (for debugging purposes)
            if hasattr(token, "message") and hasattr(token.message, "additional_kwargs"):
                function_call = token.message.additional_kwargs.get("function_call")
                if function_call and function_call.get("name"):
                    tool_name = function_call.get("name")
                    if tool_name != "final_answer":
                        tool_usage.append(tool_name)
                
        except Exception as e:
            print(f"Error streaming token: {e}")
            continue
    
    # If we have a final answer but didn't yield it yet, yield it now
    if final_answer and not content:
        yield final_answer
        
    await task

# invoke function
@app.post("/invoke")
async def invoke(content: str):
    queue: asyncio.Queue = asyncio.Queue()
    streamer = QueueCallbackHandler(queue)
    # return the streaming response
    return StreamingResponse(
        token_generator(content, streamer),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# Add a simple endpoint that returns just the final answer in JSON format
@app.post("/answer")
async def get_answer(request_body: RequestBody):
    """
    Non-streaming endpoint that returns just the final answer in a clean JSON format.
    """
    # Extract message from the request body
    input_message = request_body.content
    
    queue = asyncio.Queue()
    streamer = QueueCallbackHandler(queue)
    
    try:
        # Run the agent
        response = await agent_executor.invoke(
            input_message=input_message,
            streamer=streamer
        )
        
        # Look for final answer in the response
        final_answer = ""
        
        # Check if we have a clear final_answer in the response object
        if hasattr(response, "answer"):
            final_answer = response["answer"]
        else:
            # If not, we need to parse from the LLM response
            # Create a task to collect all tokens from the stream
            all_tokens = []
            async for token in streamer:
                all_tokens.append(token)
                
                # Look for final answer in tool calls
                if hasattr(token, "message") and hasattr(token.message, "additional_kwargs"):
                    function_call = token.message.additional_kwargs.get("function_call")
                    if function_call and function_call.get("name") == "final_answer":
                        if "arguments" in function_call:
                            try:
                                args = function_call.get("arguments", "{}")
                                if isinstance(args, str):
                                    args = json.loads(args)
                                final_answer = args.get("answer", "")
                                break
                            except Exception as e:
                                print(f"Error extracting final answer: {e}")
            
            # If we didn't find a final answer, try to extract it from text content
            if not final_answer:
                # Combine all text content from tokens
                combined_text = ""
                for token in all_tokens:
                    if isinstance(token, str):
                        combined_text += token
                    elif hasattr(token, "content") and token.content:
                        combined_text += token.content
                    elif hasattr(token, "message") and hasattr(token.message, "content"):
                        combined_text += token.message.content
                
                # Clean up the combined text
                combined_text = combined_text.replace("<<DONE>>", "").replace("<<STEP_END>>", "")
                final_answer = combined_text.strip()
        
        # Return the final answer in a clean JSON format
        return {"answer": final_answer}
        
    except Exception as e:
        print(f"Error in get_answer endpoint: {e}")
        return {"error": str(e)}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
