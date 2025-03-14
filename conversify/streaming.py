import asyncio
import logging
import json
from typing import Any, Dict, AsyncIterator, List

from langchain.callbacks.base import AsyncCallbackHandler

from conversify.config import load_config

# Load configuration
config = load_config()
streaming_config = config.get("streaming", {})

class QueueCallbackHandler(AsyncCallbackHandler):
    """
    Callback handler that puts tokens into a queue for streaming.
    
    This handler is used to stream tokens from the LLM to the client.
    """

    def __init__(self, queue: asyncio.Queue):
        """
        Initialize the queue callback handler.
        
        Args:
            queue (asyncio.Queue): Queue to put tokens into
        """
        self.queue = queue
        self.final_answer_seen = False
        self.logger = logging.getLogger("streaming")
        self.is_done = False
        self._lock = asyncio.Lock()
        
        # Get streaming configuration
        self.wait_time = streaming_config.get("wait_time")
        self.max_wait_time = streaming_config.get("max_wait_time")
        self.max_empty_count = streaming_config.get("max_empty_count")

    async def __aiter__(self) -> AsyncIterator[Any]:
        """
        Async iterator to yield tokens from the queue.
        
        Yields:
            Any: Token from the queue
        """
        empty_count = 0
        wait_time = self.wait_time
        
        while not self.is_done:
            try:
                token = await asyncio.wait_for(self.queue.get(), timeout=wait_time)
                empty_count = 0  # Reset counter and wait time when we get a token
                wait_time = self.wait_time
                
                # Check for done signal
                if token == "<<DONE>>":
                    self.logger.debug("Iterator received DONE signal")
                    self.is_done = True
                    break
                
                if token:
                    yield token
                    
                # Ensure queue is properly marked as done
                self.queue.task_done()
                
            except asyncio.TimeoutError:
                empty_count += 1
                self.logger.debug(f"Queue empty for {empty_count} checks")
                
                if empty_count > self.max_empty_count:
                    self.logger.warning("Queue has been empty too long, ending stream")
                    await self.queue.put("<<DONE>>")
                    self.is_done = True
                    break
                
                # Increase wait time up to the max
                wait_time = min(wait_time * 1.5, self.max_wait_time)
                await asyncio.sleep(0.1)  # Small sleep to prevent tight loop
                    
            except Exception as e:
                self.logger.error(f"Error in streaming iterator: {str(e)}")
                await asyncio.sleep(0.1)
        
        self.logger.debug("Token iterator completed")

    async def on_chat_model_start(self, messages: List[Dict[str, Any]], *args, **kwargs) -> None:
        """
        Called when the chat model starts generating.
        
        Args:
            messages (List[Dict[str, Any]]): The messages being processed
            **kwargs: Additional arguments
        """
        self.logger.debug("Chat model starting generation")
        self.is_done = False
        self.logger.debug(f"Processing with {len(messages)} messages")
        
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Put new token in the queue.
        
        Args:
            token (str): The new token
            **kwargs: Arbitrary keyword arguments
        """
        try:
            # Skip empty tokens
            if not token and not kwargs.get("chunk"):
                self.logger.debug("Skipping empty token")
                return
                
            self.logger.debug(f"Received token: {token[:10] if isinstance(token, str) and len(token) > 10 else str(token)[:20]}")
            chunk = kwargs.get("chunk")
            
            if chunk:
                self.logger.debug(f"Processing chunk type: {type(chunk).__name__}")
                
                # Check for tool calls in different formats
                function_call = None
                tool_calls = None
                
                # Format 1: function_call in message.additional_kwargs
                if hasattr(chunk, 'message') and hasattr(chunk.message, 'additional_kwargs'):
                    function_call = chunk.message.additional_kwargs.get("function_call")
                    if function_call:
                        self.logger.debug(f"Found function_call in message.additional_kwargs: {function_call}")
                        
                # Format 2: tool_calls in message
                if hasattr(chunk, 'message') and hasattr(chunk.message, 'tool_calls'):
                    tool_calls = chunk.message.tool_calls
                    if tool_calls:
                        self.logger.debug(f"Found tool_calls in message: {tool_calls}")
                
                # Format 3: function_call in additional_kwargs
                if hasattr(chunk, 'additional_kwargs'):
                    if not function_call:
                        function_call = chunk.additional_kwargs.get("function_call")
                        if function_call:
                            self.logger.debug(f"Found function_call in additional_kwargs: {function_call}")
                
                # Format 4: tool_calls directly on chunk
                if hasattr(chunk, 'tool_calls') and not tool_calls:
                    tool_calls = chunk.tool_calls
                    if tool_calls:
                        self.logger.debug(f"Found tool_calls on chunk: {tool_calls}")
                
                # Check for final_answer in any of the formats
                is_final_answer = False
                
                if function_call and isinstance(function_call, dict) and function_call.get("name") == "final_answer":
                    is_final_answer = True
                    self.logger.debug(f"Final answer function call detected: {function_call}")
                    
                    # Extract and queue the final answer text
                    try:
                        if isinstance(function_call.get("arguments"), str):
                            args_str = function_call.get("arguments", "{}")
                            args = json.loads(args_str)
                            if "answer" in args:
                                final_answer = args["answer"]
                                self.logger.debug(f"Extracted final answer: {final_answer[:50]}...")
                                await self.queue.put(final_answer)
                    except Exception as e:
                        self.logger.error(f"Error extracting final answer: {str(e)}")
                
                elif tool_calls:
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict) and tool_call.get("name") == "final_answer":
                            is_final_answer = True
                            self.logger.debug(f"Final answer tool call detected: {tool_call}")
                            
                            # Extract and queue the final answer text
                            try:
                                args = tool_call.get("args", {})
                                if isinstance(args, str):
                                    args = json.loads(args)
                                
                                if "answer" in args:
                                    final_answer = args["answer"]
                                    self.logger.debug(f"Extracted final answer: {final_answer[:50]}...")
                                    await self.queue.put(final_answer)
                            except Exception as e:
                                self.logger.error(f"Error extracting final answer from tool call: {str(e)}")
                
                if is_final_answer:
                    async with self._lock:
                        self.final_answer_seen = True
                
                # Put the chunk in the queue
                await self.queue.put(chunk)
                
            # If no chunk but we have a token string
            elif token:
                self.logger.debug(f"Received plain token: {token[:10] if len(token) > 10 else token}...")
                await self.queue.put(token)
                
        except Exception as e:
            self.logger.error(f"Error in on_llm_new_token: {str(e)}")
            self.logger.error("Exception details:", exc_info=True)

        return

    async def on_llm_end(self, *args, **kwargs) -> None:
        """
        Put signal in the queue to indicate completion.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        try:
            self.logger.debug(f"LLM generation ended, received: {str(kwargs)[:100]}...")
            
            async with self._lock:
                is_final = self.final_answer_seen
            
            if is_final:
                self.logger.debug("LLM generation ended with final answer, sending DONE signal")
                await self.queue.put("<<DONE>>")
            else:
                self.logger.debug("Tool generation ended, sending STEP_END signal")
                await self.queue.put("<<STEP_END>>")
                
        except Exception as e:
            self.logger.error(f"Error in on_llm_end: {str(e)}")
            self.logger.error("Exception details:", exc_info=True)
            
        return 

    async def reset(self):
        """
        Reset the handler state for a new query.
        This helps prevent carrying over state between queries.
        """
        async with self._lock:
            self.final_answer_seen = False
            self.is_done = False
            
            # Clear any pending tokens in the queue
            try:
                while not self.queue.empty():
                    await self.queue.get()
                    self.queue.task_done()
            except Exception as e:
                self.logger.error(f"Error clearing queue during reset: {str(e)}")
                
        self.logger.debug("Streaming handler has been reset") 