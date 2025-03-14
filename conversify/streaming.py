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
                    self.logger.debug(f"Yielding token: {token}")
                    yield token
                    
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
            chunk = kwargs.get("chunk")
            if not chunk:
                self.logger.debug("Skipping empty chunk")
                return
                                        
            # Check for tool calls in different formats
            tool_calls = None
                    
            # Format 2: tool_calls in message
            if hasattr(chunk, 'message') and hasattr(chunk.message, 'tool_calls'):
                tool_calls = chunk.message.tool_calls
                if tool_calls:
                    self.logger.debug(f"Found tool_calls in message: {tool_calls}")
            
            is_final_answer = False
            
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
        self.logger.debug("Resetting streaming handler state")
        async with self._lock:
            self.final_answer_seen = False
            self.is_done = False
            
            # Clear any pending tokens in the queue
            queue_items_cleared = 0
            try:
                while not self.queue.empty():
                    await self.queue.get()
                    self.queue.task_done()
                    queue_items_cleared += 1
            except Exception as e:
                self.logger.error(f"Error clearing queue during reset: {str(e)}")
                
            self.logger.debug(f"Streaming handler has been reset, cleared {queue_items_cleared} items from queue") 