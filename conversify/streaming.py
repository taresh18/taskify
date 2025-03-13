import asyncio
import logging
import json
from typing import Any, Dict, AsyncIterator, List

from langchain.callbacks.base import AsyncCallbackHandler

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
        self.logger = logging.getLogger("agent.streaming")
        self.is_done = False
        self._lock = asyncio.Lock()

    async def __aiter__(self) -> AsyncIterator[Any]:
        """
        Async iterator to yield tokens from the queue.
        
        Yields:
            Any: Token from the queue
        """
        wait_time = 0.1  
        max_wait_time = 1.0
        empty_count = 0
        max_empty_count = 10
        
        while not self.is_done:
            try:
                try:
                    token = await asyncio.wait_for(self.queue.get(), timeout=wait_time)
                    empty_count = 0  # Reset counter when we get a token
                    
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
                    
                    if empty_count > max_empty_count:
                        self.logger.warning("Queue has been empty too long, ending stream")
                        await self.queue.put("<<DONE>>")
                        self.is_done = True
                        break
                    
                    # Increase wait time up to the max
                    wait_time = min(wait_time * 1.5, max_wait_time)
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
            chunk = kwargs.get("chunk")
            if chunk:
                # Track chunk details for debugging
                if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                    content_preview = chunk.message.content[:20] if chunk.message.content else "[empty]"
                    self.logger.debug(f"Received chunk with content preview: {content_preview}...")
                
                # Check for final_answer tool call
                if tool_calls := chunk.message.additional_kwargs.get("function_call"):
                    if isinstance(tool_calls, dict) and tool_calls.get("name") == "final_answer":
                        self.logger.debug("Final answer tool call detected with details: {tool_calls}")
                        
                        # Also add the final answer text directly to the queue
                        try:
                            if isinstance(tool_calls.get("arguments"), str):
                                args_str = tool_calls.get("arguments", "{}")
                                args = json.loads(args_str)
                                if "answer" in args:
                                    final_answer = args["answer"]
                                    self.logger.debug(f"Extracted final answer: {final_answer[:50]}...")
                                    await self.queue.put(final_answer)
                        except Exception as e:
                            self.logger.error(f"Error extracting final answer: {str(e)}")
                        
                        async with self._lock:
                            self.final_answer_seen = True
                
                # Put the chunk in the queue
                await self.queue.put(chunk)
                
            # If no chunk but we have a token string
            elif token:
                self.logger.debug(f"Received plain token: {token[:10]}...")
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