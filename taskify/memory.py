from typing import List, Dict, Any, Optional

from taskify.config import load_config
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

config = load_config()
memory_config = config.get("memory", {})

class ConversationalBufferMemory:
    """
    Conversational buffer memory that stores all messages.
    
    This memory implementation keeps track of the full conversation history.
    """
    
    def __init__(self, 
                 input_key: Optional[str] = None, 
                 output_key: Optional[str] = None):
        """
        Initialize the buffer memory.
        
        Args:
            input_key (Optional[str]): Key to extract input from
            output_key (Optional[str]): Key to extract output from
        """
        self.chat_history: List[BaseMessage] = []
        self.input_key = input_key or "input"
        self.output_key = output_key or "output"
    
    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to the chat history.
        
        Args:
            message (BaseMessage): Message to add
        """
        self.chat_history.append(message)
    
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the chat history.
        
        Args:
            message (str): User message to add
        """
        self.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        """
        Add an AI message to the chat history.
        
        Args:
            message (str): AI message to add
        """
        self.add_message(AIMessage(content=message))
    
    def clear(self) -> None:
        """Clear the chat history."""
        self.chat_history = []
    
    def get_messages(self) -> List[BaseMessage]:
        """
        Get all messages in the chat history.
        
        Returns:
            List[BaseMessage]: List of messages
        """
        return self.chat_history
    
    def _messages_to_string(self, messages: List[BaseMessage]) -> str:
        """
        Convert messages to a string representation.
        
        Args:
            messages (List[BaseMessage]): Messages to convert
            
        Returns:
            str: String representation of messages
        """
        result = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                result += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                result += f"AI: {message.content}\n"
            else:
                result += f"{message.type}: {message.content}\n"
        return result


class ConversationalBufferWindowMemory(ConversationalBufferMemory):
    """
    Conversational buffer memory with a sliding window.
    
    This memory implementation keeps track of the last k conversation exchanges.
    """
    
    def __init__(self, 
                 input_key: Optional[str] = None, 
                 output_key: Optional[str] = None):
        """
        Initialize the window memory.
        
        Args:
            k (int): Window size (number of exchanges to keep)
            input_key (Optional[str]): Key to extract input from
            output_key (Optional[str]): Key to extract output from
        """
        super().__init__(input_key, output_key)
        self.k = memory_config.get("window_k")
    
    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to the chat history and maintain window size.
        
        Args:
            message (BaseMessage): Message to add
        """
        super().add_message(message)
        
        # keep only the last k message pairs (human and ai)
        if len(self.chat_history) > 2 * self.k:
            num_to_remove = len(self.chat_history) - (2 * self.k)
            self.chat_history = self.chat_history[num_to_remove:]

    def get_messages(self) -> List[BaseMessage]:
        """
        Get messages in the current window.
        
        Returns:
            List[BaseMessage]: List of messages in the window
        """
        return self.chat_history[-2 * self.k:] if self.chat_history else []


class ConversationalSummaryMemory(ConversationalBufferMemory):
    """
    Conversational memory with summarization capabilities.
    
    This memory implementation keeps only the last k message pairs and a summary
    of the conversation.
    """
    
    def __init__(self, 
                 llm,
                 input_key: Optional[str] = None, 
                 output_key: Optional[str] = None):
        """
        Initialize the summary memory.
        
        Args:
            llm: LLM to use for summarization
            input_key (Optional[str]): Key to extract input from
            output_key (Optional[str]): Key to extract output from
        """
        super().__init__(input_key, output_key)
        self.llm = llm
        self.k = memory_config.get("summary_k")
        
        # Define a summary prompt template
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """
                You are an AI assistant tasked with summarizing a conversation history.
                Given the existing conversation summary and new messages, generate a new summary of the conversation. 
                Maintain as much relevant information as possible keeping the summary concise.
                Your summary should be factual and objective.
             """),
            ("human", "Existing conversation summary:\n{existing_summary}\n\nNew messages:\n{dropped_messages}")
        ])
    
    def add_message(self, message: BaseMessage) -> None:
        super().add_message(message)
        self._check_and_summarize()
    
    def _check_and_summarize(self) -> None:
        """
        Check if the conversation exceeds 2*k messages and summarize if needed.
        """
        # Check if a summary message exists at index 0
        existing_summary = None
        non_summary_messages = self.chat_history
        if self.chat_history and isinstance(self.chat_history[0], SystemMessage):
            existing_summary = self.chat_history.pop(0)
            non_summary_messages = self.chat_history 
        
        # If number of message pairs is less than 2*k, no need to summarize
        if len(non_summary_messages) <= 2 * self.k:
            if existing_summary:
                self.chat_history = [existing_summary] + non_summary_messages
            return
        
        # Determine how many messages to drop
        num_to_drop = len(non_summary_messages) - 2 * self.k
        dropped_messages = non_summary_messages[:num_to_drop]
        remaining_messages = non_summary_messages[num_to_drop:]
        
        dropped_str = self._messages_to_string(dropped_messages)
        
        if existing_summary:
            prompt_input = {
                "existing_summary": existing_summary.content,
                "dropped_messages": dropped_str
            }
        else:
            prompt_input = {
                "existing_summary": "",
                "dropped_messages": dropped_str
            }
        
        # Create a new summary
        try:
            chain = self.summary_prompt | self.llm
            result = chain.invoke(prompt_input)
            new_summary = result.content.strip()
        except Exception as e:
            new_summary = f"Error summarizing dropped messages: {str(e)}"
        
        summary_message = SystemMessage(content=f"Summary of previous conversation: {new_summary}")
        
        # Rebuild the chat history: summary message followed by the remaining messages
        self.chat_history = [summary_message] + remaining_messages
    
    def clear(self) -> None:
        """Clear the chat history and summary."""
        super().clear()
