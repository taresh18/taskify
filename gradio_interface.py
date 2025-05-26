import gradio as gr
import requests
import re
import json

# API endpoint
API_URL = "http://localhost:8000/answer"  # Use the new /answer endpoint

def process_query(message, history):
    """
    Process a user query by sending it to the FastAPI server.
    
    Args:
        message: The user's query
        history: Current chat history
        
    Returns:
        Updated chat history
    """
    # Add user message to history
    history = history + [{"role": "user", "content": message}]
    
    # Create a temporary thinking message
    history = history + [{"role": "assistant", "content": "Thinking..."}]
    yield history
    
    try:
        # Send request to the API - use the /answer endpoint which returns clean JSON
        response = requests.post(
            url=API_URL,
            json={"content": message},  # Send as JSON in the request body
            timeout=60
        )
        
        if response.status_code == 200:
            try:
                # Parse the JSON response
                data = response.json()
                print("Received response from server:", data)
                
                # Extract the answer from the JSON
                if "answer" in data:
                    answer = data["answer"]
                elif "error" in data:
                    answer = f"Error: {data['error']}"
                else:
                    answer = "Received an unexpected response format from the server."
                    
                # Update the assistant's message with the answer
                history = history[:-1] + [{"role": "assistant", "content": answer}]
            except ValueError as e:
                # If JSON parsing fails, use the raw text
                print("Error parsing JSON:", e)
                history = history[:-1] + [{"role": "assistant", "content": f"Error parsing response: {response.text[:200]}..."}]
        else:
            error_msg = f"Error: Server returned status code {response.status_code}"
            history = history[:-1] + [{"role": "assistant", "content": error_msg}]
            
    except requests.RequestException as e:
        # Handle request errors
        error_msg = f"Failed to connect to the server: {str(e)}"
        history = history[:-1] + [{"role": "assistant", "content": error_msg}]
    
    # Yield the final result
    yield history

def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(title="Taskify Agent", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🤖 Taskify Agent")
        gr.Markdown("Ask questions and get answers from the Taskify AI agent.")
        
        chatbot = gr.Chatbot(
            label="Conversation",
            height=500,
            avatar_images=("👤", "🤖"),
            type="messages"
        )
        
        with gr.Row():
            query_input = gr.Textbox(
                placeholder="Ask me anything...",
                lines=2,
                show_label=False
            )
            submit_btn = gr.Button("Submit", variant="primary")
        
        # Set up event handlers
        submit_btn.click(
            fn=process_query,
            inputs=[query_input, chatbot],
            outputs=chatbot
        ).then(
            fn=lambda: "",  # Clear input after submission
            outputs=query_input
        )
        
        query_input.submit(
            fn=process_query,
            inputs=[query_input, chatbot],
            outputs=chatbot
        ).then(
            fn=lambda: "",  # Clear input after submission
            outputs=query_input
        )
        
        # Example queries
        gr.Markdown("## Example Queries")
        examples = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms",
            "How tall is Mount Everest?",
            "What's the weather like in New York today?",
        ]
        
        # Create example buttons
        for example in examples:
            gr.Button(example).click(
                fn=lambda example=example: example,  # Use default parameter to capture each example correctly
                outputs=query_input
            ).then(
                fn=process_query,
                inputs=[query_input, chatbot],
                outputs=chatbot
            ).then(
                fn=lambda: "",  # Clear input after submission
                outputs=query_input
            )
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", share=True) 