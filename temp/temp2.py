import os
from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
import asyncio
from langchain.schema import HumanMessage


dotenv.load_dotenv()

LLM_CONFIG = {
    "model": "gemini-2.0-flash-exp",
    "temperature": 0.1,
    "max_tokens": 1000,
    "google_api_key": os.environ.get("GOOGLE_API_KEY"),
    "disable_streaming": False,
    "verbose": True
}

llm = ChatGoogleGenerativeAI(
    model=LLM_CONFIG["model"],
    temperature=LLM_CONFIG["temperature"],
    max_tokens=LLM_CONFIG["max_tokens"],
    google_api_key=LLM_CONFIG["google_api_key"],
    disable_streaming=LLM_CONFIG["disable_streaming"],
    verbose=LLM_CONFIG["verbose"]
)

async def stream_response(input_text):
    # create messages to be passed to chat LLM
    messages = [HumanMessage(content="tell me a long story")]

    # stream = llm.stream(messages)
    # full = next(stream)
    # for chunk in stream:
    #     full += chunk
    #     print(chunk.content, ' | ')
    
    async for chunk in llm.astream(messages):
        print(chunk.content, ' | ')

    # messages = [{"role": "user", "content": input_text}]
    # await llm.ainvoke(messages)
    # async for chunk in (await llm.astream(messages)):
    #     print(chunk)

async def main():
    user_input = "Tell me about artificial intelligence"  # Example input
    await stream_response(user_input)

if __name__ == "__main__":
    asyncio.run(main())
    
    
    