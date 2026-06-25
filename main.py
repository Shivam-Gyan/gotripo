
"""

"""


import asyncio
import uuid
from supervisor import create_supervisor_agent
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file


_supervisor_agent = None

def initialize_supervisor_agent(model_name: str = "groq:openai/gpt-oss-120b", temperature: float = 0.7):
    """Initialize the supervisor agent with the specified model and temperature."""
    return create_supervisor_agent(model_name=model_name, temperature=temperature)



def conversate(user_input: str, config) -> str:

    global _supervisor_agent
    if not _supervisor_agent:
        _supervisor_agent = initialize_supervisor_agent()

    response =  _supervisor_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ],
        
    },config=config) #type: ignore

    return response["messages"][-1].text  #type: ignore


if __name__ == "__main__":
    
    thread_id = str(uuid.uuid4())
    config: dict = {"configurable": {"thread_id": thread_id}}
    
    while True:
        user_input = input("Enter your travel query: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the travel assistant. Goodbye!")
            break
        response = conversate(user_input, config)
        print("Response from the agent:")
        print(response)

