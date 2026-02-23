import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = init_chat_model(
    model="MiniMaxAI/MiniMax-M2.5:novita",
    model_provider="openai",
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

class TravelState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def get_flight_price(persons: int) -> float:
    """Calculate total flight cost for given number of persons."""
    return persons * 25000.0

@tool
def get_hotel_price(persons: int, days: int) -> float:
    """Calculate total hotel cost for given persons and days."""
    return persons * days * 5000.0

tools = [get_flight_price, get_hotel_price]

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: TravelState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: TravelState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END

builder = StateGraph(TravelState)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", tool_node)

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", should_continue)
builder.add_edge("tools", "chatbot")

graph = builder.compile()

if __name__ == "__main__":
  state = None
  while True:
    in_message = input("You: ")
    if in_message.lower() in {"quit","exit"}:
        break
    if state is None:
        state: TravelState = {
            "messages": [
                {"role": "system", "content": "You are a travel planner. Use tools when needed and provide final summarized answer"},
                {"role": "user", "content": in_message}
            ]
        }
    else:
        state["messages"].append({"role": "user", "content": in_message})

    state = graph.invoke(state)
    print("Bot:", state["messages"][-1].content)