import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = init_chat_model(
    model="MiniMaxAI/MiniMax-M2.5:novita",
    model_provider="openai",
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def get_stock_price(stock: str) -> float | None:
    """Get the stock price by passing the stock name. If you receive None/null then we currently didn't have that stock details"""
    stock = stock.upper()
    stock_prices = {
        "APPL": 101.23,
        "GOOG": 202.31,
        "AMZN": 303.12,
    }
    return stock_prices.get(stock)

tools = [get_stock_price]
tool_node = ToolNode(tools)

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return { "messages": [response] }

builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", tool_node)

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)

graph = builder.compile()

if __name__ == "__main__":
    state = None
    while True:
      in_message = input("You: ")
      if in_message.lower() in { 'quit', 'exit' }:
          break
      
      if state is None:
        state: State = {
            "messages": [
                { "role": "system", "content": "You are an stock price auditor having an list of stock prices. Return the stock price of the stock which user asks if you didn't find that stock then give the user an clear message"
                },
                { "role": "user", "content": in_message}
            ]
        }
      else:
        state["messages"].append({ "role": "user", "content": in_message})

      state = graph.invoke(state)
      print("Bot: ", state["messages"][-1].content)
        