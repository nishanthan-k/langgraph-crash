import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import add_messages, StateGraph, START, END

load_dotenv()

llm = init_chat_model(
    model="MiniMaxAI/MiniMax-M2.5:novita",
    model_provider="openai",
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

# resp = llm.invoke([{"role": "user", "content": "What is the capital of France?"}])
# print(resp.content)

class AgentState(TypedDict):
  messages: Annotated[list, add_messages]

def chatbot(state: AgentState):
  return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(AgentState)
builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

if __name__ == "__main__":
  state = None
  while True:
    in_message = input("You: ")
    if in_message.lower() in {"quit","exit"}:
        break
    if state is None:
        state: AgentState = {
            "messages": [{"role": "user", "content": in_message}]
        }
    else:
        state["messages"].append({"role": "user", "content": in_message})

    state = graph.invoke(state)
    print("Bot:", state["messages"][-1].content)