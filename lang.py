from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# -----------------------
# 1️⃣ Define Graph State
# -----------------------

class AgentState(TypedDict):
    prompt: str
    classification: str
    tool_result: str
    final_answer: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -----------------------
# 2️⃣ Classifier Node
# -----------------------

def classify_node(state: AgentState):
    prompt = state["prompt"]

    response = llm.invoke([
        SystemMessage(content="""
        Classify the user query into anyone of the
          - math
          - explain

        Only return the label.
        """),
        HumanMessage(content=prompt)
    ])

    return {"classification": response.content.strip().lower()}


# -----------------------
# 3️⃣ Tool Nodes
# -----------------------

def math_tool(state: AgentState):
    response = llm.invoke([
        SystemMessage(content="""
        You are a math expert. Solve the math problem.
        """),
        HumanMessage(content=state["prompt"])
    ])
    return {"tool_result": f"Performed math for: {response.content}"}

def explain_tool(state: AgentState):
    response = llm.invoke([
        SystemMessage(content="""
        You are an expert in explaining complex topics in a simple and understandable way.
        """),
        HumanMessage(content=state["prompt"])
    ])
    return {"tool_result": f"Explanation for: {response.content}"}

# -----------------------
# 4️⃣ Router Logic
# -----------------------

def route_tools(state: AgentState) -> Literal[
    "math_tool",
    "explain_tool"
]:
    return state["classification"]


# -----------------------
# 5️⃣ Summarizer Node
# -----------------------

def summarize_node(state: AgentState):
    return {"final_answer": f"The final answer is {state['tool_result']}"}


# -----------------------
# 6️⃣ Build Graph
# -----------------------

builder = StateGraph(AgentState)

builder.add_node("classify", classify_node)
builder.add_node("math", math_tool)
builder.add_node("explain", explain_tool)
builder.add_node("summarize", summarize_node)

builder.set_entry_point("classify")

builder.add_conditional_edges(
    "classify",
    route_tools
)

builder.add_edge("math", "summarize")
builder.add_edge("explain", "summarize")

builder.add_edge("summarize", END)

graph = builder.compile()

# -----------------------
# 7️⃣ Run It
# -----------------------

result = graph.invoke({
    "prompt": "What is the square root of 16?"
})

print(result["final_answer"])
