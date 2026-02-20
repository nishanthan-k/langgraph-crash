from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class AmountConverter(TypedDict):
  total_usd: float
  total_inr: float
  interest_rate: float

def calculate_interest(state: AmountConverter):
  state["total_usd"] = state["total_usd"] * (1 + state["interest_rate"])
  return state

def convert_usd_to_inr(state: AmountConverter):
  inr_rate = 82.0
  state["total_inr"] = state["total_usd"] * inr_rate
  return state


builder = StateGraph(AmountConverter)

builder.add_node(calculate_interest)
builder.add_node(convert_usd_to_inr)

builder.add_edge(START, "calculate_interest")
builder.add_edge("calculate_interest", "convert_usd_to_inr")
builder.add_edge("convert_usd_to_inr", END)

graph = builder.compile()

if __name__ == "__main__":
  initial_state = AmountConverter(total_usd=1000.0, total_inr=0.0, interest_rate=0.05)
  final_state = graph.invoke(initial_state)
  print(final_state)
