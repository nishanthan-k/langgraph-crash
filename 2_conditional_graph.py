from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END

class AmountConverter(TypedDict):
  total_usd: float
  total_eur: float
  total_inr: float
  amount_type: Literal["usd", "eur"]
  interest_rate: float

def target_currency(state: AmountConverter):
  return state["amount_type"] 

def calculate_usd_interest(state: AmountConverter):
  state["total_usd"] = state["total_usd"] * (1 + state["interest_rate"])
  return state

def calculate_eur_interest(state: AmountConverter):
  state["total_eur"] = state["total_eur"] * (1 + state["interest_rate"])
  return state

def convert_usd_to_inr(state: AmountConverter):
  usd_to_inr_rate = 82.0
  state["total_inr"] = state["total_usd"] * usd_to_inr_rate
  return state

def convert_eur_to_inr(state: AmountConverter):
  eur_to_inr_rate = 90.0
  state["total_inr"] = state["total_eur"] * eur_to_inr_rate
  return state

builder = StateGraph(AmountConverter)

builder.add_node("calculate_usd_interest", calculate_usd_interest)
builder.add_node("calculate_eur_interest", calculate_eur_interest)
builder.add_node("convert_usd_to_inr", convert_usd_to_inr)
builder.add_node("convert_eur_to_inr", convert_eur_to_inr)

builder.add_conditional_edges(START, target_currency, {
  "usd": "calculate_usd_interest",
  "eur": "calculate_eur_interest"
})

builder.add_edge("calculate_usd_interest", "convert_usd_to_inr")
builder.add_edge("calculate_eur_interest", "convert_eur_to_inr")
builder.add_edge(["convert_usd_to_inr", "convert_eur_to_inr"], END)

graph = builder.compile()

if __name__ == "__main__":
  initial_state = AmountConverter(total_usd=0, total_eur=1000.0, total_inr=0.0, amount_type="eur", interest_rate=0.05)
  final_state = graph.invoke(initial_state)
  print(final_state)