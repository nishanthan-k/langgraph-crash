import os
import json
from dotenv import load_dotenv
from typing import Literal, TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = init_chat_model(
    model="MiniMaxAI/MiniMax-M2.5:novita",
    model_provider="openai",
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

class ExpenseDetails(TypedDict):
    name: str
    amount: float
    description: str

class ExpenseStatus(TypedDict):
    status: Literal["Approved", "Rejected"]
    reason: str

class ExpenseState(TypedDict):
    request: ExpenseDetails
    require_manager_approval: bool
    response: ExpenseStatus

def validate_expense(state: ExpenseState) -> ExpenseState:
    amount = state["request"]["amount"]
    state["require_manager_approval"] = amount > 5000
    return state

def can_auto_approve(state: ExpenseState) -> bool:
    return state["require_manager_approval"]

def auto_approve(state: ExpenseState) -> ExpenseState:
    name = state["request"]["name"]
    state["response"]["status"] = "Approved"
    state["response"]["reason"] = f"Hi {name}, your expense has been auto-approved."
    return state

def manager_approval(state: ExpenseState) -> ExpenseState:
    system_prompt = """You are a manager reviewing an expense request.
    Respond ONLY in JSON:

    {
        "status": "Approved" or "Rejected",
        "reason": string
    }
    """

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(state["request"]))
        ])

        response_json = json.loads(response.content)

        print("\n")
        print(response_json)
        print("\n")

        state["response"]["status"] = response_json["status"]
        state["response"]["reason"] = response_json["reason"]

        return state

    except Exception as e:
        state["response"] = {
            "status": "Rejected",
            "reason": f"Error processing request: {str(e)}"
        }
        return state

def finalize(state: ExpenseState) -> None:
    # We can get the employee and manager email and send an email here
    # Can initiate an cron job to an manager to send notification after 12hrs when employee submits an request
    pass

builder = StateGraph(ExpenseState)

builder.add_node("validate_expense", validate_expense)
builder.add_node("auto_approve", auto_approve)
builder.add_node("manager_approval", manager_approval)
builder.add_node("finalize", finalize)

builder.add_edge(START, "validate_expense")
builder.add_conditional_edges("validate_expense", can_auto_approve, {
    True: "manager_approval",
    False: "auto_approve"
})
builder.add_edge(["auto_approve", "manager_approval"], "finalize")

graph = builder.compile()

if __name__ == "__main__":
    expense_request = ExpenseDetails(
        name="Alice",
        amount=6000,
        description="Team dinner"
    )
    initial_state = ExpenseState(request=expense_request, response={"status": "", "reason": ""})
    final_state = graph.invoke(initial_state)
    print(final_state)