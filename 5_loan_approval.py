from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class LoanReport(TypedDict):
  credit_check: Literal["Pass", "Fail", "In-Progress"]
  threshold_income_check: Literal["Pass", "Fail", "In-Progress"]
  high_risk_loan_check: Literal["Yes", "No", "In-Progress"]
  loan_status: Literal["Approved", "Rejected", "In-Progress"]
  reason: str

class LoanState(TypedDict):
  loan_amount: float
  credit_score: float
  income: float
  threshold_amount: float
  high_risk_loan: bool
  report: LoanReport
  current_step: Literal["credit_check", "threshold_income_check", "high_risk_loan_check", "finalize", "end"]

def check_credit_score(state: LoanState) -> bool:
  status = state['credit_score'] >= 650
  state["report"]["credit_check"] = "Pass" if status else "Fail" 
  return status

def check_threshold_income(state: LoanState) -> bool:
  threshold_amount = state['income'] * (30 / 100)
  state["threshold_amount"] = threshold_amount
  status = state["income"] > threshold_amount
  state["report"]["threshold_income_check"] = "Pass" if status else "Fail" 
  return status

def check_high_risk_loan(state: LoanState) -> bool:
  income = state["income"]
  loan_amout = state["loan_amount"]
  step_diff = loan_amout // income
  status = step_diff >= 5
  state["high_risk_loan"] = status
  state["report"]["high_risk_loan_check"] = "Pass" if status else "Fail" 
  return status

def finalize(state: LoanState) -> LoanState:
  # We can get the employee and manager email and send an email here
  # Can initiate an cron job to an manager to send notification after 12hrs when employee submits an request
  report = state["report"]
  credit_check = report["credit_check"]
  threshold_income_check = report["threshold_income_check"]
  high_risk_loan_check = report["high_risk_loan_check"]

  checks = [credit_check, threshold_income_check, high_risk_loan_check]
  failed_cases = [check for check in checks if check != "Pass"]


  if len(failed_cases) == 0:
    state["report"]["loan_status"] = "Approved"
  else:
    state["report"]["loan_status"] = "Rejected"
  
  state["report"]["reason"] = f"Dear Applicant, your loan request has been {state['report']['loan_status']}"
  return state

def validate_loan(state: LoanState) -> LoanState:
  current_step = state["current_step"]
  
  match current_step:
    case "credit_check":
      check_credit_score(state)
      state["current_step"] = "threshold_income_check"
      return state
    
    case "threshold_income_check":
      check_threshold_income(state)
      state["current_step"] = "high_risk_loan_check"
      return state
    
    case "high_risk_loan_check":
      check_high_risk_loan(state)
      state["current_step"] = "finalize"
      return state
    
    case "finalize":
      finalize(state)
      state["current_step"] = "end"
      return state
    
  return state


def router(state: LoanState) -> bool:
  current_step = state["current_step"]
  return current_step == "end" 

builder = StateGraph(LoanState)

builder.add_node("validate_loan", validate_loan)

builder.add_edge(START, "validate_loan")
builder.add_conditional_edges("validate_loan", router, {
  True: END,
  False: "validate_loan"
})

graph = builder.compile()


if __name__ == "__main__":
    loan_report = LoanReport(
      credit_check="In-Progress",
      high_risk_loan_check="In-Progress",
      loan_status="In-Progress",
      reason=""
    )
    loan_request = LoanState(
      loan_amount= 500000.00,
      credit_score=720,
      income=45000.00,
      current_step="credit_check",
      report=loan_report
    )
    # initial_state = ExpenseState(request=expense_request, response={"status": "", "reason": ""})
    final_state = graph.invoke(loan_request)
    print(final_state)