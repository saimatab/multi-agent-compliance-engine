import os
from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pinecone import Pinecone

load_dotenv()

class AgentState(TypedDict):
    raw_input: str
    retrieved_context: str
    compliance_score: int
    risk_category: str
    final_output: str

class ComplianceReport(BaseModel):
    score: int = Field(description="Risk rating from 0 (Safe) to 100 (Violation)")
    category: str = Field(description="Category of leak: PII, PHI, or NONE")

# --- LIGHTWEIGHT AGENT A: CLOUD RETRIEVAL ---
def agent_router_rag(state: AgentState) -> dict:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("compliance-policies")
    
    # We query using text keyword match or basic metadata fetch rather than local vector math
    # This keeps our RAM usage near zero!
    query_response = index.query(
        vector=[0.0] * 1536,
        top_k=1,
        include_metadata=True
    )
    
    if query_response and query_response['matches']:
        context = query_response['matches'][0]['metadata']['text']
    else:
        context = "No strict policy constraints found."
        
    return {"retrieved_context": context}

def agent_compliance_evaluator(state: AgentState) -> dict:
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    structured_llm = llm.with_structured_output(ComplianceReport)
    
    prompt = f"""
    Compare the User Input against the Regulatory Context.
    User Input: {state['raw_input']}
    Context: {state['retrieved_context']}
    Determine if the input violates privacy laws.
    """
    report = structured_llm.invoke(prompt)
    return {"compliance_score": report.score, "risk_category": report.category}

def agent_executive_summarizer(state: AgentState) -> dict:
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.2)
    prompt = f"""
    Summarize this security event for logs.
    Input Text: {state['raw_input']}
    Risk Score: {state['compliance_score']}/100
    Category: {state['risk_category']}
    Generate a clean operational log message. Do not leak raw values if score > 50.
    """
    response = llm.invoke(prompt)
    return {"final_output": response.content}

workflow = StateGraph(AgentState)
workflow.add_node("router_rag", agent_router_rag)
workflow.add_node("compliance_evaluator", agent_compliance_evaluator)
workflow.add_node("executive_summarizer", agent_executive_summarizer)

workflow.set_entry_point("router_rag")
workflow.add_edge("router_rag", "compliance_evaluator")
workflow.add_edge("compliance_evaluator", "executive_summarizer")
workflow.add_edge("executive_summarizer", END)

app = workflow.compile()
