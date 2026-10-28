import os
from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from pinecone import Pinecone

load_dotenv()

# Define the shared memory graph state
class AgentState(TypedDict):
    raw_input: str
    retrieved_context: str
    compliance_score: int
    risk_category: str
    final_output: str

# Pydantic schema for Agent B's structured assessment output
class ComplianceReport(BaseModel):
    score: int = Field(description="Risk rating from 0 (Safe) to 100 (Violation)")
    category: str = Field(description="Category of leak: PII, PHI, or NONE")

# --- AGENT A: FREE LOCAL RAG ROUTER ---
def agent_router_rag(state: AgentState) -> dict:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode(state["raw_input"]).tolist()
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("compliance-policies")
    
    query_response = index.query(vector=query_vector, top_k=1, include_metadata=True)
    
    if query_response and query_response['matches']:
        context = query_response['matches'][0]['metadata']['text']
    else:
        context = "No strict policy constraints found."
        
    return {"retrieved_context": context}

# --- AGENT B: FREE COMPLIANCE AUDITOR ---
def agent_compliance_evaluator(state: AgentState) -> dict:
    # Using Llama-3 on Groq completely free
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    structured_llm = llm.with_structured_output(ComplianceReport)
    
    prompt = f"""
    Compare the User Input against the Regulatory Context.
    User Input: {state['raw_input']}
    Context: {state['retrieved_context']}
    
    Determine if the input violates privacy laws. Return your output matching the structured schema.
    """
    report = structured_llm.invoke(prompt)
    return {"compliance_score": report.score, "risk_category": report.category}

# --- AGENT C: OPERATIONS SUMMARIZER ---
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

# --- BUILD THE STATE ARCHITECTURE ---
workflow = StateGraph(AgentState)

workflow.add_node("router_rag", agent_router_rag)
workflow.add_node("compliance_evaluator", agent_compliance_evaluator)
workflow.add_node("executive_summarizer", agent_executive_summarizer)

workflow.set_entry_point("router_rag")
workflow.add_edge("router_rag", "compliance_evaluator")
workflow.add_edge("compliance_evaluator", "executive_summarizer")
workflow.add_edge("executive_summarizer", END)

app = workflow.compile()

if __name__ == "__main__":
    test_payload = {"raw_input": "Customer support leak ticket. User phone is 555-999-1234."}
    result = app.invoke(test_payload)
    print("\n--- FREE MULTI-AGENT EXECUTION SUCCESS ---")
    print(f"Risk Category: {result['risk_category']}")
    print(f"Risk Score: {result['compliance_score']}")
    print(f"Log: {result['final_output']}")
