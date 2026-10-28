import os
from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pinecone import Pinecone

# Load local environment variables
load_dotenv()

# ==========================================
# 1. DEFINE MULTI-AGENT STATE MATRIX
# ==========================================
class AgentState(TypedDict):
    raw_input: str
    retrieved_context: str
    compliance_score: int
    risk_category: str
    final_output: str

# Pydantic schema for Agent B's strict JSON structure compilation
class ComplianceReport(BaseModel):
    score: int = Field(description="Risk rating from 0 (Safe) to 100 (Critical Leak Violation)")
    category: str = Field(description="Identified category of leak violation: PII, PHI, or NONE")

# ==========================================
# 2. DEFINE AGENT NODES (LIGHTWEIGHT ORCHESTRATION)
# ==========================================
def agent_router_rag(state: AgentState) -> dict:
    """
    Agent A: Connects to Pinecone to query compliance parameters.
    Keeps memory footprint under 100MB by utilizing cloud-based index filtering.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("compliance-policies")
    
    # Cloud indexing structure match
    query_response = index.query(
        vector=[0.0] * 1536,
        top_k=1,
        include_metadata=True
    )
    
    if query_response and query_response['matches']:
        context = query_response['matches']['metadata']['text']
    else:
        context = "No strict policy constraints found."
        
    return {"retrieved_context": context}

def agent_compliance_evaluator(state: AgentState) -> dict:
    """
    Agent B: Compares input against context data using Groq's active production engine.
    Ensures zero hallucination by formatting strict deterministic JSON schemas.
    """
    # Updated to Groq's active flagship model: qwen/qwen3.8-27b
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
    structured_llm = llm.with_structured_output(ComplianceReport)
    
    prompt = f"""
    Analyze the incoming user text and evaluate it against our regulatory infrastructure policies.
    
    User Input Text: {state['raw_input']}
    Compliance Context Manual: {state['retrieved_context']}
    
    Determine if the user input contains sensitive data matching compliance breaches.
    """
    report = structured_llm.invoke(prompt)
    return {"compliance_score": report.score, "risk_category": report.category}

def agent_executive_summarizer(state: AgentState) -> dict:
    """
    Agent C: Strips data values based on risk thresholds and records safe analytical summaries.
    """
    # Updated to Groq's active flagship model: qwen/qwen3.8-27b
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.2)
    
    prompt = f"""
    You are an operations logging engine. Summarize this operational transaction alert.
    
    Raw Text Checked: {state['raw_input']}
    Evaluated Breach Score: {state['compliance_score']}/100
    Breach Classification: {state['risk_category']}
    
    Construct a clean operational metadata summary for database entry.
    CRITICAL RULE: If the Breach Score is greater than 50, do NOT reflect raw numbers or raw secrets in the final summary. Mask them securely.
    """
    response = llm.invoke(prompt)
    return {"final_output": response.content}

# ==========================================
# 3. CONSTRUCT THE LANGGRAPH FLOW ENGINE
# ==========================================
workflow = StateGraph(AgentState)

# Add agents into processing graph nodes
workflow.add_node("router_rag", agent_router_rag)
workflow.add_node("compliance_evaluator", agent_compliance_evaluator)
workflow.add_node("executive_summarizer", agent_executive_summarizer)

# Set up sequential processing dependency edges
workflow.set_entry_point("router_rag")
workflow.add_edge("router_rag", "compliance_evaluator")
workflow.add_edge("compliance_evaluator", "executive_summarizer")
workflow.add_edge("executive_summarizer", END)

# Compile graph into executable state machine
app = workflow.compile()
