import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app import app as langgraph_agent # Imports the compiled graph from app.py

# Initialize FastAPI app
server = FastAPI(
    title="Privacy Compliance Multi-Agent API",
    description="Production-grade API layer wrapping a LangGraph orchestration engine."
)

# Define the expected incoming JSON payload format from n8n
class AnalyzeRequest(BaseModel):
    raw_input: str

# Define the structured output the server will return
class AnalyzeResponse(BaseModel):
    risk_category: str
    compliance_score: int
    retrieved_context: str
    final_output: str

@server.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(payload: AnalyzeRequest):
    """
    Accepts text strings from n8n, runs it through the LangGraph architecture,
    and returns a structured compliance audit report.
    """
    if not payload.raw_input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    try:
        # Prepare the state dictionary for LangGraph
        initial_state = {"raw_input": payload.raw_input}
        
        # Invoke the compiled graph (this automatically triggers LangSmith tracing)
        result = langgraph_agent.invoke(initial_state)
        
        # Return the structured data payload back to the sender
        return {
            "risk_category": result.get("risk_category", "UNKNOWN"),
            "compliance_score": result.get("compliance_score", 0),
            "retrieved_context": result.get("retrieved_context", ""),
            "final_output": result.get("final_output", "")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Engine Error: {str(e)}")

# Health check route for cloud platform deployment
@server.get("/health")
def health_check():
    return {"status": "healthy"}
