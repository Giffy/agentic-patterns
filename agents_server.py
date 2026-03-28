import time
import logging
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from unified_entry_point import UnifiedAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_server")

app = FastAPI(title="Agentic Patterns API", version="1.0.0")

# --- OpenAI Compatibility Models ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionUsage

# --- Helper Logic ---

def map_model_to_arch(model_name: str) -> str:
    """
    Maps the model identifier from the IDE to a specific agent architecture.
    """
    mapping = {
        "agent-router": "router",
        "agent-orchestrator": "orchestrator",
        "agent-sequential": "prompt_chain",
        "agent-parallel": "parallel",
        "agent-direct": "direct"
    }
    # Default to router if not specified
    return mapping.get(model_name.lower(), "router")

# --- Endpoints ---

@app.get("/")
async def root():
    return {"status": "ok", "message": "Agentic Patterns Server is running."}

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    logger.info(f"Received completion request for model: {request.model}")
    
    # 1. Extract the last message as the task
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")
    
    user_task = request.messages[-1].content
    
    # 2. Determine architecture based on model name
    arch = map_model_to_arch(request.model)
    
    # 3. Create the agent and run
    try:
        # We use 'cloud' mode for the server by default to ensure reliability in the IDE,
        # but this could be parameterized or pulled from env.
        agent = UnifiedAgent(model_type="cloud", architecture=arch)
        result = agent.run(user_task)
        
        # 4. Extract final answer
        # The result structure varies by workflow, but we'll try to find the last result.
        final_answer = ""
        if "completed_results" in result and result["completed_results"]:
            final_answer = result["completed_results"][-1]["result"]
        elif "results" in result and result["results"]:
             final_answer = result["results"][-1]["result"]
        elif "result" in result:
             final_answer = str(result["result"])
        else:
             final_answer = "No clear result was returned from the agent logic."

        # 5. Extract metrics
        meta = result.get("execution_metadata", {})
        usage = ChatCompletionUsage(
            prompt_tokens=meta.get("usage", {}).get("input", 0),
            completion_tokens=meta.get("usage", {}).get("output", 0),
            total_tokens=meta.get("total_tokens", 0)
        )

        # 6. Build OpenAI response
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=final_answer)
                )
            ],
            usage=usage
        )
        
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Defaulting to 127.0.0.1:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
