import time
import logging
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ExecutionAgent(BaseAgent):
    """
    Agent responsible for executing a specific step in a plan.
    It can be enhanced later manually with specific tools depending on the workflow.
    """
    
    def __init__(self, llm: BaseChatModel, tools: List[Any] = None, **kwargs):
        system_prompt = (
            "You are a diligent execution agent. Your task is to complete the given action step "
            "as described, providing a clear and detailed output of your work. "
            "If you are provided with context from previous steps, use it to inform your output."
        )
        super().__init__(
            llm=llm, 
            system_prompt=kwargs.get("system_prompt", system_prompt),
            agent_name="ExecutionAgent"
        )
        self.tools = tools or []
        if self.tools:
            # Bind tools to the LLM using LangGraph's native react agent graph
            try:
                self.agent_executor = create_react_agent(self.llm, tools=self.tools, state_modifier=self.system_prompt)
            except TypeError:
                # Fallback for older langgraph versions where state_modifier/messages_modifier doesn't exist
                self.agent_executor = create_react_agent(self.llm, tools=self.tools)
        else:
            self.agent_executor = None
        
    def execute_step(self, step_description: str, context: str = "", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Executes a single step given optional context.
        
        Args:
            step_description: The task to perform.
            context: Previous context for the task.
            metadata: Optional dictionary to capture performance metrics.
        """
        logger.info(f"[{self.agent_name}] Executing step: {step_description[:50]}")
        
        input_prompt = f"Step to Execute: {step_description}"
        if context:
            input_prompt += f"\n\nContext:\n{context}"
            
        if self.agent_executor:
            try:
                # Include system prompt directly in the user message to ensure it's respected across all langgraph versions
                full_prompt = f"{self.system_prompt}\n\n{input_prompt}"
                
                start_time = time.perf_counter()
                response = self.agent_executor.invoke({"messages": [("user", full_prompt)]})
                duration = time.perf_counter() - start_time
                
                # Aggregate tokens across the graph execution steps
                total_in = 0
                total_out = 0
                for msg in response.get("messages", []):
                    # Check for usage metadata in each message
                    usage = getattr(msg, "usage_metadata", {})
                    if not usage:
                        meta = getattr(msg, "response_metadata", {})
                        usage = {
                            "input_tokens": meta.get("prompt_eval_count", meta.get("token_usage", {}).get("prompt_tokens", 0)),
                            "output_tokens": meta.get("eval_count", meta.get("token_usage", {}).get("completion_tokens", 0)),
                        }
                    total_in += usage.get("input_tokens", 0)
                    total_out += usage.get("output_tokens", 0)
                
                logger.info(f"[{self.agent_name}] Graph execution finished: duration={duration:.2f}s, tokens={total_in + total_out} (in={total_in}, out={total_out})")
                
                # Store metadata if requested
                if metadata is not None:
                    metadata["duration"] = duration
                    metadata["usage"] = {
                        "input_tokens": total_in,
                        "output_tokens": total_out,
                        "total_tokens": total_in + total_out
                    }
                
                return response["messages"][-1].content
            except Exception as e:
                logger.error(f"[{self.agent_name}] AgentExecutor failed: {e}")
                
        return self.invoke(input_prompt, metadata=metadata)
