import time
import logging
from typing import Any, Dict, List, Optional, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent:
    """
    A foundational agent class that can be reused and extended by specific agent roles.
    Designed to be easily instantiated and called by workflows or scripts.
    """
    
    def __init__(
        self, 
        llm: BaseChatModel, 
        system_prompt: str = "You are a helpful AI assistant.",
        agent_name: str = "BaseAgent"
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.agent_name = agent_name
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])
        self.last_metrics = {
            "duration": 0.0,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }
    
    def invoke(self, user_input: str, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """
        Main execution method for the agent.
        
        Args:
            user_input: The main task or prompt for the agent.
            metadata: Optional dictionary to capture performance metrics (duration, usage).
            **kwargs: Additional variables to format the prompt.
            
        Returns:
            The agent's response.
        """
        logger.info(f"[{self.agent_name}] Invoked with input: {user_input[:50]}...")
        
        start_time = time.perf_counter()
        chain = self.prompt_template | self.llm
        
        try:
            response = chain.invoke({"input": user_input, **kwargs})
            duration = time.perf_counter() - start_time
            
            # Extract usage metadata (standard for newer LangChain)
            usage = getattr(response, "usage_metadata", {})
            if not usage:
                # Fallback to response_metadata (Ollama often uses this)
                meta = getattr(response, "response_metadata", {})
                usage = {
                    "input_tokens":  meta.get("prompt_eval_count", meta.get("token_usage", {}).get("prompt_tokens", 0)),
                    "output_tokens": meta.get("eval_count", meta.get("token_usage", {}).get("completion_tokens", 0)),
                }
                usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
            
            # Formulate token log string
            tokens_info = f"tokens={usage.get('total_tokens', 0)}"
            if usage.get("input_tokens") or usage.get("output_tokens"):
                tokens_info += f" (in={usage.get('input_tokens', 0)}, out={usage.get('output_tokens', 0)})"

            logger.info(f"[{self.agent_name}] Invocation finished: duration={duration:.2f}s, {tokens_info}")
            
            # Store metadata if requested
            if metadata is not None:
                metadata["duration"] = duration
                metadata["usage"] = usage
            
            return response.content
        except Exception as e:
            logger.error(f"[{self.agent_name}] Error during invocation: {str(e)}")
            raise
