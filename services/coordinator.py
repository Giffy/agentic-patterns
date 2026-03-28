import logging
from typing import Dict, Any, Union, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from services.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

class Coordinator:
    """
    Analyzes task complexity and decides which architecture to use.
    """
    
    def __init__(self, llm: BaseChatModel = None):
        # Default coordinator llm is cloud for higher precision, 
        # but can use Local/Hybrid depending on preference
        self.llm = llm or LLMFactory.get_llm(mode="cloud", role="coordinator")
        
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI Coordinator. Your job is to analyze the complexity of a user task and decide which execution architecture is best suited.
            
            Architectures:
            1. 'prompt_chain': Simple sequence of steps. Use for well-defined, straightforward tasks.
            2. 'parallel': Tasks that can be broken into independent sub-tasks.
            3. 'orchestrator': Complex tasks requiring planning, iterative execution, and self-correction.
            
            Complexity Score: 1-10 (1 is simplest, 10 is most complex).
            
            Return ONLY a JSON object with:
            {{
                "complexity_score": <int>,
                "architecture": <string>,
                "reasoning": <string>
            }}
            """),
            ("human", "Analyze this task: {task}")
        ])

    def select_architecture(self, task: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Determines the best architecture for the task.
        """
        import time
        import json
        
        chain = self.routing_prompt | self.llm
        
        try:
            start_time = time.perf_counter()
            # Capture metadata for this specific call
            m = {}
            response = chain.invoke({"task": task})
            duration = time.perf_counter() - start_time
            
            # Extract usage metadata
            usage = getattr(response, "usage_metadata", {})
            if not usage:
                meta = getattr(response, "response_metadata", {})
                usage = {
                    "input_tokens": meta.get("prompt_eval_count", meta.get("token_usage", {}).get("prompt_tokens", 0)),
                    "output_tokens": meta.get("eval_count", meta.get("token_usage", {}).get("completion_tokens", 0)),
                }
            
            tokens_info = f"tokens={usage.get('input_tokens', 0) + usage.get('output_tokens', 0)}"
            tokens_info += f" (in={usage.get('input_tokens', 0)}, out={usage.get('output_tokens', 0)})"
            
            # Populate caller's metadata
            if metadata is not None:
                metadata["duration"] = duration
                metadata["usage"] = usage

            # Handle potential markdown formatting in response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            decision = json.loads(content)
            logger.info(f"[Coordinator] Decision for task: {decision['architecture']} (Score: {decision['complexity_score']}) | duration={duration:.2f}s, {tokens_info}")
            return decision
        except Exception as e:
            logger.error(f"[Coordinator] Error during architecture selection: {str(e)}")
            # Fallback to orchestrator for safety
            return {
                "complexity_score": 5,
                "architecture": "orchestrator",
                "reasoning": "Fallback due to coordinator error."
            }
