import json
import logging
from typing import Any, Dict, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class EvaluatorAgent(BaseAgent):
    """
    Agent responsible for evaluating the output of an execution step 
    against its initial objective to determine success and provide feedback.
    """
    
    def __init__(self, llm: BaseChatModel, **kwargs):
        system_prompt = (
            "You are a strict monitoring and evaluation agent. "
            "Your task is to compare the Original Objective of a step "
            "with the Actual Output produced by an execution agent. "
            "Determine if the objective was successfully met. "
            "Return ONLY a valid JSON object with two keys: "
            "1. 'success' (boolean: true or false) "
            "2. 'feedback' (string: explanation of why it succeeded or failed, and how to fix if failed). "
            "Example: {{\\\"success\\\": true, \\\"feedback\\\": \\\"Objective met completely.\\\"}}"
        )
        super().__init__(
            llm=llm, 
            system_prompt=kwargs.get("system_prompt", system_prompt),
            agent_name="EvaluatorAgent"
        )
        
    def evaluate(self, objective: str, result: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluates the execution result against the objective.
        
        Args:
            objective: What the step was supposed to do.
            result: What the execution agent actually did.
            metadata: Optional dictionary to capture performance metrics.
            
        Returns:
            Dictionary with 'success' (bool) and 'feedback' (str).
        """
        logger.info(f"[{self.agent_name}] Evaluating objective: {objective[:50]}")
        
        evaluation_prompt = (
            f"Original Objective:\n{objective}\n\n"
            f"Actual Output:\n{result}\n\n"
            "Evaluate if the Actual Output successfully meets the Original Objective."
        )
        
        response_text = self.invoke(evaluation_prompt, metadata=metadata)
        
        try:
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            parsed_data = json.loads(clean_text)
            
            # Ensure required keys exist
            success = bool(parsed_data.get("success", False))
            feedback = str(parsed_data.get("feedback", "No feedback provided by evaluating agent."))
            
            return {"success": success, "feedback": feedback}
            
        except json.JSONDecodeError as e:
            logger.warning(f"[{self.agent_name}] Failed to parse JSON evaluation. Returning failure. Error: {e}")
            return {
                "success": False,
                "feedback": f"Failed to parse monitoring response: {response_text}"
            }
