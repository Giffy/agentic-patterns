import json
import logging
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class PlanningAgent(BaseAgent):
    """
    Agent responsible for breaking down a high-level task into a clear, 
    sequential list of steps to be executed.
    """
    
    def __init__(self, llm: BaseChatModel, **kwargs):
        system_prompt = (
            "You are an expert planning agent. Your task is to analyze the user's request "
            "and create a structured, step-by-step plan to achieve the goal. "
            "Return ONLY a valid JSON object containing a 'plan' key, which is a list of strings "
            "describing each step. "
            "Example: {{\\\"plan\\\": [\\\"Step 1: Do X\\\", \\\"Step 2: Do Y\\\"]}}"
        )
        super().__init__(
            llm=llm, 
            system_prompt=kwargs.get("system_prompt", system_prompt),
            agent_name="PlanningAgent"
        )
        
    def generate_plan(self, task: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generates a plan from a given task.
        
        Args:
            task: The high-level objective.
            metadata: Optional dictionary to capture performance metrics.
            
        Returns:
            A list of strings representing the steps.
        """
        logger.info(f"[{self.agent_name}] Generating plan for task: {task[:50]}")
        
        response_text = self.invoke(task, metadata=metadata)
        
        try:
            # Clean up potential markdown formatting (e.g., ```json ... ```)
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            parsed_data = json.loads(clean_text)
            return parsed_data.get("plan", [response_text])
        except json.JSONDecodeError as e:
            logger.warning(f"[{self.agent_name}] Failed to parse JSON plan. Returning raw response as single step. Error: {e}")
            return [response_text]
