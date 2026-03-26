from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class BaseWorkflow(ABC):
    """
    Base architecture for workflows.
    Allows passing agents and optional tools/skills to be coordinated.
    """
    
    def __init__(
        self, 
        agents: Dict[str, BaseAgent], 
        tools: Optional[List[Any]] = None,
        workflow_name: str = "BaseWorkflow"
    ):
        """
        Initializes the workflow with required agents and available tools.
        
        Args:
            agents: A dictionary of agent name to agent instance.
                    Example: {"planner": PlanningAgent(...), "executor": ExecutionAgent(...)}
            tools: A list of tools or skills the workflow components can use.
            workflow_name: The name of the workflow.
        """
        self.agents = agents
        self.tools = tools or []
        self.workflow_name = workflow_name
        
    def get_agent(self, name: str) -> BaseAgent:
        """
        Retrieves a specified agent by name. Raises error if missing.
        """
        if name not in self.agents:
            raise ValueError(f"Agent '{name}' is not registered in this workflow.")
        return self.agents[name]
        
    @abstractmethod
    def run(self, task: str, **kwargs: Any) -> Any:
        """
        The main method to run the workflow. Must be implemented by subclasses.
        
        Args:
            task: The overarching task or objective.
            
        Returns:
            The final output of the workflow.
        """
        pass
