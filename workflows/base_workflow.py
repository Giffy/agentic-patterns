from abc import ABC, abstractmethod
import logging
import os
from typing import Any, Dict, List, Optional
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
        """
        pass
        
    @abstractmethod
    def get_mermaid(self) -> str:
        """
        Generates a Mermaid diagram string for the workflow.
        """
        pass
        
    def draw(self, output_path: Optional[str] = None) -> str:
        """
        Generates the Mermaid diagram and saves it to a file. 
        If output_path has .png extension, it tries to render it.
        """
        mermaid_code = self.get_mermaid()
        
        # Use default name if not provided (always in workflows/ folder)
        if not output_path:
            output_path = os.path.join("workflows", f"{self.workflow_name}.mmd")
            
        # Ensure folder exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save Mermaid text
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(mermaid_code)
        logger.info(f"[{self.workflow_name}] Mermaid diagram saved to {output_path}")

        # Try to save PNG
        png_path = output_path.replace(".mmd", ".png") if ".mmd" in output_path else output_path
        if not png_path.endswith(".png"):
            png_path += ".png"

        try:
            # If the workflow provides a LangGraph representation, use it to draw PNG
            if hasattr(self, "_to_graph"):
                graph = self._to_graph()
                if graph:
                    png_bytes = graph.get_graph().draw_mermaid_png()
                    with open(png_path, "wb") as f:
                        f.write(png_bytes)
                    logger.info(f"[{self.workflow_name}] Graph visualization saved to {png_path}")
        except Exception as e:
            # Fallback for systems without Graphviz/pygraphviz
            logger.warning(f"[{self.workflow_name}] Could not save graph as PNG: {e}")
            
        return mermaid_code
