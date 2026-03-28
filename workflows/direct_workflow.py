import logging
from typing import Any, Dict, List, Optional
from .base_workflow import BaseWorkflow
from agents.execution_agent import ExecutionAgent

logger = logging.getLogger(__name__)

class DirectWorkflow(BaseWorkflow):
    """
    A simple pass-through workflow that sends the task directly to an LLM
    via an ExecutionAgent, without any planning or complex coordination.
    """
    
    def __init__(self, agents: Dict[str, Any], tools: List[Any] = None):
        super().__init__(agents=agents, tools=tools, workflow_name="DirectWorkflow")
        if "executor" not in self.agents:
            raise ValueError("DirectWorkflow requires an 'executor' agent.")

    def run(self, task: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Runs the direct completion.
        """
        logger.info(f"[{self.workflow_name}] Starting direct task: {task[:50]}...")
        
        executor: ExecutionAgent = self.get_agent("executor")
        
        # Capture metrics
        metadata = {}
        result = executor.execute_step(task, metadata=metadata)
        
        return {
            "status": "success",
            "results": [{"step": task, "result": result}],
            "execution_metadata": {
                "total_duration": metadata.get("duration", 0),
                "total_tokens": metadata.get("usage", {}).get("total_tokens", 0),
                "usage": {
                    "input": metadata.get("usage", {}).get("input_tokens", 0),
                    "output": metadata.get("usage", {}).get("output_tokens", 0)
                }
            }
        }

    def get_mermaid(self) -> str:
        """
        Generates a Mermaid diagram for the DirectWorkflow.
        """
        return """
graph TD
    Start((Start)) --> User([User Query])
    User --> Executor{{Execution Agent}}
    Executor --> Result([Result])
    Result --> End((End))
        """.strip()

    def _to_graph(self):
        """
        Produce a langgraph representation of this pattern for orchestration.
        """
        from langgraph.graph import StateGraph, END
        # Dummy structure for visualization
        workflow = StateGraph(dict)
        workflow.add_node("user_query", lambda x: x)
        workflow.add_node("execution_agent", lambda x: x)
        workflow.set_entry_point("user_query")
        workflow.add_edge("user_query", "execution_agent")
        workflow.add_edge("execution_agent", END)
        return workflow.compile()
