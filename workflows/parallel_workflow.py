import logging
import concurrent.futures
from typing import Any, Dict, List
from .base_workflow import BaseWorkflow
from agents.execution_agent import ExecutionAgent

logger = logging.getLogger(__name__)

class ParallelWorkflow(BaseWorkflow):
    """
    A workflow designed to execute multiple independent tasks concurrently
    using the Execution agent. This simulates the Parallelization pattern.
    """
    
    def __init__(self, agents: Dict[str, Any], tools: List[Any] = None):
        super().__init__(agents=agents, tools=tools, workflow_name="ParallelWorkflow")
        
        assert "executor" in self.agents, "ParallelWorkflow requires an 'executor' agent."
        
    def run(self, tasks: List[str], max_workers: int = 5) -> Dict[str, Any]:
        """
        Runs multiple tasks in parallel.
        
        Args:
            tasks: A list of independent task descriptions.
            max_workers: Maximum number of threads for parallel execution.
            
        Returns:
            A dictionary containing the results of each task.
        """
        logger.info(f"[{self.workflow_name}] Starting {len(tasks)} parallel tasks.")
        executor_agent: ExecutionAgent = self.get_agent("executor")
        
        results = {}
        
        def execute_wrapper(task: str) -> str:
            # Simple wrapper to just call execute_step
            return executor_agent.execute_step(task)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as thread_executor:
            # Submit all tasks
            future_to_task = {thread_executor.submit(execute_wrapper, task): task for task in tasks}
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results[task] = result
                    logger.info(f"[{self.workflow_name}] Task completed: {task[:30]}...")
                except Exception as exc:
                    logger.error(f"[{self.workflow_name}] Task generated an exception: {task}. Error: {exc}")
                    results[task] = f"ERROR: {str(exc)}"
                    
        return {
            "status": "completed",
            "results": results
        }
