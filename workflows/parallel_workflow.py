import logging
import concurrent.futures
from typing import Any, Dict, List, Union
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
        
    def run(self, task: Union[str, List[str]], max_workers: int = 5) -> Dict[str, Any]:
        """
        Runs multiple tasks in parallel.
        
        Args:
            task: A single task string or a list of independent task descriptions.
            max_workers: Maximum number of threads for parallel execution.
            
        Returns:
            A dictionary containing the results of each task.
        """
        if isinstance(task, str):
            tasks = [task]
        else:
            tasks = task

        logger.info(f"[{self.workflow_name}] Starting {len(tasks)} parallel tasks.")
        executor_agent: ExecutionAgent = self.get_agent("executor")
        
        results = {}
        
        def execute_wrapper(task: str) -> Any:
            # Captures metrics for each individual parallel task
            m = {}
            res = executor_agent.execute_step(task, metadata=m)
            return res, m
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as thread_executor:
            # Submit all tasks
            future_to_task = {thread_executor.submit(execute_wrapper, task): task for task in tasks}
            
            # Collect results
            total_duration = 0.0
            total_in = 0
            total_out = 0
            
            for future in concurrent.futures.as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result, m = future.result()
                    results[task_name] = result
                    
                    # Accumulate metrics
                    total_duration += m.get("duration", 0)
                    u = m.get("usage", {})
                    total_in += u.get("input_tokens", 0)
                    total_out += u.get("output_tokens", 0)
                    
                    logger.info(f"[{self.workflow_name}] Task completed: {task_name[:30]}...")
                except Exception as exc:
                    logger.error(f"[{self.workflow_name}] Task generated an exception: {task_name}. Error: {exc}")
                    results[task_name] = f"ERROR: {str(exc)}"
                    
        # Format results as a list to match the unified display logic
        formatted_results = []
        for task_str, res_str in results.items():
            formatted_results.append({
                "step": task_str,
                "result": res_str
            })

        return {
            "status": "completed",
            "results": formatted_results,
            "execution_metadata": {
                "total_duration": total_duration,
                "total_tokens": total_in + total_out,
                "usage": {"input": total_in, "output": total_out}
            }
        }
