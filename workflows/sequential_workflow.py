import logging
from typing import Any, Dict, List
from .base_workflow import BaseWorkflow
from agents.planning_agent import PlanningAgent
from agents.execution_agent import ExecutionAgent
from agents.monitoring_agent import MonitoringAgent

logger = logging.getLogger(__name__)

class SequentialWorkflow(BaseWorkflow):
    """
    A workflow that uses a PlanningAgent to break down the task,
    then evaluates each step sequentially using ExecutionAgent and MonitoringAgent.
    """
    
    def __init__(self, agents: Dict[str, Any], tools: List[Any] = None):
        super().__init__(agents=agents, tools=tools, workflow_name="SequentialWorkflow")
        
        # Ensure the right types of agents were passed
        assert "planner" in self.agents, "SequentialWorkflow requires a 'planner' agent."
        assert "executor" in self.agents, "SequentialWorkflow requires an 'executor' agent."
        assert "monitor" in self.agents, "SequentialWorkflow requires a 'monitor' agent."

    def run(self, task: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Runs the full sequential orchestration.
        """
        logger.info(f"[{self.workflow_name}] Starting task: {task}")
        
        planner: PlanningAgent = self.get_agent("planner")
        executor: ExecutionAgent = self.get_agent("executor")
        monitor: MonitoringAgent = self.get_agent("monitor")
        
        # 1. Generate the plan
        plan_steps = planner.generate_plan(task)
        logger.info(f"[{self.workflow_name}] Generated {len(plan_steps)} steps.")
        
        results = []
        context = ""
        
        # 2. Execute sequentially
        for i, step in enumerate(plan_steps, 1):
            logger.info(f"[{self.workflow_name}] --- Step {i}/{len(plan_steps)} ---")
            
            success = False
            attempts = 0
            step_result = ""
            feedback = ""
            
            while not success and attempts <= max_retries:
                # Add failure context if retrying
                current_context = context
                
                # Compress context if tools are available (e.g., active_compressor)
                if current_context and getattr(self, 'tools', None):
                    compressor = self.tools[0]
                    if hasattr(compressor, 'invoke'):
                        current_context = compressor.invoke(current_context)
                    elif hasattr(compressor, '_run'):
                        current_context = compressor._run(current_context)
                        
                if attempts > 0:
                    current_context += f"\nPrevious attempt failed. Feedback: {feedback}"
                
                # Execute
                step_result = executor.execute_step(step, context=current_context)
                
                # Monitor/Evaluate
                eval_data = monitor.evaluate(objective=step, result=step_result)
                success = eval_data.get("success", False)
                feedback = eval_data.get("feedback", "")
                
                attempts += 1
                if not success:
                    logger.warning(f"[{self.workflow_name}] Step failed. Retrying ({attempts}/{max_retries}). Feedback: {feedback}")
                    
            if not success:
                logger.error(f"[{self.workflow_name}] Pipeline aborted. Step failed after {max_retries} retries.")
                return {
                    "status": "failed",
                    "failed_step": step,
                    "completed_results": results
                }
            
            # Append success result to context for the next step
            logger.info(f"[{self.workflow_name}] Step {i} succeeded.")
            results.append({"step": step, "result": step_result})
            context += f"\n[Step {i} Result]: {step_result}"
            
        logger.info(f"[{self.workflow_name}] Task completed successfully.")
        return {
            "status": "success",
            "completed_results": results
        }
