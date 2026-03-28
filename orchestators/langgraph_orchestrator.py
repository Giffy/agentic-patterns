import logging
import operator
from typing import Annotated, Any, Dict, List, Sequence, TypedDict
from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph, END
from agents.planning_agent import PlanningAgent
from agents.execution_agent import ExecutionAgent
from agents.evaluator_agent import EvaluatorAgent

logger = logging.getLogger(__name__)

# Define the state for the LangGraph
class OrchestratorState(TypedDict):
    """
    The state structure passed between nodes in the LangGraph orchestrator.
    """
    task: str
    plan: List[str]
    current_step_index: int
    context: str
    results: Annotated[List[Dict[str, Any]], operator.add]
    attempts: int
    max_retries: int
    status: str  # e.g., "planning", "executing", "evaluating", "success", "failed"
    total_duration: float
    total_in: int
    total_out: int


class LangGraphOrchestrator:
    """
    An instantiable Orchestrator that uses LangGraph to coordinate
    Planning, Execution, and Monitoring agents in a robust state machine.
    """
    
    def __init__(
        self,
        planner: PlanningAgent,
        executor: ExecutionAgent,
        evaluator: EvaluatorAgent,
        summarizer: Any = None,
        max_retries: int = 2
    ):
        self.planner = planner
        self.executor = executor
        self.evaluator = evaluator
        self.summarizer = summarizer
        self.max_retries = max_retries
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Builds and wires the robust LangGraph State Machine."""
        
        workflow = StateGraph(OrchestratorState)
        
        # 1. Define Nodes
        workflow.add_node("planner_node", self._node_planner)
        workflow.add_node("executor_node", self._node_executor)
        workflow.add_node("evaluator_node", self._node_evaluator)
        
        # 2. Define Entry Point
        workflow.set_entry_point("planner_node")
        
        # 3. Define Edges and Conditional Logic
        workflow.add_edge("planner_node", "executor_node")
        workflow.add_edge("executor_node", "evaluator_node")
        
        # Conditional edge from monitor: Either next step, retry current, or fail/end.
        workflow.add_conditional_edges(
            "evaluator_node",
            self._route_after_evaluator,
            {
                "next_step": "executor_node",
                "retry": "executor_node",
                "end": END
            }
        )
        
        return workflow.compile()
        
    # --- Node Implementations ---
    
    def _node_planner(self, state: OrchestratorState) -> Dict[str, Any]:
        logger.info("[Orchestrator] Running Planner Node...")
        m = {}
        plan = self.planner.generate_plan(state["task"], metadata=m)
        
        return {
            "plan": plan,
            "current_step_index": 0,
            "status": "executing",
            "total_duration": m.get("duration", 0),
            "total_in": m.get("usage", {}).get("input_tokens", 0),
            "total_out": m.get("usage", {}).get("output_tokens", 0)
        }
        
    def _node_executor(self, state: OrchestratorState) -> Dict[str, Any]:
        step_index = state.get("current_step_index", 0)
        plan = state.get("plan", [])
        
        if step_index >= len(plan):
            return {"status": "success"}
            
        current_step = plan[step_index]
        logger.info(f"[Orchestrator] Running Executor Node for step {step_index + 1}/{len(plan)}...")
        
        context = state.get("context", "")
        
        total_duration = state.get("total_duration", 0)
        total_in = state.get("total_in", 0)
        total_out = state.get("total_out", 0)
        
        # Summarize context if a compressor is provided
        if context and self.summarizer:
            if hasattr(self.summarizer, 'invoke'):
                m_comp = {}
                context = self.summarizer.invoke(context, metadata=m_comp)
                total_duration += m_comp.get("duration", 0)
                total_in += m_comp.get("usage", {}).get("input_tokens", 0)
                total_out += m_comp.get("usage", {}).get("output_tokens", 0)
            elif hasattr(self.compressor, '_run'):
                context = self.compressor._run(context)
                
        # Execute the step
        m_exec = {}
        result = self.executor.execute_step(current_step, context=context, metadata=m_exec)
        total_duration += m_exec.get("duration", 0)
        total_in += m_exec.get("usage", {}).get("input_tokens", 0)
        total_out += m_exec.get("usage", {}).get("output_tokens", 0)
        
        # We append a temporary pending result that the monitor will validate
        return {
            "results": [{"step": current_step, "result": result, "status": "pending_validation"}],
            "total_duration": total_duration,
            "total_in": total_in,
            "total_out": total_out
        }
        
    def _node_evaluator(self, state: OrchestratorState) -> Dict[str, Any]:
        logger.info("[Orchestrator] Running Evaluator Node...")
        
        step_index = state.get("current_step_index", 0)
        plan = state.get("plan", [])
        current_step = plan[step_index]
        
        # Get the latest result
        latest_result_obj = state["results"][-1]
        actual_result = latest_result_obj["result"]
        
        # Evaluate
        m_eval = {}
        eval_data = self.evaluator.evaluate(current_step, actual_result, metadata=m_eval)
        success = eval_data.get("success", False)
        feedback = eval_data.get("feedback", "")
        
        total_duration = state.get("total_duration", 0) + m_eval.get("duration", 0)
        total_in = state.get("total_in", 0) + m_eval.get("usage", {}).get("input_tokens", 0)
        total_out = state.get("total_out", 0) + m_eval.get("usage", {}).get("output_tokens", 0)
        
        if success:
            logger.info(f"[Orchestrator] Step {step_index + 1} Succeeded.")
            
            # Mutate the dictionary reference so the final state reflects it
            state["results"][-1]["status"] = "validated"
            
            # Update context for the next step
            new_context = state.get("context", "") + f"\n[Step {step_index + 1} Result]: {actual_result}"
            
            # If this was the last step in the plan, mark workflow as success
            next_status = "success" if (step_index + 1) >= len(plan) else "executing"
            
            # The result object is valid, simply update state to move to next step
            return {
                "current_step_index": step_index + 1,
                "context": new_context,
                "attempts": 0,  # Reset attempts for the next step
                "status": next_status,
                "total_duration": total_duration,
                "total_in": total_in,
                "total_out": total_out
            }
        else:
            state["results"][-1]["status"] = "failed"
            attempts = state.get("attempts", 0) + 1
            logger.warning(f"[Orchestrator] Step {step_index + 1} Failed. Attempt {attempts}/{self.max_retries}. Feedback: {feedback}")
            
            new_context = state.get("context", "") + f"\n[Failed Attempt Feedback]: {feedback}"
            
            if attempts > self.max_retries:
                logger.error("[Orchestrator] Max retries reached. Aborting.")
                return {
                    "status": "failed",
                    "total_duration": total_duration,
                    "total_in": total_in,
                    "total_out": total_out
                }
                
            return {
                "attempts": attempts,
                "context": new_context,
                "status": "executing",
                "total_duration": total_duration,
                "total_in": total_in,
                "total_out": total_out
            }
            
    # --- Routing Logic ---
    
    def _route_after_evaluator(self, state: OrchestratorState) -> str:
        status = state.get("status")
        
        if status == "failed":
            return "end"
            
        # Check if all steps are completed
        step_index = state.get("current_step_index", 0)
        plan = state.get("plan", [])
        
        if step_index >= len(plan):
            return "end"
            
        # If there are attempts > 0 but it's not failed, it implies we just incremented attempt counter (retry)
        if state.get("attempts", 0) > 0:
            return "retry"
            
        # Otherwise, proceed to next step
        return "next_step"

    # --- Public API ---
    
    def run(self, task: str) -> Dict[str, Any]:
        """
        Executes the orchestrator graph with the given task.
        """
        initial_state = {
            "task": task,
            "plan": [],
            "current_step_index": 0,
            "context": "",
            "results": [],
            "attempts": 0,
            "max_retries": self.max_retries,
            "status": "planning",
            "total_duration": 0.0,
            "total_in": 0,
            "total_out": 0
        }
        
        logger.info(f"[Orchestrator] Starting workflow for task: {task[:50]}...")
        final_state = self.graph.invoke(initial_state)
        
        # Add a unified metadata object to match other workflows
        final_state["execution_metadata"] = {
            "total_duration": final_state.get("total_duration", 0),
            "total_tokens": final_state.get("total_in", 0) + final_state.get("total_out", 0),
            "usage": {"input": final_state.get("total_in", 0), "output": final_state.get("total_out", 0)}
        }
        
        return final_state
