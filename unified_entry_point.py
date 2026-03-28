import os
import logging
from typing import Dict, Any, Union, List

# Core Agent Imports
from agents import PlanningAgent, ExecutionAgent, EvaluatorAgent, SummarizerAgent
from workflows import SequentialWorkflow, ParallelWorkflow, DirectWorkflow
from orchestators import LangGraphOrchestrator
from services.llm_factory import LLMFactory
from services.coordinator import Coordinator
from tools.web_search_tool import WebSearchTool
from tools.compress_context_tool import CompressContextTool

logger = logging.getLogger(__name__)

class UnifiedAgent:
    """
    The main entry point for AI tasks, supporting multi-model and multi-architecture selectors.
    """
    
    def __init__(self, model_type: str = "cloud", architecture: str = "router"):
        self.model_type = model_type
        self.requested_architecture = architecture
        
        # 1. Initialize LLMs for different roles
        self.llm_bundle = LLMFactory.get_all_agents_llms(mode=model_type)
        
        # 2. Tools
        self.web_search_tool = WebSearchTool()
        self.compress_context_tool = CompressContextTool()
        
        # 3. Agents
        self.planner    = PlanningAgent(llm=self.llm_bundle["planner"])
        self.executor   = ExecutionAgent(llm=self.llm_bundle["executor"])
        self.evaluator  = EvaluatorAgent(llm=self.llm_bundle["evaluator"])
        self.summarizer = SummarizerAgent(llm=self.llm_bundle["summarizer"])
        
        self.agent_dict = {
            "planner":  self.planner,
            "executor": self.executor,
            "evaluator":  self.evaluator,
            "summarizer": self.summarizer
        }
        
        # 4. Coordinator for routing
        self.coordinator = Coordinator(llm=self.llm_bundle["coordinator"])

    def run(self, task: str) -> Dict[str, Any]:
        """
        Executes a task through selected model and architecture.
        """
        logger.info(f"[UnifiedAgent] Received task: {task[:100]}...")
        
        # Architecture selection logic
        arch_to_use = self.requested_architecture
        routing_metadata = {}
        
        if self.requested_architecture == "router":
            decision = self.coordinator.select_architecture(task, metadata=routing_metadata)
            arch_to_use = decision["architecture"]
            logger.info(f"[UnifiedAgent] Coordinator selected architecture: {arch_to_use}")
            
        # Implementation dispatch
        result = {}
        if arch_to_use == "prompt_chain":
            workflow = SequentialWorkflow(agents=self.agent_dict, tools=[self.compress_context_tool])
            result = workflow.run(task=task)
            
        elif arch_to_use == "parallel":
            # Assuming parallel sub-workflow for simple tasks (can be customized further)
            workflow = ParallelWorkflow(agents=self.agent_dict)
            result = workflow.run(task=task)
            
        elif arch_to_use == "direct":
            workflow = DirectWorkflow(agents=self.agent_dict)
            result = workflow.run(task=task)
            
        elif arch_to_use == "orchestrator":
            orchestrator = LangGraphOrchestrator(
                planner=self.planner,
                executor=self.executor,
                evaluator=self.evaluator,
                summarizer=self.summarizer,
                max_retries=2
            )
            result = orchestrator.run(task=task)
            
        else:
            logger.warning(f"Unknown architecture '{arch_to_use}', falling back to orchestrator.")
            orchestrator = LangGraphOrchestrator(
                planner=self.planner,
                executor=self.executor,
                evaluator=self.evaluator,
                summarizer=self.summarizer
            )
            result = orchestrator.run(task=task)

        # Merge routing metrics if they exist
        if routing_metadata and "execution_metadata" in result:
            result["execution_metadata"]["total_duration"] += routing_metadata.get("duration", 0)
            u = routing_metadata.get("usage", {})
            result["execution_metadata"]["total_tokens"] += (u.get("input_tokens", 0) + u.get("output_tokens", 0))
            result["execution_metadata"]["usage"]["input"] += u.get("input_tokens", 0)
            result["execution_metadata"]["usage"]["output"] += u.get("output_tokens", 0)
            
        return result

def run_agent(task: str, model_type: str = "hybrid", architecture: str = "router") -> Dict[str, Any]:
    """
    Simplified functional API for calling agents.
    """
    agent = UnifiedAgent(model_type=model_type, architecture=architecture)
    return agent.run(task=task)
