import pytest
from unittest.mock import MagicMock, patch
from unified_entry_point import UnifiedAgent, run_agent

@pytest.fixture
def mock_llm_factory():
    with patch("unified_entry_point.LLMFactory") as mock:
        mock.get_all_agents_llms.return_value = {
            "planner": MagicMock(),
            "executor": MagicMock(),
            "evaluator": MagicMock(),
            "summarizer": MagicMock(),
            "coordinator": MagicMock(),
        }
        yield mock

@pytest.fixture
def mock_agents():
    with patch("unified_entry_point.PlanningAgent") as p, \
         patch("unified_entry_point.ExecutionAgent") as ex, \
         patch("unified_entry_point.EvaluatorAgent") as ev, \
         patch("unified_entry_point.SummarizerAgent") as s:
        yield {
            "planner": p,
            "executor": ex,
            "evaluator": ev,
            "summarizer": s
        }

@pytest.fixture
def mock_workflows():
    with patch("unified_entry_point.SequentialWorkflow") as seq, \
         patch("unified_entry_point.ParallelWorkflow") as par, \
         patch("unified_entry_point.LangGraphOrchestrator") as orch:
        yield {
            "sequential": seq,
            "parallel": par,
            "orchestrator": orch
        }

@pytest.fixture
def mock_coordinator():
    with patch("unified_entry_point.Coordinator") as mock:
        instance = mock.return_value
        instance.select_architecture.return_value = {"architecture": "orchestrator"}
        yield mock

def test_initialization(mock_llm_factory, mock_agents, mock_coordinator):
    """Test that UnifiedAgent initializes correctly with its components."""
    agent = UnifiedAgent(model_type="cloud", architecture="router")
    
    # Check if LLM factory was called
    mock_llm_factory.get_all_agents_llms.assert_called_once_with(mode="cloud")
    
    # Check if agents were initialized
    assert agent.planner is not None
    assert agent.executor is not None
    assert agent.evaluator is not None
    assert agent.summarizer is not None
    assert agent.coordinator is not None

def test_run_with_router(mock_llm_factory, mock_agents, mock_coordinator, mock_workflows):
    """Test that 'router' architecture correctly calls coordinator and dispatches."""
    agent = UnifiedAgent(model_type="cloud", architecture="router")
    
    # Mock return value for orchestrator run
    mock_orch_instance = mock_workflows["orchestrator"].return_value
    mock_orch_instance.run.return_value = {"status": "success", "result": "orchestrated output"}
    
    # Execute
    result = agent.run("test task")
    
    # Verify coordinator was called
    agent.coordinator.select_architecture.assert_called_once()
    
    # Verify the results
    assert result["result"] == "orchestrated output"
    mock_orch_instance.run.assert_called_once_with(task="test task")

def test_run_parallel(mock_llm_factory, mock_agents, mock_workflows):
    """Test that 'parallel' architecture works correctly."""
    agent = UnifiedAgent(model_type="local", architecture="parallel")
    
    # Mock return value
    mock_par_instance = mock_workflows["parallel"].return_value
    mock_par_instance.run.return_value = {"status": "success", "result": "parallel output"}
    
    # Execute
    result = agent.run("parallel task")
    
    # Verify parallel workflow was used
    mock_workflows["parallel"].assert_called_once()
    assert result["result"] == "parallel output"

def test_run_orchestrator(mock_llm_factory, mock_agents, mock_workflows):
    """Test that 'orchestrator' architecture works correctly."""
    agent = UnifiedAgent(model_type="hybrid", architecture="orchestrator")
    
    # Mock return value
    mock_orch_instance = mock_workflows["orchestrator"].return_value
    mock_orch_instance.run.return_value = {"status": "success", "result": "orchestrated output"}
    
    # Execute
    result = agent.run("orchestrator task")
    
    # Verify orchestrator was used
    mock_workflows["orchestrator"].assert_called_once()
    assert result["result"] == "orchestrated output"

def test_run_prompt_chain(mock_llm_factory, mock_agents, mock_workflows):
    """Test that 'prompt_chain' architecture works correctly (formerly sequential)."""
    agent = UnifiedAgent(model_type="cloud", architecture="prompt_chain")
    
    # Mock return value
    mock_seq_instance = mock_workflows["sequential"].return_value
    mock_seq_instance.run.return_value = {"status": "success", "result": "sequential output"}
    
    # Execute
    result = agent.run("sequential task")
    
    # Verify sequential workflow was used
    mock_workflows["sequential"].assert_called_once()
    assert result["result"] == "sequential output"

def test_metrics_merging(mock_llm_factory, mock_agents, mock_coordinator, mock_workflows):
    """Test that metrics from coordinator and workflow are merged correctly."""
    agent = UnifiedAgent(model_type="cloud", architecture="router")
    
    # Coordinator metrics
    routing_metadata = {
        "duration": 0.5,
        "usage": {"input_tokens": 100, "output_tokens": 50}
    }
    
    # Mock coordinator to modify metadata (emulating pass-by-reference if that's how it works)
    # Actually, unified_entry_point.py line 57 passes routing_metadata
    def side_effect(task, metadata=None):
        if metadata is not None:
            metadata.update(routing_metadata)
        return {"architecture": "orchestrator"}
        
    agent.coordinator.select_architecture.side_effect = side_effect
    
    # Workflow result with its own metrics
    workflow_result = {
        "status": "success",
        "result": "the output",
        "execution_metadata": {
            "total_duration": 2.0,
            "total_tokens": 500,
            "usage": {"input": 300, "output": 200}
        }
    }
    mock_orch_instance = mock_workflows["orchestrator"].return_value
    mock_orch_instance.run.return_value = workflow_result
    
    # Execute
    result = agent.run("metrics task")
    
    # Verify merged metrics
    # Expected: 2.0 + 0.5 = 2.5 duration
    # Expected: 500 + 150 = 650 tokens
    # Expected usage: input 300 + 100 = 400, output 200 + 50 = 250
    meta = result["execution_metadata"]
    assert meta["total_duration"] == 2.5
    assert meta["total_tokens"] == 650
    assert meta["usage"]["input"] == 400
    assert meta["usage"]["output"] == 250

def test_functional_run_agent(mock_llm_factory, mock_agents, mock_workflows):
    """Test the run_agent module-level function."""
    with patch("unified_entry_point.UnifiedAgent") as mock_agent_class:
        mock_instance = mock_agent_class.return_value
        mock_instance.run.return_value = {"status": "success"}
        
        result = run_agent("test task", model_type="local", architecture="parallel")
        
        mock_agent_class.assert_called_once_with(model_type="local", architecture="parallel")
        mock_instance.run.assert_called_once_with(task="test task")
        assert result["status"] == "success"
