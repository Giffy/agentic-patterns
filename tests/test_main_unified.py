import sys
import io
import pytest
from unittest.mock import patch, MagicMock
from main_unified import main

def test_main_success():
    """Test that main executes correctly with standard arguments."""
    test_args = ["main_unified.py", "--task", "hello", "--model", "local", "--arch", "prompt_chain"]
    mock_result = {
        "status": "success",
        "results": [{"step": "Step 1", "result": "Hello world"}]
    }
    
    with patch("sys.argv", test_args), \
         patch("main_unified.run_agent", return_value=mock_result) as mock_run, \
         patch("sys.stdout", new=io.StringIO()) as fake_out:
        
        main()
        
        # Verify run_agent was called with correct arguments
        mock_run.assert_called_once_with(
            task="hello",
            model_type="local",
            architecture="prompt_chain"
        )
        
        output = fake_out.getvalue()
        assert "INITIALIZING UNIFIED AGENT" in output
        assert "Task: hello" in output
        assert "EXECUTION COMPLETED" in output
        assert "Step 1: Step 1" in output
        assert "Result: Hello world" in output

def test_main_with_metrics():
    """Test that main correctly displays performance metrics."""
    test_args = ["main_unified.py", "--task", "metrics test"]
    mock_result = {
        "status": "success",
        "execution_metadata": {
            "total_duration": 12.345,
            "total_tokens": 1500,
            "usage": {"input": 1000, "output": 500}
        },
        "completed_results": [{"step": "Final Step", "result": "Done"}]
    }
    
    with patch("sys.argv", test_args), \
         patch("main_unified.run_agent", return_value=mock_result), \
         patch("sys.stdout", new=io.StringIO()) as fake_out:
        
        main()
        
        output = fake_out.getvalue()
        assert "Total Duration: 12.35s" in output
        assert "Total Tokens: 1,500 (In: 1,000, Out: 500)" in output
        assert "Step 1: Final Step" in output

def test_main_error():
    """Test that main handles errors gracefully and exits with status 1."""
    test_args = ["main_unified.py", "--task", "fail task"]
    
    with patch("sys.argv", test_args), \
         patch("main_unified.run_agent", side_effect=Exception("Simulated Failure")), \
         patch("sys.stdout", new=io.StringIO()) as fake_out:
        
        with pytest.raises(SystemExit) as excinfo:
            main()
            
        assert excinfo.value.code == 1
        output = fake_out.getvalue()
        assert "[ERROR] execution failed: Simulated Failure" in output

def test_main_default_args():
    """Test that main uses default arguments correctly."""
    # Only --task is required
    test_args = ["main_unified.py", "--task", "default test"]
    mock_result = {"status": "success"}
    
    with patch("sys.argv", test_args), \
         patch("main_unified.run_agent", return_value=mock_result) as mock_run, \
         patch("sys.stdout", new=io.StringIO()):
        
        main()
        
        mock_run.assert_called_once_with(
            task="default test",
            model_type="hybrid",  # Default from argparse
            architecture="router"  # Default from argparse
        )
