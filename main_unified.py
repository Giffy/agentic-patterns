import argparse
import sys
import logging
from unified_entry_point import run_agent

# Set up logging to follow the orchestration steps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description="Agentic Patterns - Unified Entry Point CLI")
    
    parser.add_argument(
        "--task", 
        type=str, 
        required=True, 
        help="The task you want the agent to perform."
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="hybrid", 
        choices=["local", "hybrid", "cloud"],
        help="Model mode: local (all local), hybrid (planning cloud, execution local), or cloud (all cloud)."
    )
    
    parser.add_argument(
        "--arch", 
        type=str, 
        default="router", 
        choices=["prompt_chain", "parallel", "orchestrator", "router"],
        help="Architecture to use. 'router' auto-selects based on task complexity."
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print(f" INITIALIZING UNIFIED AGENT ".center(60, "="))
    print(f" Task: {args.task}")
    print(f" Model Mode: {args.model}")
    print(f" Architecture: {args.arch}")
    print("="*60 + "\n")

    try:
        result = run_agent(
            task=args.task,
            model_type=args.model,
            architecture=args.arch
        )
        
        print("\n" + "="*60)
        print(f" EXECUTION COMPLETED ".center(60, "="))
        print(f" Status: {result.get('status', 'unknown')}")
        
        # Display performance metrics if available
        meta = result.get("execution_metadata", {})
        if meta:
            duration = meta.get("total_duration", 0)
            tokens = meta.get("total_tokens", 0)
            usage = meta.get("usage", {})
            print(f" Total Duration: {duration:.2f}s")
            print(f" Total Tokens: {tokens:,} (In: {usage.get('input', 0):,}, Out: {usage.get('output', 0):,})")
            
        print("="*60 + "\n")
        
        # Friendly output of results
        if "completed_results" in result:
            for i, res in enumerate(result["completed_results"], 1):
                print(f"Step {i}: {res['step']}")
                print(f"Result: {res['result']}\n")
        elif "results" in result:
             for i, res in enumerate(result["results"], 1):
                print(f"Step {i}: {res['step']}")
                print(f"Result: {res['result']}\n")
        else:
            print("Result Object:", result)

    except Exception as e:
        print(f"\n[ERROR] execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
