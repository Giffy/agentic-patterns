import os
from dotenv import load_dotenv

# Cloud LLMs
from langchain_openai import ChatOpenAI
# Local LLMs
from langchain_ollama import ChatOllama
from agents.local_agent import LocalAgent

# Import your custom agent classes
from agents import PlanningAgent, ExecutionAgent, MonitoringAgent

# Import your workflows
from workflows import SequentialWorkflow, ParallelWorkflow

# Import your orchestrator
from orchestators import LangGraphOrchestrator

# Import the new local compression tool
from tools.compress_context_tool import CompressContextTool
from tools.curl_search_tool import CurlSearchTool

def main():
    # Load environment variables (e.g., OPENAI_API_KEY)
    load_dotenv()
    
    LLM_HOST=os.getenv("SUMMARY_HOST")
    MODEL=os.getenv("SUMMARY_MODEL")
    AGENT_API_KEY=os.getenv("SUMMARY_AGENT_API_KEY")
    LOCAL_MODEL = os.getenv("LOCAL_MODEL")
    
    print("=== Initializing Agentic Patterns Components ===")
    
    
    # 1. Initialize the Main LLM
    # We use OpenAI by default, assuming an API key is in .env or environment
    #llm_cloud = ChatOpenAI(model_name=MODEL, temperature=0.3, openai_api_key=AGENT_API_KEY, base_url=LLM_HOST)
    llm = ChatOllama(model=LOCAL_MODEL)
    llm_cloud = llm
    #llm = llm_cloud
    # 1.5 Initialize Context Compressor
    compressor_tool = CompressContextTool(max_length=10000)
    curl_tool = CurlSearchTool()

    context_compressor_agent = None
    
    if LOCAL_MODEL:
        try:
            context_compressor_agent = LocalAgent(llm=llm)
            print(f"[*] Using LocalAgent with model '{LOCAL_MODEL}' for context compression.")
        except ImportError:
            print("[!] Unable to import langchain_ollama. Falling back to simple CompressContextTool.")
    else:
        print("[*] No 'LOCAL_MODEL' environment variable found. Using simple CompressContextTool for compression.")
    
    # 2. Instantiate Agents
    planner = PlanningAgent(llm=llm)
    executor = ExecutionAgent(llm=llm, tools=[curl_tool])
    monitor = MonitoringAgent(llm=llm_cloud)
    
    agent_dict = {
        "planner": planner,
        "executor": executor,
        "monitor": monitor
    }
    
    # Define a test task
    # task = "Write a python function that calculates the Fibonacci sequence and add comments explaining the complex parts."
    task = "check again What is the capital of Andorra?"
    #task = "What is the capital of France?"
    
    # Import memory
    from memory.short_term_memory import SQLiteShortTermMemory
    memory = SQLiteShortTermMemory()

    # --- CHECK CACHE FIRST ---
    cached_response = memory.get_exact_match_answer(task)
    if cached_response:
        print(f"\n[CACHE HIT] Found previous answer for task: '{task}'")
        print("Skipping expensive LLM orchestration...\n")
        print(f"Cached Answer:\n{cached_response}")
        return # Exit early instead of calling LLMs
        
    print(f"\n[CACHE MISS] No previous answer found for: '{task}'")
    print("Proceeding to execute through workflows...\n")
    
    # Demo compress tool locally before workflow starts
    long_task = task + " AND " * 20 + "Make sure it is extremely performant."
    print("\n--- Testing Compression Tool ---")
    print(f"Original Length: {len(long_task)}")
    
    # Determine the fastest/configured compressor
    active_compressor = compressor_tool
    
    if context_compressor_agent:
        try:
            compressed_task = context_compressor_agent.invoke(long_task)
            print(f"Compressed by LocalAgent:\n{compressed_task}")
            active_compressor = context_compressor_agent
        except Exception as e:
            print(f"\n[!] LocalAgent failed (Is Ollama running locally?). Falling back to python tool. Error details: {str(e)[:100]}...")
            compressed_task = compressor_tool._run(long_task)
            print(f"Compressed by Tool fallback:\n{compressed_task}")
    else:
        compressed_task = compressor_tool._run(long_task)
        print(f"Compressed by Tool:\n{compressed_task}")
    
    # --- METHOD 1: Using the Workflow Architecture ---
    print("\n--- Running Sequential Workflow ---")
    sequential_workflow = SequentialWorkflow(agents=agent_dict, tools=[active_compressor, curl_tool])
    
    try:
        workflow_result = sequential_workflow.run(task=task)
        print(f"Workflow Status: {workflow_result['status']}")
        for res in workflow_result.get('completed_results', []):
            print(f"\nStep: {res['step']}\nOutput:\n{res['result']}")
    except Exception as e:
        print(f"Workflow failed: {e}")
        
    print("\n" + "="*50 + "\n")
    
    # --- METHOD 2: Using the Orchestrator Architecture (LangGraph) ---
    print("--- Running LangGraph Orchestrator ---")
    orchestrator = LangGraphOrchestrator(
        planner=planner,
        executor=executor,
        monitor=monitor,
        compressor=active_compressor,
        max_retries=2
    )
    
    try:
        orchestrator_result = orchestrator.run(task=task)
        print(f"Orchestrator Final Status: {orchestrator_result['status']}")
        print(f"Plan steps generated: {len(orchestrator_result['plan'])}")

        print(orchestrator_result)
        
        # Write result to a file called result.txt
        import json
        with open("result.txt", "w", encoding="utf-8") as f:
            json.dump(orchestrator_result, f, indent=4, ensure_ascii=False)
        
        # Output the accumulated results across the state graph steps
        for res in orchestrator_result.get('results', []):
            if res.get('status') == 'validated' and orchestrator_result['status'] == 'success':
                # Simplified assumption for demo
                print(f"\nStep: {res['step']}\nOutput:\n{res['result']}")

        # --- SAVE TO CACHE ---
        # If the orchestrator succeeded, we store the question and the final parsed results into the SQLite memory
        if orchestrator_result['status'] == 'success':
            session = "global_cache_session"
            # 1. User query
            memory.add_memory(session_id=session, role="user", content=task)
            
            # Combine all successful steps for the final cached Assistant answer
            final_answer = "\n".join([f"Step: {r['step']}\nResult: {r['result']}" for r in orchestrator_result.get('results', []) if r.get('status') == 'validated'])
            
            # 2. Assistant answer
            memory.add_memory(session_id=session, role="assistant", content=final_answer)
            print("[CACHE] Successfully saved Orchestrator final output to SQLite ShortTermMemory.")
            
    except Exception as e:
        print(f"Orchestrator failed: {e}")

if __name__ == "__main__":
    main()
