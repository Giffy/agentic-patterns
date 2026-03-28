# Agentic Patterns Framework

A modular, extensible framework for implementing sophisticated AI agent architectures using LangChain, LangGraph, and Ollama.

## 🚀 Unified Agent Entry Point

The framework provides a unified interface to execute tasks using different model strategies and execution patterns.

### 🧠 Model Modes

Select the model strategy that fits your needs:
-   `local`: Executes all steps using a local model via Ollama.
-   `cloud`: Executes all steps using a cloud-based model (OpenAI, Anthropic, or OpenAI-compatible APIs).
-   `hybrid`: Uses Cloud LLMs for high-reasoning tasks (Planning, Monitoring) and Local LLMs for task-specific execution (Execution, Context Compression).

### 🏗️ Architectures

Choose the best execution pattern for your task:
-   `prompt_chain`: A simple sequential execution of steps.
-   `parallel`: Breaks the task into independent sub-tasks executed in parallel.
-   `orchestrator`: A complex State Graph-based architecture (LangGraph) with iterative planning, execution, and validation.
-   `router`: **(Recommended)** Automatically selects the most efficient architecture based on the task's complexity using an intelligent Coordinator agent.

---

## 🛠️ Usage

### Terminal (CLI)
Use `uv run main_unified.py` to trigger the agent from the command line:

```bash
# Auto-route a complex task using hybrid models
uv run main_unified.py --task "Research the history of AI and identify 5 key milestones" --model hybrid --arch router

# Run a simple task using only local models
uv run main_unified.py --task "What is 2+2?" --model local --arch prompt_chain
```

### Python API
Integrate the framework directly into your applications:

```python
from unified_entry_point import run_agent

result = run_agent(
    task="Write a high-performance Python function for Fibonacci",
    model_type="hybrid",
    architecture="orchestrator"
)

print(f"Status: {result['status']}")
print(f"Steps Completed: {len(result['results'])}")
```

---

## ⚙️ Configuration

Create a `.env` file in the root directory:

```env
# Cloud Model Configuration
HOST=https://api.openai.com/v1
MODEL=gpt-4o
API_KEY=your_api_key_here

# Local Model Configuration (Ollama)
OLLAMA_HOST=http://localhost:11434
LOCAL_MODEL=llama3
```

## 📂 Project Structure

-   `agents/`: Specialized agent implementations (Planning, Execution, Monitoring, etc.)
-   `workflows/`: Pre-defined execution patterns (Sequential, Parallel, etc.)
-   `orchestators/`: Advanced orchestrators (LangGraph)
-   `services/`: Core utilities (LLM Factory, Coordinator)
-   `tools/`: Reusable tools (Web Search, Context Compression)
-   `memory/`: Short-term and long-term memory implementations
