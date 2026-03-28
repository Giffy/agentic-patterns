import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from typing import Dict, Any, Union

load_dotenv()

class LLMFactory:
    """
    Factory to initialize LLMs based on execution mode: local, cloud, or hybrid.
    """
    
    @staticmethod
    def get_llm(mode: str = "cloud", role: str = "general") -> Union[ChatOpenAI, ChatOllama]:
        """
        Initialization logic for different modes.
        Modes:
            - local: Always returns a local model (Ollama).
            - cloud: Always returns a cloud model (OpenAI-compatible).
            - hybrid: Returns cloud for high-reasoning roles (planner, monitor) 
                     and local for high-volume roles (executor, compressor).
        """
        
        # Load config from env
        cloud_model = os.getenv("MODEL", "gpt-4o")
        cloud_host = os.getenv("HOST", "https://api.openai.com/v1")
        cloud_api_key = os.getenv("API_KEY")

        local_model = os.getenv("LOCAL_MODEL", "smollm3:135m")
        local_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        if mode == "local" or (mode == "cloud" and "localhost" in cloud_host.lower()):
            # If "cloud" mode points to a local host, use ChatOllama
            use_host = local_host if mode == "local" else cloud_host
            use_model = local_model if mode == "local" else cloud_model
            return ChatOllama(model=use_model, base_url=use_host)
        
        elif mode == "cloud":
            # Ensure cloud_host has /v1 if missing and it's not a known host
            if "openai" in cloud_host.lower() and not cloud_host.endswith("/v1"):
                cloud_host = f"{cloud_host.rstrip('/')}/v1"
            
            return ChatOpenAI(
                model_name=cloud_model, 
                openai_api_key=cloud_api_key, 
                base_url=cloud_host,
                temperature=0.3
            )
            
        elif mode == "hybrid":
            # high reasoning roles -> cloud
            if role in ["planner", "monitor", "coordinator", "evaluator", "summarizer"]:
                return LLMFactory.get_llm(mode="cloud")
            # execution/support roles -> local
            else:
                return LLMFactory.get_llm(mode="local")
        
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def get_all_agents_llms(mode: str) -> Dict[str, Any]:
        """
        Returns a dictionary of LLMs for all standard agent roles.
        """
        return {
            "planner":    LLMFactory.get_llm(mode=mode, role="planner"),
            "executor":   LLMFactory.get_llm(mode=mode, role="executor"),
            "evaluator":  LLMFactory.get_llm(mode=mode, role="evaluator"),
            "summarizer": LLMFactory.get_llm(mode=mode, role="summarizer"),
            "monitor":    LLMFactory.get_llm(mode=mode, role="monitor"),
            "compressor": LLMFactory.get_llm(mode=mode, role="compressor"),
            "coordinator": LLMFactory.get_llm(mode=mode, role="coordinator")
        }
