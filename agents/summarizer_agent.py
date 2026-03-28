import logging
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class SummarizerAgent(BaseAgent):
    """
    A summarizer agent designed to summarize its context into a concise, condensed format.
    Its primary purpose is to dramatically reduce text volume, saving context tokens 
    for downstream execution while strictly preserving all important and relevant information.
    """
    
    def __init__(
        self, 
        llm: BaseChatModel, 
        system_prompt: str = "You are a specialized summarization agent. Condense the provided text aggressively to save tokens while keeping all crucial facts and information.",
        agent_name: str = "SummarizerAgent"
    ):
        super().__init__(
            llm=llm, 
            system_prompt=system_prompt,
            agent_name=agent_name
        )

