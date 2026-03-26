import logging
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalAgent:
    """
    A local agent designed to summarize its context into a concise, condensed format.
    Its primary purpose is to dramatically reduce text volume, saving context tokens 
    for downstream execution while strictly preserving all important and relevant information.
    """
    
    def __init__(
        self, 
        llm: BaseChatModel, 
        system_prompt: str = "You are a specialized summarization agent. Condense the provided text aggressively to save tokens while keeping all crucial facts and information.",
        agent_name: str = "LocalAgent"
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.agent_name = agent_name
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])
    
    def invoke(self, user_input: str, **kwargs: Any) -> Any:
        """
        Local execution method for the agent.
        
        Args:
            user_input: The main task or prompt for the agent.
            **kwargs: Additional variables to format the prompt.
            
        Returns:
            The agent's response.
        """
        logger.info(f"[{self.agent_name}] Invoked with input: {user_input[:50]}...")
        
        chain = self.prompt_template | self.llm
        
        try:
            response = chain.invoke({"input": user_input, **kwargs})
            return response.content
        except Exception as e:
            logger.error(f"[{self.agent_name}] Error during invocation: {str(e)}")
            raise
