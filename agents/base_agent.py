import logging
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent:
    """
    A foundational agent class that can be reused and extended by specific agent roles.
    Designed to be easily instantiated and called by workflows or scripts.
    """
    
    def __init__(
        self, 
        llm: BaseChatModel, 
        system_prompt: str = "You are a helpful AI assistant.",
        agent_name: str = "BaseAgent"
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
        Main execution method for the agent.
        
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
