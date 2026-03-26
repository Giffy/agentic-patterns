from typing import Optional
from langchain_core.tools import BaseTool
import re

class CompressContextTool(BaseTool):
    """
    A tool to locally compress text context by removing unnecessary whitespace,
    common filler words, and optionally truncating to a maximum length.
    This saves tokens and reduces prompt size without relying on external LLM calls.
    """
    
    name: str = "compress_context"
    description: str = (
        "Compresses a long string of text locally to save context window space. "
        "Useful when you have downloaded a large document or webpage and need to "
        "extract the core information densely."
    )
    
    max_length: Optional[int] = 4000
    
    def _run(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Locally compresses the text.
        """
        # 1. Remove extra whitespaces and newlines
        compressed = re.sub(r'\s+', ' ', text).strip()
        
        # 2. Simple removal of common filler words (optional/heuristic)
        # This is a very basic list for fast local execution.
        stop_words = {" a ", " an ", " the ", " is ", " are ", " was ", " were ", " and ", " or ", " but "}
        for word in stop_words:
            # Case insensitive replace, simple naive approach
            compressed = re.sub(word, " ", compressed, flags=re.IGNORECASE)
            
        # Clean up double spaces created by replacement
        compressed = re.sub(r'\s+', ' ', compressed).strip()
        
        # 3. Truncate if specified
        limit = max_length if max_length is not None else self.max_length
        if limit and len(compressed) > limit:
            compressed = compressed[:limit] + "... [TRUNCATED]"
            
        return compressed

    def _arun(self, text: str, max_length: Optional[int] = None) -> str:
        """Async version of the tool."""
        return self._run(text, max_length)

# Example helper function if you don't want to use the BaseTool class directly
def compress_text_locally(text: str, max_length: int = 4000) -> str:
    """Utility function to compress text directly."""
    tool = CompressContextTool(max_length=max_length)
    return tool._run(text)
