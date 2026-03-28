from duckduckgo_search import DDGS
from typing import Optional, List, Dict
from langchain_core.tools import BaseTool

class WebSearchTool(BaseTool):
    """
    A tool that uses DuckDuckGo to perform a search query and retrieve 
    both snippets and direct links.
    """
    
    name: str = "web_search"
    description: str = (
        "Queries DuckDuckGo Search to retrieve concise answers, context, and links. "
        "Input should be a specific search query."
    )
    
    def _run(self, query: str) -> str:
        """
        Executes a DuckDuckGo search and returns a formatted string of results.
        """
        try:
            with DDGS() as ddgs:
                # Get top 5 results
                results = list(ddgs.text(query, max_results=5))
            
            if not results:
                return f"No results found for query: '{query}'"
            
            print(f"Results found in DuckDuckGo for: {query}")
            
            formatted_results = []
            for i, res in enumerate(results):
                title = res.get("title", "No Title")
                href = res.get("href", "No Link")
                body = res.get("body", "No Snippet")
                
                formatted_results.append(
                    f"Result {i+1}:\n"
                    f"Title: {title}\n"
                    f"Link: {href}\n"
                    f"Snippet: {body}\n"
                )
                
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Search failed with an unexpected error: {str(e)}"

    def _arun(self, query: str) -> str:
        """Async version of the tool."""
        return self._run(query)

# Testing the function directly
if __name__ == "__main__":
    tool = WebSearchTool()
    print(tool._run("Capital of Andorra"))
