import subprocess
import json
from typing import Optional
from langchain_core.tools import BaseTool

class CurlSearchTool(BaseTool):
    """
    A tool that uses the system's `curl` command to perform a search query against
    a searcher API (e.g., Wikipedia) and retrieve an answer. 
    """
    
    name: str = "curl_search"
    description: str = (
        "Uses the `curl` command-line utility to query a search engine (like Wikipedia) "
        "and retrieve concise answers or context. Input should be a specific search query."
    )
    
    def _run(self, query: str) -> str:
        """
        Executes a curl command to search for the query and returns the parsed text.
        """
        # We use Wikipedia's open search API as a reliable "searcher" that doesn't 
        # heavily block curl requests like standard web search engines do.
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&utf8=&format=json"
        
        # Spoof an old Android mobile phone
        user_agent = "Mozilla/5.0 (Linux; U; Android 4.0.3; en-us; HTC Sensation Build/IML74K) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30"
        
        # Build the curl command
        # -s: Silent mode (don't show progress bar)
        # -L: Follow redirects
        # -A: Specify User-Agent string to bypass simple blocks
        cmd = ["curl", "-s", "-L", "-A", user_agent, url]
        
        try:
            # Execute the curl command using subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            response_text = result.stdout
            
            # Parse the JSON response text
            data = json.loads(response_text)
            search_results = data.get("query", {}).get("search", [])
            
            if not search_results:
                return f"No results found for query: '{query}'"
            print(f"Result found in Wikipedia {query}")
            # Aggregate the snippets from the top results
            answers = []
            for i, item in enumerate(search_results[:3]): # Keep it concise (top 3)
                # Wikipedia snippets often contain HTML tags like <span class="searchmatch">
                import re
                clean_snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
                answers.append(f"Result {i+1} ({item.get('title')}): {clean_snippet}")
                
            return "\n".join(answers)
            
        except subprocess.CalledProcessError as e:
            return f"Curl command failed with error: {e.stderr}"
        except json.JSONDecodeError:
            return f"Failed to parse search results. Raw output: {response_text[:200]}..."
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

    def _arun(self, query: str) -> str:
        """Async version of the tool."""
        return self._run(query)

# Testing the function directly
if __name__ == "__main__":
    tool = CurlSearchTool()
    print(tool._run("Capital of Andorra"))
