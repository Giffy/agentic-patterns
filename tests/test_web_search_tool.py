import pytest
from unittest.mock import MagicMock, patch
from tools.web_search_tool import WebSearchTool

@pytest.fixture
def search_tool():
    return WebSearchTool()

def test_web_search_success(search_tool):
    """Test successful search results parsing."""
    mock_results = [
        {"title": "Result 1", "href": "https://example1.com", "body": "Snippet 1"},
        {"title": "Result 2", "href": "https://example2.com", "body": "Snippet 2"},
    ]
    
    with patch("tools.web_search_tool.DDGS") as mock_ddgs:
        mock_instance = mock_ddgs.return_value.__enter__.return_value
        mock_instance.text.return_value = iter(mock_results)
        
        result = search_tool._run("test query")
        
        assert "Title: Result 1" in result
        assert "Link: https://example1.com" in result
        assert "Snippet: Snippet 1" in result
        assert "Title: Result 2" in result

def test_web_search_no_results(search_tool):
    """Test behavior when no results are found."""
    with patch("tools.web_search_tool.DDGS") as mock_ddgs:
        mock_instance = mock_ddgs.return_value.__enter__.return_value
        mock_instance.text.return_value = iter([])
        
        result = search_tool._run("test query")
        
        assert "No results found" in result

def test_web_search_error(search_tool):
    """Test behavior when an error occurs."""
    with patch("tools.web_search_tool.DDGS") as mock_ddgs:
        mock_instance = mock_ddgs.return_value.__enter__.return_value
        mock_instance.text.side_effect = Exception("DDG error")
        
        result = search_tool._run("test query")
        
        assert "Search failed with an unexpected error: DDG error" in result
