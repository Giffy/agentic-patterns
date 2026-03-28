# Agentic Patterns - Tests

- `test_unified_entry_point.py`: Core logic for the unified agent.
- `test_main_unified.py`: CLI interface and result presentation.
- `test_web_search_tool.py`: Specialized tool tests.

## How to run tests

To run all tests in this repository, use the following command:

```powershell
uv run pytest tests/
```

To run a specific test file:

```powershell
uv run pytest tests/test_unified_entry_point.py
uv run pytest tests/test_main_unified.py
```

### Dependencies

Tests require `pytest`, which is included in the `dev` dependency group in `pyproject.toml`.
