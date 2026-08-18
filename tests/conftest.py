"""
conftest.py — Pytest Configuration for the Research Agent Test Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THIS FILE?
    When pytest runs tests from the `tests/` subdirectory, Python's import
    system doesn't automatically include the parent directory (research_agent_project/)
    in sys.path. This means `from memory_manager import MemoryManager` would fail
    with ModuleNotFoundError.

    conftest.py is pytest's configuration file — any fixtures or settings defined
    here are automatically available to all test files in the same directory and
    subdirectories. Placing it at the tests/ level and adding the parent to sys.path
    solves the import problem for the entire test suite.
"""

import sys                           # for manipulating Python's import path
import os                            # for resolving absolute paths
from pathlib import Path             # clean cross-platform path handling

# Add the project root (one level above the tests/ directory) to sys.path.
# This allows all test files to do `from memory_manager import MemoryManager`
# without installing the project as a package.
project_root = str(Path(__file__).parent.parent)  # /path/to/research_agent_project
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # insert at position 0 for highest priority

# Load .env file if present — provides OPENAI_API_KEY etc. for any tests that need it.
# Tests that DON'T need API keys (the majority) will just ignore the env vars.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(project_root) / ".env", override=False)
except ImportError:
    pass  # dotenv not installed yet (e.g., bare environment) — tests still run
