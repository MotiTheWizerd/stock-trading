"""Pytest configuration for the trading-stock-agents project."""
import sys
import os
from pathlib import Path

# Add project root and src to path so we can import trading
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

# Ensure both project root and src are in the path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Debug information
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}...")
