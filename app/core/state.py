"""
Global application state and in-memory registries.
Isolates state from entrypoints to prevent circular dependencies.
"""
from typing import Dict, Any

# Global dictionary to hold ML artifacts in RAM (O(1) access)
ml_models: Dict[str, Any] = {}