"""
Snake Game Environment Package
Contains environment definitions for the Snake game
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .snake_env import SnakeEnv

__all__ = ['SnakeEnv']
