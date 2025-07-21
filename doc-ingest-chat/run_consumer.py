#!/usr/bin/env python3
"""
Main entry point for the consumer worker.
"""
from config.env_strategy import get_env_strategy
get_env_strategy().apply()

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workers.consumer_worker import main

if __name__ == "__main__":
    main() 