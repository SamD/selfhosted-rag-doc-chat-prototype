#!/usr/bin/env python3
"""
Main entry point for the Retype to Markdown LLM worker.
"""

import os
import sys

from config.env_strategy import get_env_strategy
from workers.retype_llm_worker import main as _retype_main


def main() -> None:
    # Apply environment strategy early
    get_env_strategy().apply()

    # Ensure package imports work when running this file directly
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    _retype_main()


if __name__ == "__main__":
    main()
