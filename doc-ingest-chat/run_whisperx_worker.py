#!/usr/bin/env python3
"""
Main entry point for the WhisperX worker.
Sets up the path so internal imports work.
"""

import os
import sys


def main() -> None:
    # Ensure package imports work when running this file directly
    # This allows 'from utils.trace_utils import ...' to work
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from workers.whisperx_worker import main as _whisper_main
    _whisper_main()


if __name__ == "__main__":
    main()
