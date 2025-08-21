#!/usr/bin/env python3
"""
Project entrypoint.

Delegates to src.main_batch for CSV-driven creator crawling and analysis.
"""

import sys


def main() -> int:
    try:
        from src.main_batch import run_sync
    except Exception as e:
        print(f"Failed to import src.main_batch: {e}")
        return 1
    return run_sync()


if __name__ == "__main__":
    sys.exit(main())



