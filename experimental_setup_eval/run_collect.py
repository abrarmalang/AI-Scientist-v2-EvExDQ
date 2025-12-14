#!/usr/bin/env python3
"""
Wrapper script for paper collection.
Runs the paper collector from experimental_setup_eval.collection.collector.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experimental_setup_eval.collection.collector import main

if __name__ == "__main__":
    sys.exit(main())
