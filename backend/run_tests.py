"""
run_tests.py

Convenience runner for Knovera's unified test suite.
Run:
    python run_tests.py
"""

import sys
import unittest
from pathlib import Path

# Ensure root directory is in sys.path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_all.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
