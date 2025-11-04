"""
Test script for generalized DF-VQLS

Usage:
    python test_generalized.py [8x8|16x16|quick]
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.vqls.generalized.test_generalized import (
    test_8x8,
    test_16x16,
    test_8x8_quick
)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "8x8":
            test_8x8()
        elif test_name == "16x16":
            test_16x16()
        elif test_name == "quick":
            test_8x8_quick()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: 8x8, 16x16, quick")
    else:
        # Run quick test by default
        print("Running quick test (8×8 system, 50 iterations)...")
        print("Use 'python test_generalized.py 8x8' for full 8×8 test")
        print("Use 'python test_generalized.py 16x16' for 16×16 test")
        print()
        test_8x8_quick()

