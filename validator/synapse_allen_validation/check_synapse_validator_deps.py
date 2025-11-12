#!/usr/bin/env python3
"""Check all dependencies for synapse validator"""

import sys

print("Checking dependencies for synapse validator...")
print("="*60)

deps = {
    'numpy': 'pip install numpy',
    'pandas': 'pip install pandas',
    'matplotlib': 'pip install matplotlib',
    'torch': 'pip install torch',
    'allensdk': 'pip install allensdk'
}

missing = []
for package, install_cmd in deps.items():
    try:
        __import__(package)
        print(f" {package}")
    except ImportError:
        print(f" {package} - MISSING")
        print(f"   Install with: {install_cmd}")
        missing.append(package)

print("="*60)

if missing:
    print(f"\n  Missing {len(missing)} dependencies. Install them first!")
    sys.exit(1)
else:
    print("\n All dependencies installed! Ready to run validator.")
    sys.exit(0)