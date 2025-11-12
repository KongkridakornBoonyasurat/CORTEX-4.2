# debug_suppressor.py
"""
Utility to suppress debug output from CORTEX modules
"""

import os
import sys
import warnings
from contextlib import contextmanager
from io import StringIO

# Set environment variables to suppress verbose output
os.environ['CORTEX_VERBOSE'] = '0'
os.environ['CORTEX_DEBUG'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*clone.*detach.*')

@contextmanager
def suppress_stdout():
    """Context manager to temporarily suppress stdout"""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout

@contextmanager 
def suppress_stderr():
    """Context manager to temporarily suppress stderr"""
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr

@contextmanager
def suppress_all_output():
    """Context manager to suppress both stdout and stderr"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def monkey_patch_print():
    """Replace print function to filter debug messages"""
    original_print = print
    
    def silent_print(*args, **kwargs):
        # Filter out common debug patterns
        if args:
            message = str(args[0])
            debug_patterns = [
                ' Using GPU:',
                'Enhanced Neuron',
                ' RobustSelfAwareSynapse42',
                ' EnhancedSynapse42',
                ' Enhanced AstrocyteNetwork',
                'Enhanced Modulator',
                'CORTEX 4.1 BALANCED CONFIG',
                'Key parameter fixes:',
                '- Reduced noise',
                '- Balanced STDP',
                '- Conservative learning',
                '- Proper voltage',
                '- Stable dendritic',
                'CAdEx=ON',
                'Device=cuda',
                'PyTorch=True'
            ]
            
            for pattern in debug_patterns:
                if pattern in message:
                    return  # Skip this print
        
        # If not a debug message, print normally
        original_print(*args, **kwargs)
    
    # Replace the built-in print function
    import builtins
    builtins.print = silent_print
    return original_print

def restore_print(original_print):
    """Restore original print function"""
    import builtins
    builtins.print = original_print

# Auto-apply suppression when module is imported
monkey_patch_print()