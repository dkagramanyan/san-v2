#!/usr/bin/env python3
"""
Temporarily patches the codebase to disable custom CUDA ops.
This is for TESTING ONLY to identify if custom ops are causing slowdowns.

Usage:
    python disable_custom_ops.py enable   # Disable custom CUDA ops (use PyTorch fallback)
    python disable_custom_ops.py disable  # Re-enable custom CUDA ops
"""

import sys
import os

# Files to patch
FILES_TO_PATCH = [
    ('torch_utils/ops/bias_act.py', "impl='cuda'", "impl='ref'"),
    ('torch_utils/ops/upfirdn2d.py', "impl='cuda'", "impl='ref'"),
    ('torch_utils/ops/filtered_lrelu.py', "impl='cuda'", "impl='ref'"),
]

def patch_files(use_ref=True):
    """Patch files to use ref or cuda implementation."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for filepath, cuda_str, ref_str in FILES_TO_PATCH:
        full_path = os.path.join(base_dir, filepath)
        
        if not os.path.exists(full_path):
            print(f"Warning: {filepath} not found, skipping...")
            continue
        
        with open(full_path, 'r') as f:
            content = f.read()
        
        if use_ref:
            # Replace cuda with ref
            new_content = content.replace(cuda_str, ref_str)
            change_desc = f"CUDA -> ref"
        else:
            # Replace ref with cuda
            new_content = content.replace(ref_str, cuda_str)
            change_desc = f"ref -> CUDA"
        
        if content != new_content:
            with open(full_path, 'w') as f:
                f.write(new_content)
            print(f"✓ Patched {filepath}: {change_desc}")
        else:
            print(f"  {filepath}: no changes needed")

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('enable', 'disable'):
        print(__doc__)
        sys.exit(1)
    
    if sys.argv[1] == 'enable':
        print("Disabling custom CUDA ops (using PyTorch reference implementations)...")
        print("This will be SLOWER but helps identify if custom ops are the problem.\n")
        patch_files(use_ref=True)
        print("\nCustom CUDA ops DISABLED. Training will use PyTorch fallback.")
        print("If training is still slow, the issue is NOT with custom CUDA ops.")
        print("If training is faster, the custom CUDA ops need fixing (GCC/CUDA compatibility).")
    else:
        print("Re-enabling custom CUDA ops...\n")
        patch_files(use_ref=False)
        print("\nCustom CUDA ops RE-ENABLED.")

if __name__ == "__main__":
    main()
