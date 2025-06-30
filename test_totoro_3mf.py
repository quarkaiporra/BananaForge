#!/usr/bin/env python3
"""Test 3MF export with totoro image."""

import sys
import subprocess

def main():
    """Test 3MF export functionality."""
    print("🧪 Testing BananaForge 3MF Export with Totoro Image")
    print("=" * 60)
    
    # Test command
    cmd = [
        sys.executable, "-m", "bananaforge.cli", "convert",
        "./examples/totoro.jpg",
        "--output", "./examples/outputs/totoro_3mf_test",
        "--max-layers", "10",
        "--max-materials", "3", 
        "--export-format", "3mf,stl,instructions",
        "--bambu-compatible",
        "--project-name", "totoro_3mf",
        "--iterations", "500",  # Reduced for quick test
        "--device", "cpu"  # Use CPU for reliability
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ 3MF export test successful!")
        else:
            print("❌ 3MF export test failed!")
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    main()