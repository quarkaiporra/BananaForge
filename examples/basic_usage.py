#!/usr/bin/env python3
"""Basic usage example for BananaForge."""

import subprocess
import sys
from pathlib import Path


def run_cli_example():
    """Demonstrate advanced BananaForge CLI usage with all transparency and 3MF features."""
    
    # The advanced command line with all transparency and 3MF features
    cmd = [
        "bananaforge", "convert", "./chihiro-4color.png",
        "--output", "./outputs/chihiro",
        "--enable-transparency", 
        "--mixed-precision",
        "--max-materials", "4",
        "--materials", "./materials.csv",
        "--optimize-base-layers",
        "--enable-gradients",
        "--export-format", "3mf,stl,instructions",
        "--bambu-compatible",
        "--include-3mf-metadata"
    ]
    
    print("🚀 Running BananaForge with advanced transparency and 3MF features")
    print(f"📋 Command: {' '.join(cmd)}")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    output_dir = Path("./outputs/chihiro")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("✅ Command completed successfully!")
        print("\n📤 STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\n⚠️  STDERR:")
            print(result.stderr)
            
        # List generated files
        if output_dir.exists():
            files = list(output_dir.glob("*"))
            if files:
                print(f"\n📁 Generated files in {output_dir}:")
                for file in sorted(files):
                    print(f"   📄 {file.name}")
            else:
                print(f"\n📁 Output directory {output_dir} is empty")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with return code {e.returncode}")
        print(f"\n📤 STDOUT:\n{e.stdout}")
        print(f"\n📤 STDERR:\n{e.stderr}")
        return False
    
    except FileNotFoundError:
        print("❌ BananaForge CLI not found. Make sure it's installed and in PATH.")
        print("   Install with: pip install -e .")
        return False
    
    return True


def main():
    """Run the CLI example demonstrating advanced features."""
    print("🍌 BananaForge Advanced CLI Example")
    print("=" * 50)
    print()
    print("This example demonstrates:")
    print("• Transparency-aware color mixing")
    print("• Mixed precision optimization")
    print("• Base layer optimization")
    print("• Gradient enhancement")
    print("• 3MF export with Bambu compatibility")
    print("• Complete metadata embedding")
    print()
    
    success = run_cli_example()
    
    if success:
        print("\n🎉 Example completed successfully!")
        print("\nNext steps:")
        print("1. Check the output directory for generated files")
        print("2. Import the .3mf file into Bambu Studio or PrusaSlicer")
        print("3. Review the swap instructions for optimal printing")
    else:
        print("\n❌ Example failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()