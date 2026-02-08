"""
Convert MATLAB reference data to NumPy format.

Run this after generating MATLAB reference data with:
    matlab -batch "cd tests; generate_matlab_reference"

This script converts .mat files to .npz files for use in Python tests.
"""

from pathlib import Path
import numpy as np

try:
    from scipy.io import loadmat
except ImportError:
    print("Error: scipy is required. Install with: pip install scipy")
    exit(1)

# Directory containing fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Files to convert
MAT_FILES = [
    "matlab_three_circles.mat",
    "matlab_affinity.mat",
    "matlab_laplacian.mat"
]


def convert_mat_to_npz(mat_file):
    """Convert a .mat file to .npz format."""
    mat_path = FIXTURES_DIR / mat_file
    npz_path = FIXTURES_DIR / mat_file.replace(".mat", ".npz")
    
    if not mat_path.exists():
        print(f"⚠ Skipping {mat_file}: file not found")
        return False
    
    try:
        # Load MATLAB file
        data = loadmat(mat_path)
        
        # Remove MATLAB metadata keys
        clean_data = {
            key: val for key, val in data.items()
            if not key.startswith('__')
        }
        
        # Save as NumPy
        np.savez(npz_path, **clean_data)
        
        print(f"✓ Converted {mat_file} -> {npz_path.name}")
        return True
        
    except Exception as e:
        print(f"✗ Error converting {mat_file}: {e}")
        return False


def main():
    """Main conversion function."""
    print("=" * 70)
    print("Converting MATLAB reference data to NumPy format")
    print("=" * 70)
    print()
    
    if not FIXTURES_DIR.exists():
        FIXTURES_DIR.mkdir(parents=True)
        print(f"Created fixtures directory: {FIXTURES_DIR}")
    
    # Convert each file
    converted = 0
    for mat_file in MAT_FILES:
        if convert_mat_to_npz(mat_file):
            converted += 1
    
    print()
    print("=" * 70)
    print(f"Conversion complete: {converted}/{len(MAT_FILES)} files")
    print("=" * 70)
    print()
    
    if converted == len(MAT_FILES):
        print("All reference data converted successfully!")
        print("Run: pytest tests/test_matlab_equivalence.py")
    else:
        print("Some files were not converted.")
        print("Make sure MATLAB reference data was generated first:")
        print("  matlab -batch \"cd tests; generate_matlab_reference\"")


if __name__ == "__main__":
    main()
