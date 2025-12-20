#!/usr/bin/env python3
"""
Verify R setup and ppmSuite installation for PPMx R backend.

This script checks:
1. R is available on the system
2. rpy2 can connect to R
3. ppmSuite package is installed
4. Basic ppmSuite functionality works
"""

import sys
import subprocess


def check_r_available():
    """Check if R is available on system."""
    try:
        result = subprocess.run(['R', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ R found: {version_line}")
            return True
        else:
            print("✗ R not found or returned error")
            return False
    except FileNotFoundError:
        print("✗ R not found in PATH")
        return False


def check_rpy2():
    """Check if rpy2 can be imported and connect to R."""
    try:
        import rpy2
        # rpy2 3.6+ doesn't have __version__ attribute
        try:
            version = rpy2.__version__
        except AttributeError:
            version = "3.6+"
        print(f"✓ rpy2 version {version} imported successfully")
        
        # Try to initialize R interface
        import rpy2.robjects as ro
        r_version = ro.r('R.version.string')[0]
        print(f"✓ rpy2 connected to R: {r_version}")
        return True
    except ImportError as e:
        print(f"✗ rpy2 not installed: {e}")
        print("  Install with: pip install rpy2")
        return False
    except Exception as e:
        print(f"✗ rpy2 failed to connect to R: {e}")
        return False


def check_ppmSuite():
    """Check if ppmSuite package is installed in R."""
    try:
        import rpy2.robjects as ro
        
        # Try to load ppmSuite
        result = ro.r('library(ppmSuite)')
        print("✓ ppmSuite package loaded successfully")
        
        # List available functions
        funcs = ro.r('ls("package:ppmSuite")')
        func_list = list(funcs)
        print(f"  Available functions: {', '.join(func_list[:10])}")
        
        # Check for relevant functions (PPMx, PPMxcpp, etc.)
        ppmx_funcs = [f for f in func_list if 'ppm' in f.lower()]
        if ppmx_funcs:
            print(f"✓ PPMx-related functions found: {', '.join(ppmx_funcs)}")
            return True
        else:
            print("✗ No PPMx-related functions found")
            return False
            
    except Exception as e:
        print(f"✗ ppmSuite package not available: {e}")
        print("\nTo install ppmSuite in R:")
        print("  R -e 'install.packages(\"ppmSuite\", repos=\"https://cloud.r-project.org\")'")
        print("\nOr from within R:")
        print("  install.packages('ppmSuite')")
        return False


def test_ppmSuite_basic():
    """Test basic ppmSuite functionality."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
        import numpy as np
        
        print("\nTesting basic ppmSuite functionality...")
        
        # Create small test dataset
        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 2)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1
        
        # Load library
        ro.r('library(ppmSuite)')
        
        # Check available functions
        funcs = list(ro.r('ls("package:ppmSuite")'))
        ppmx_funcs = [f for f in funcs if 'ppm' in f.lower()]
        print(f"  Trying PPMx function from: {ppmx_funcs}")
        
        # Try PPMx or PPMxcpp (common names)
        for func_name in ['gaussian_ppmx', 'PPMx', 'PPMxcpp', 'ppmx']:
            if func_name in funcs:
                print(f"  Using function: {func_name}")
                
                # Convert data using context manager
                with localconverter(ro.default_converter + numpy2ri.converter):
                    r_X = ro.r.matrix(ro.FloatVector(X.flatten()), nrow=n, ncol=2)
                    r_y = ro.FloatVector(y)
                
                    # Try to run function
                    ppmx_func = ro.r[func_name]
                    result = ppmx_func(
                        y=r_y,
                        X=r_X,
                        cohesion=1,  # Gaussian
                        draws=100,
                        burn=50,
                        thin=1
                    )
                
                print(f"✓ ppmSuite basic test with {func_name} completed successfully")
                print(f"  Result type: {type(result)}")
                return True
        
        print("✗ Could not find suitable PPMx function to test")
        return False
        
    except Exception as e:
        print(f"✗ ppmSuite basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("R and ppmSuite Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("R availability", check_r_available),
        ("rpy2 connectivity", check_rpy2),
        ("ppmSuite package", check_ppmSuite),
        ("ppmSuite basic test", test_ppmSuite_basic),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        try:
            passed = check_func()
            results.append(passed)
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            results.append(False)
        print()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for (name, _), passed in zip(checks, results):
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    if all(results):
        print("\n✓ All checks passed! R backend is ready to use.")
        return 0
    else:
        print("\n✗ Some checks failed. Fix issues above before using R backend.")
        print("\nQuick setup guide:")
        print("1. Install R: https://cran.r-project.org/")
        print("2. Install rpy2: pip install rpy2")
        print("3. Install ppmSuite in R: install.packages('ppmSuite')")
        return 1


if __name__ == '__main__':
    sys.exit(main())
