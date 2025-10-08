#!/usr/bin/env python3
"""
Test runner script for AI Assignment Evaluator tests.
Provides easy commands to run different test categories.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found. Make sure pytest is installed: pip install pytest pytest-asyncio pytest-mock")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run AI Assignment Evaluator tests")
    parser.add_argument(
        "test_type",
        choices=["all", "integration", "e2e", "celery", "unit", "quick"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Run with verbose output"
    )
    parser.add_argument(
        "--file",
        help="Run specific test file"
    )
    
    args = parser.parse_args()
    
    # Change to backend directory
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term"])
    
    # Add verbose if requested
    if args.verbose:
        cmd.append("-v")
    
    # Determine test selection
    if args.file:
        cmd.append(f"tests/{args.file}")
        description = f"specific test file: {args.file}"
    elif args.test_type == "all":
        cmd.append("tests/")
        description = "all tests"
    elif args.test_type == "quick":
        cmd.extend(["-m", "not slow", "--maxfail=5"])
        description = "quick tests (excluding slow tests)"
    else:
        cmd.extend(["-m", args.test_type])
        description = f"{args.test_type} tests"
    
    # Run the tests
    success = run_command(cmd, description)
    
    if success:
        print(f"\n🎉 All {description} passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print(f"\n💥 Some {description} failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
