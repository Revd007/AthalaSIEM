#!/bin/bash

# Exit on any error
set -e

echo "Starting build process..."

# Check if required tools are installed
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting." >&2; exit 1; }
command -v pip3 >/dev/null 2>&1 || { echo "pip3 is required but not installed. Aborting." >&2; exit 1; }

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run system compatibility check
echo "Checking system compatibility..."
./scripts/check_compatibility.sh

# Initialize database
echo "Initializing database..."
python3 scripts/init_db.py

# Run tests
echo "Running tests..."
python3 -m pytest tests/

# Build documentation
echo "Building documentation..."
cd docs && make html && cd ..

# Create distribution package
echo "Creating distribution package..."
python3 setup.py sdist bdist_wheel

echo "Build completed successfully"
