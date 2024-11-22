#!/bin/bash

# Exit on any error
set -e

echo "Starting deployment..."

# Load environment variables
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found"
    exit 1
fi

# Check system compatibility
./scripts/check_compatibility.sh

# Initialize database
echo "Initializing database..."
python3 scripts/init_db.py

# Run tests
echo "Running tests..."
python3 scripts/run_tests.py

# Create release package
echo "Creating release package..."
python3 scripts/package_release.py

# Deploy
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Deploying to production..."
    # Add production deployment steps here
elif [ "$ENVIRONMENT" = "staging" ]; then
    echo "Deploying to staging..."
    # Add staging deployment steps here
else
    echo "Error: Invalid environment specified in .env"
    exit 1
fi

echo "Deployment completed successfully"
