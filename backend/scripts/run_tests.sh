#!/bin/bash

# Set environment variables for testing
export TESTING=1
export PYTHONPATH=.

# Create test directories if they don't exist
mkdir -p tests/uploads
mkdir -p tests/logs
mkdir -p tests/coverage

# Clean up previous test artifacts
rm -rf tests/uploads/*
rm -rf tests/logs/*
rm -rf tests/coverage/*
rm -f .coverage
rm -f coverage.xml

# Run tests with coverage
pytest \
    --verbose \
    --cov=. \
    --cov-report=term-missing \
    --cov-report=html:tests/coverage/html \
    --cov-report=xml:coverage.xml \
    --junitxml=tests/coverage/junit.xml \
    tests/

# Check test exit code
if [ $? -eq 0 ]; then
    echo "Tests completed successfully!"
    echo "Coverage report available at tests/coverage/html/index.html"
else
    echo "Tests failed!"
    exit 1
fi 