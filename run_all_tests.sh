#!/usr/bin/env bash
set -e
echo "Running Capstone Testing Harness on Flask /src Format..."
PYTHONPATH=. python -m pytest tests/test_production_suite.py -v --tb=short
