#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export DISPLAY=:99
exec python -m pytest tests/ src/ -q --junitxml=scratch/junit_final_130.xml > scratch/final_suite_130.log 2>&1
