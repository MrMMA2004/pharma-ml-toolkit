#!/usr/bin/env bash
cd /home/z/my-project/pharma-ml-toolkit
export DISPLAY=:99
exec /home/z/.venv/bin/python -m pytest tests/ src/ -q --junitxml=scratch/junit_final_130.xml > scratch/final_suite_130.log 2>&1
