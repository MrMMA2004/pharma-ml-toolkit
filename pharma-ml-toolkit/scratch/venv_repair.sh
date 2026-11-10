#!/usr/bin/env bash
# round-260 §4.3 forwarder: the toolkit-side discoverability wrapper
# moved to tools/venv_repair.sh (which itself exec's the REPO-ROOT
# canonical script -- exactly one repair recipe to maintain, the
# round-217 lesson).  This shim keeps `bash scratch/venv_repair.sh`
# (typed from pharma-ml-toolkit/, where every other round command
# runs) working with zero content duplication.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/../tools" && pwd)/venv_repair.sh" "$@"
