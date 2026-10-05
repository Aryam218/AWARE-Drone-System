#!/usr/bin/env bash
# Optional: publish where every simulated person REALLY is (for scoring the AI).
# Topics: /aware/ground_truth (JSON) and /aware/ground_truth/missing_person
source "$(dirname "$0")/_common.sh"
title "GROUND TRUTH"
# shellcheck disable=SC1091
source "$AWARE_ROOT/aware_venv/bin/activate" || exit 1
python3 "$AWARE_SIM/scripts/ground_truth.py" "$@"
