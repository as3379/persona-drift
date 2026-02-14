#!/usr/bin/env bash
# Run Persona Drift Auditor (uses .venv with Python 3.11)
cd "$(dirname "$0")"
.venv/bin/streamlit run persona_auditor.py
