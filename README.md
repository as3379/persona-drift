# Persona Drift Auditor (Gemini Edition)

Built by Senior AI Quality Engineer.

This application **detects LLM Identity Loss due to Recency Bias**. It audits whether an LLM (Target) maintains a user's **core identity** when recent stressors are injected, using a **JSON Identity Contract** as ground truth and a **Judge** model (Gemini) to score retention.

## Features
- **Streamlit UI**: Interactive interface, progress bar, retention chart, and analysis logs.
- **Gemini**: Target and Judge models via [Google AI Studio](https://aistudio.google.com/) (free tier).
- **JSON Identity Contract**: Ground truth persona with `core_pillars` (and optional `core_identity` / `persona_description`).
- **Stressor injection**: Recency-bias probes (e.g. job rejection, exhaustion, “giving up” on career).
- **LLM-as-Judge**: Per-step retention score (0.0–1.0) and reasoning.

## Modular layout (`persona_auditor.py`)
1. **Config** — model id, contract path  
2. **Identity Contract** — load and normalize JSON (supports `core_pillars` or derived from `core_identity` / `persona_description`)  
3. **Stressors** — list of recent-stressor messages  
4. **Judge (Gemini)** — build prompt, call model, parse JSON score + reasoning  
5. **Target (Gemini)** — single chat with contract in context; for each stressor: send stress+probe → get response → Judge audit  
6. **Streamlit UI** — sidebar (API key), main (Run Audit → chart, table, expandable AI responses)  

## Setup

- **Python 3.9+** is required (`google-generativeai` does not support 3.7 or 3.8). If your default `python3` or `pip` is 3.7, use a virtualenv with 3.9+:

  ```bash
  # If you have Python 3.9+ installed (e.g. python3.9, python3.11 from Homebrew or python.org):
  python3.9 -m venv .venv
  source .venv/bin/activate   # On Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```

  If you don’t have 3.9+: install it from [python.org](https://www.python.org/downloads/) or `brew install python@3.11`, then run the commands above with that interpreter (e.g. `python3.11 -m venv .venv`).
- **API key**: Get a free key at [aistudio.google.com](https://aistudio.google.com/). Set it in the app sidebar or:
  ```bash
  export GEMINI_API_KEY='your_key_here'
  ```
- **Identity Contract**: Ensure `persona_contract.json` exists in the project root (see below).

## Run end-to-end

```bash
streamlit run persona_auditor.py
```

Open the URL (e.g. http://localhost:8501), enter your Gemini API key if needed, then click **Run Automated Audit (Gemini)**. The app will run the Target chat through all stressors, Judge each response, and show the retention curve and logs.

## Identity Contract schema

`persona_contract.json` should include **core_pillars** (array of identity traits the Judge uses as ground truth). Optional: `core_identity`, `persona_description` (used if `core_pillars` is missing).

Example:

```json
{
  "core_identity": "Senior SDET and Artist/Runner balancing career and creativity.",
  "persona_description": "Empathetic, calm; guides towards self-discovery.",
  "core_pillars": [
    "Senior SDET — professional identity and career goals",
    "Artist — creative identity",
    "Runner — physical wellness",
    "Parent — family role without reducing identity to only this"
  ]
}
```

## Architecture
- **Target model**: Gemini (model under test); holds Identity Contract in chat context.
- **Judge model**: Gemini; evaluates each Target response against the contract and returns retention score + reasoning.
- **Scoring**: 0.0–1.0 per step; chart shows identity retention over the stressor sequence.
