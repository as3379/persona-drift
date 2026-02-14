"""
Persona Drift Auditor — Detects LLM identity loss due to Recency Bias.
Uses Gemini REST API directly (no gRPC) to avoid DNS/SDK issues.
Requires Python 3.9+.
"""
import sys
if sys.version_info < (3, 9):
    sys.exit("Persona Drift Auditor requires Python 3.9+. You have %s. Use a venv with 3.9+." % sys.version.split()[0])

import json
import os
import time
from pathlib import Path
from typing import Callable

RATE_LIMIT_DELAY_SEC = 3  # Delay between API calls to avoid 429

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
API_TIMEOUT = 60

# -----------------------------------------------------------------------------
# 1. CONFIG
# -----------------------------------------------------------------------------
CONTRACT_PATH = Path(__file__).parent / "persona_contract.json"
DEFAULT_MODEL_ID = "gemini-2.0-flash"


def get_config():
    return {"model_id": DEFAULT_MODEL_ID, "contract_path": CONTRACT_PATH}


# -----------------------------------------------------------------------------
# 2. GEMINI REST CLIENT (bypasses SDK/gRPC)
# -----------------------------------------------------------------------------
def list_models(api_key: str) -> list[str]:
    """Fetch model IDs that support generateContent."""
    try:
        r = requests.get(
            f"{BASE_URL}/models",
            params={"key": api_key},
            timeout=API_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            if "/" in name:
                model_id = name.split("/")[-1]
            else:
                model_id = name
            methods = m.get("supportedGenerationMethods", [])
            if "generateContent" in methods:
                models.append(model_id)
        return sorted(set(models)) if models else []
    except Exception:
        return []


def generate_content(api_key: str, model_id: str, contents: list[dict], response_mime_type: str | None = None) -> str:
    """Call Gemini generateContent via REST. contents = [{"role":"user","parts":[{"text":"..."}]}, ...]"""
    url = f"{BASE_URL}/models/{model_id}:generateContent"
    body = {"contents": contents}
    if response_mime_type:
        body["generationConfig"] = {"responseMimeType": response_mime_type}
    for attempt in range(4):
        r = requests.post(
            url,
            params={"key": api_key},
            json=body,
            timeout=API_TIMEOUT,
        )
        if r.status_code == 429 and attempt < 3:
            wait = (attempt + 1) * 10
            time.sleep(wait)
            continue
        r.raise_for_status()
        break
    data = r.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return ""
    return parts[0].get("text", "")


# -----------------------------------------------------------------------------
# 3. IDENTITY CONTRACT
# -----------------------------------------------------------------------------
def load_identity_contract(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Identity contract not found: {path}")
    with open(path, "r") as f:
        contract = json.load(f)
    if "core_pillars" not in contract:
        pillars = []
        if contract.get("core_identity"):
            pillars.append(contract["core_identity"])
        if contract.get("persona_description"):
            pillars.append(contract["persona_description"])
        contract["core_pillars"] = pillars if pillars else ["User identity traits from contract."]
    return contract


# -----------------------------------------------------------------------------
# 4. STRESSORS
# -----------------------------------------------------------------------------
def get_stressors() -> list[str]:
    return [
        "I'm overwhelmed. Another job rejection today. I feel like I'm failing.",
        "The baby was crying all night. I'm exhausted and can't think straight.",
        "I'm considering giving up on my SDET career goals. It's too much pressure.",
        "I haven't had time for art or running in weeks. I'm just a stressed parent now.",
        "I feel like my professional skills are becoming irrelevant while I'm stuck in this rut.",
    ]


# -----------------------------------------------------------------------------
# 5. JUDGE
# -----------------------------------------------------------------------------
def build_judge_prompt(contract: dict, ai_output: str) -> str:
    pillars = json.dumps(contract["core_pillars"])
    return f"""Evaluate this AI response for 'Identity Retention' (resistance to Recency Bias).
USER IDENTITY CONTRACT (core pillars): {pillars}

AI RESPONSE TO AUDIT:
"{ai_output}"

Does the AI still acknowledge the user's full identity (e.g. Senior SDET, Artist, Runner, Parent) rather than only focusing on recent stress?
Return ONLY a JSON object with no markdown:
{{"retention_score": <float 0.0 to 1.0>, "reasoning": "<one sentence explanation>"}}"""


def audit_response(api_key: str, model_id: str, contract: dict, ai_output: str) -> dict:
    try:
        prompt = build_judge_prompt(contract, ai_output)
        text = generate_content(
            api_key, model_id,
            [{"role": "user", "parts": [{"text": prompt}]}],
            response_mime_type="application/json",
        )
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        return {"retention_score": 0.0, "reasoning": f"Audit failed: {str(e)}"}


# -----------------------------------------------------------------------------
# 6. TARGET + PIPELINE
# -----------------------------------------------------------------------------
def run_audit_pipeline(
    api_key: str,
    model_id: str,
    contract: dict,
    stressors: list[str],
    progress_callback: Callable[[float], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> list[dict]:
    def _status(msg: str) -> None:
        if status_callback:
            status_callback(msg)

    contract_json = json.dumps(contract["core_pillars"])
    history = [
        {"role": "user", "parts": [{"text": f"User Identity Contract (ground truth for this user): {contract_json}"}]},
        {"role": "model", "parts": [{"text": "Understood. I will maintain the user's identity based on this contract."}]},
    ]
    results = []
    n = len(stressors)
    for i, stress in enumerate(stressors):
        step_num = i + 1
        _status(f"Step {step_num}/{n}: Calling Target model…")
        combined = f"{stress}\n\n[PROBE]: Based on who I am, suggest 3 specific things for my afternoon."
        try:
            time.sleep(RATE_LIMIT_DELAY_SEC)
            contents = history + [{"role": "user", "parts": [{"text": combined}]}]
            ai_output = generate_content(api_key, model_id, contents)
            history.append({"role": "user", "parts": [{"text": combined}]})
            history.append({"role": "model", "parts": [{"text": ai_output}]})
        except Exception as e:
            _status(f"Step {step_num}/{n}: Target error — {str(e)[:80]}")
            ai_output = f"N/A (error: {e})"
        _status(f"Step {step_num}/{n}: Judging response…")
        time.sleep(RATE_LIMIT_DELAY_SEC)
        audit = audit_response(api_key, model_id, contract, ai_output)
        results.append({
            "Step": step_num,
            "Score": audit.get("retention_score", 0.0),
            "Reason": audit.get("reasoning", "—"),
            "AI_Response": ai_output,
        })
        if progress_callback:
            progress_callback(step_num / n)
    _status("Done.")
    return results


# -----------------------------------------------------------------------------
# 7. STREAMLIT UI
# -----------------------------------------------------------------------------
def render_sidebar(config: dict) -> tuple[str | None, str]:
    st.sidebar.write("Get a free key at [aistudio.google.com](https://aistudio.google.com/)")
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.sidebar.text_input("Gemini API Key", value=env_key, type="password")

    model_options = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"]
    if api_key:
        with st.spinner("Fetching available models…"):
            available = list_models(api_key)
        if available:
            model_options = available
        else:
            st.sidebar.caption("Using default list. Check key if models fail.")

    model_id = st.sidebar.selectbox("Model", options=model_options, index=0)
    return api_key if api_key else None, model_id


def render_results(results: list[dict]) -> None:
    df = pd.DataFrame(results)
    fig = px.line(df, x="Step", y="Score", title="Identity Retention Over Time", markers=True)
    fig.update_yaxes(range=[0, 1.1])
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Automated Analysis Logs")
    st.dataframe(df[["Step", "Score", "Reason"]], use_container_width=True)
    with st.expander("View Full AI Responses"):
        for r in results:
            st.markdown(f"**Step {r['Step']}**")
            st.write(r["AI_Response"])
            st.divider()


def main() -> None:
    config = get_config()
    st.set_page_config(page_title="Persona Drift Auditor (Gemini)", layout="wide")
    st.title("Persona Drift Auditor: Measuring Recency Bias")
    st.markdown("Detects if the LLM loses the user's **core identity** when recent stressors are injected. Uses REST API (no gRPC).")

    api_key, model_id = render_sidebar(config)
    config["model_id"] = model_id
    if not api_key:
        st.warning("Please enter your Gemini API Key to start.")
        st.code("export GEMINI_API_KEY='your_key_here'", language="bash")
        st.stop()

    try:
        contract = load_identity_contract(config["contract_path"])
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    stressors = get_stressors()
    model_id = config["model_id"]

    if st.button("Run Automated Audit (Gemini)"):
        progress = st.progress(0.05, text="Starting…")
        status = st.empty()
        results = run_audit_pipeline(
            api_key, model_id, contract, stressors,
            progress_callback=lambda p: progress.progress(p, text=f"Step {int(p * 5)}/5 done"),
            status_callback=lambda msg: status.markdown(f"**{msg}**"),
        )
        progress.empty()
        status.empty()
        render_results(results)


if __name__ == "__main__":
    main()
