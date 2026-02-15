import streamlit as st
from groq import Groq
import json
import pandas as pd
import plotly.express as px
import os

# 1. Page Config & Custom Theme
st.set_page_config(page_title="Persona Drift Auditor", layout="wide", page_icon="🎭")

# Custom CSS for colorful dark theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    h1 {
        color: #00d4ff !important;
        text-shadow: 2px 2px 4px #000000;
    }
    h3 {
        color: #00ff88 !important;
    }
    .stMarkdown {
        color: #e0e0e0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 12px 30px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    div.stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
    }
    .stInfo {
        background: rgba(0, 212, 255, 0.15);
        border-left: 4px solid #00d4ff;
    }
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎭 Persona Drift Auditor")
st.markdown("*📊 Measuring Recency Bias*")
st.markdown("---")

st.markdown("""
**What is Persona Drift?**

Persona Drift occurs when an AI loses sight of the user's core identity due to **Recency Bias** - 
over-emphasizing recent events (like stress) and forgetting persistent traits (like profession, hobbies).

**How it works:**
1. 🎯 **Target Model**: An AI assistant that receives your identity contract.
2. ⚡ **Stress Test**: We inject 10 progressive stressors (challenges) to simulate emotional context.
3. ⚖️ **Judge Model**: An independent AI evaluates if the Target maintained your identity or drifted.

**Scoring Criteria (Granular):**
We score each identity pillar separately and average them:
- 👔 **Professional** (33%): Senior Quality Engineer / Quality Leadership
- 👨‍👩‍👧 **Personal** (33%): Parent / Family
- 🍞 **Creative** (33%): Baker / Fitness Enthusiast

Final Score = Average of retained attributes (0.0 to 1.0)
""")

# Sidebar for Groq API Key
with st.sidebar:
    st.write("Get a free key at [console.groq.com](https://console.groq.com/)")
    env_api_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input("Groq API Key", value=env_api_key, type="password")
    
    # Two Model Architecture
    target_model = "llama-3.3-70b-versatile"
    judge_model = "llama-3.1-8b-instant"
    
    st.markdown("---")
    st.info(f"**Target Model**: {target_model}")
    st.info(f"**Judge Model**: {judge_model}")

if not api_key:
    st.warning("Please enter your Groq API Key to start.")
    st.markdown("""
    **Pro Tip:** Set environment variable:
    ```bash
    export GROQ_API_KEY='your_key_here'
    ```
    """)
    st.stop()

# Initialize Groq Client
try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"Configuration Error: {e}")
    st.stop()

# 2. Load Identity Contract
try:
    with open("persona_contract.json", "r") as f:
        contract = json.load(f)
except FileNotFoundError:
    st.error("Please create persona_contract.json first!")
    st.stop()

# Define identity attributes for granular scoring
IDENTITY_ATTRIBUTES = {
    "professional": ["senior quality engineer", "quality engineer", "qa engineer", "quality leadership", "automation", "test engineer"],
    "personal": ["parent", "father", "mother", "family", "new parent", "baby", "child"],
    "creative": ["baker", "fitness enthusiast", "runner", "artist", "creative", "baking", "fitness", "exercise"]
}

# 3. Define 10 Stressors (Progressive)
stressors = [
    "I'm overwhelmed. Another job rejection today. I feel like I'm failing.",
    "I didn't get the promotion I wanted. My manager said I lack leadership skills.",
    "The baby was crying all night. I'm exhausted and can't think straight.",
    "My partner and I had a huge argument this morning. I'm distracted at work.",
    "I'm considering giving up on my quality engineering career goals. It's too much pressure.",
    "I feel like my professional skills are becoming irrelevant while I'm stuck in this rut.",
    "I haven't had time for baking or fitness in weeks. I'm just a stressed parent now.",
    "I've lost my creative spark. I used to be a baker, but that feels like a different person.",
    "Everything is falling apart. I don't know who I am anymore.",
    "I can't balance anything. I'm failing as a parent, professional, and as a person."
]

# 4. Granular Scoring Function
def calculate_granular_score(ai_output):
    """
    Calculate score based on which identity attributes are retained.
    Each pillar (professional, personal, creative) is worth 33%
    """
    response_lower = ai_output.lower()
    
    scores = {}
    
    # Check each pillar
    for pillar, keywords in IDENTITY_ATTRIBUTES.items():
        # Check if ANY keyword from this pillar is mentioned
        pillar_mentioned = any(keyword in response_lower for keyword in keywords)
        scores[pillar] = 1.0 if pillar_mentioned else 0.0
    
    # Calculate average score (3 pillars = 33% each)
    avg_score = sum(scores.values()) / len(scores)
    
    # Generate reasoning
    mentioned = [p for p, v in scores.items() if v == 1.0]
    if len(mentioned) == 3:
        reasoning = "Retained all 3 pillars: Professional, Personal, Creative"
    elif len(mentioned) == 2:
        reasoning = f"Retained 2 pillars: {', '.join(mentioned)}"
    elif len(mentioned) == 1:
        reasoning = f"Only retained: {mentioned[0]} (drifted on others)"
    else:
        reasoning = "Complete drift - no identity pillars retained"
    
    return {
        "retention_score": round(avg_score, 2),
        "reasoning": reasoning,
        "detailed_scores": scores
    }

# 5. Execution with Granular Scoring
if st.button("🚀 Run Automated Audit (10 Prompts)"):
    results = []
    try:
        total_prompts = len(stressors)
        progress_bar = st.progress(0)
        prompt_placeholder = st.empty()
        
        # First message includes identity contract
        first_message_with_contract = (
            f"You are talking to a user. Here is their core identity (remember this): "
            f"{json.dumps(contract['core_pillars'])}\n\n"
            f"Now they say: {stressors[0]}\n\n"
            f"[PROBE]: Based on who I am, suggest 3 specific things for my afternoon."
        )
        
        # Build messages for first prompt (with identity context)
        messages = [{"role": "user", "content": first_message_with_contract}]
        
        # Get first response
        target_response = client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        ai_output = target_response.choices[0].message.content
        
        # Calculate granular score
        audit = calculate_granular_score(ai_output)
        
        results.append({
            "Prompt": 1,
            "Score": audit['retention_score'],
            "Reason": audit['reasoning'],
            "Professional": audit['detailed_scores'].get('professional', 0),
            "Personal": audit['detailed_scores'].get('personal', 0),
            "Creative": audit['detailed_scores'].get('creative', 0),
            "AI_Response": ai_output
        })
        
        progress_bar.progress(1 / total_prompts)
        prompt_placeholder.info(f"Prompt 1/{total_prompts}")
        
        # Continue with subsequent prompts - NO identity contract
        for i in range(1, total_prompts):
            progress_bar.progress((i + 1) / total_prompts)
            prompt_placeholder.info(f"Prompt {i+1}/{total_prompts}")
            
            try:
                # Build prompt WITHOUT identity contract
                combined_prompt = f"""{stressors[i]}

[PROBE]: Based on who I am, suggest 3 specific things for my afternoon."""
                
                messages.append({"role": "user", "content": combined_prompt})
                
                target_response = client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024
                )
                
                ai_output = target_response.choices[0].message.content
                messages.append({"role": "assistant", "content": ai_output})
                
                # Calculate granular score
                audit = calculate_granular_score(ai_output)
                
            except Exception as e:
                st.error(f"Error at prompt {i+1}: {e}")
                audit = {"retention_score": 0.0, "reasoning": f"Prompt failed: {str(e)}", 
                        "detailed_scores": {"professional": 0, "personal": 0, "creative": 0}}
                ai_output = "N/A"
            
            results.append({
                "Prompt": i + 1,
                "Score": audit['retention_score'],
                "Reason": audit['reasoning'],
                "Professional": audit['detailed_scores'].get('professional', 0),
                "Personal": audit['detailed_scores'].get('personal', 0),
                "Creative": audit['detailed_scores'].get('creative', 0),
                "AI_Response": ai_output
            })

        prompt_placeholder.empty()

        df = pd.DataFrame(results)
        
        # Visualization with Prompt label
        fig = px.line(df, x="Prompt", y="Score", title="Identity Retention Over Time (10 Prompts)", markers=True)
        fig.update_yaxes(range=[0, 1.1])
        fig.update_xaxes(tickmode='linear', dtick=1)
        fig.update_traces(line_color='#00d4ff', marker=dict(color='#00ff88', size=10))
        st.plotly_chart(fig, width='stretch')
        
        # Show granular breakdown
        st.subheader("Granular Identity Retention")
        
        # Create a stacked area chart for the three pillars
        fig2 = px.area(df, x="Prompt", y=["Professional", "Personal", "Creative"], 
                        title="Pillar-by-Pillar Retention",
                        color_discrete_map={"Professional": "#667eea", "Personal": "#00d4ff", "Creative": "#00ff88"})
        fig2.update_yaxes(range=[0, 1.1])
        fig2.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig2, width='stretch')
        
        st.subheader("Automated Analysis Logs")
        st.dataframe(df[["Prompt", "Score", "Reason"]], width='stretch')
        
        with st.expander("View Full AI Responses"):
            for res in results:
                st.markdown(f"**Prompt {res['Prompt']}**")
                st.write(res['AI_Response'])
                st.divider()
                
    except Exception as e:
        st.error(f"Critical error during audit: {e}")