import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_chat import message as st_message
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import io
import base64
import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ---------------- Load Groq API ----------------
load_dotenv()

groq_chat = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY", "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"),
    model_name="llama-3.1-8b-instant",
    temperature=0.9,
    max_tokens=200
)

# ---------------- Web Search (DuckDuckGo fallback) ----------------
def search(query, max_results=5):
    try:
        url = "https://duckduckgo.com/html/"
        params = {"q": query}
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(res.text, "html.parser")
        results = []
        for a in soup.select(".result__a")[:max_results]:
            title = a.get_text()
            link = a.get("href")
            snippet_tag = a.find_parent().select_one(".result__snippet")
            snippet = snippet_tag.get_text() if snippet_tag else ""
            results.append({"title": title, "link": link, "snippet": snippet})
        return {"results": results}
    except Exception as e:
        return {"results": [], "error": str(e)}

# ---------------- Load default data ----------------
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_enterprise_data.csv")

if "user_df" not in st.session_state:
    st.session_state.user_df = None

uploaded_file = st.file_uploader("Upload your own enterprise data (CSV)", type="csv")
if uploaded_file:
    st.session_state.user_df = pd.read_csv(uploaded_file)
    st.success("Custom data uploaded! Agent will use this data.")

df = st.session_state.user_df if st.session_state.user_df is not None else load_data()

# ---------------- Session State ----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = None
# Store last structured answer PER scenario to ground follow-ups
if "last_structured_by_scenario" not in st.session_state:
    st.session_state.last_structured_by_scenario = {}

# ---------------- Helper Functions ----------------
def compute_dynamic_scores(df):
    df = df.copy()
    df['support_tickets'] = df['support_tickets'].fillna(0)
    df['sentiment'] = df['sentiment'].fillna(0.5)
    df['usage'] = df['usage'].fillna(0)
    df['anomaly_flag'] = df['anomaly_flag'].fillna(0)
    scaler = StandardScaler()
    metrics = scaler.fit_transform(df[['usage','support_tickets','sentiment','anomaly_flag']])
    weights = np.array([0.4,0.3,0.2,0.1])
    df['dynamic_score'] = metrics.dot(weights)
    return df

def detect_anomalies(df, n_clusters=3):
    df = df.copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['usage','support_tickets','sentiment']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    df['cluster'] = kmeans.labels_
    cluster_means = df.groupby('cluster')['support_tickets'].mean()
    extreme_clusters = cluster_means[cluster_means > cluster_means.mean() + cluster_means.std()].index.tolist()
    df['anomaly_flag'] = df['cluster'].apply(lambda x: 1 if x in extreme_clusters else 0)
    return df

def predict_trends(df, metric='usage'):
    df_sorted = df.sort_values(['product','feature'])
    trend_fig, ax = plt.subplots()
    for _, group in df_sorted.groupby(['product','feature']):
        X = np.arange(len(group)).reshape(-1,1)
        y = group[metric].values
        if len(y) < 2:
            continue
        model = LinearRegression().fit(X,y)
        y_pred = model.predict(X)
        ax.plot(X, y_pred, label=f"{group['feature'].iloc[0]} ({group['product'].iloc[0]})")
    ax.set_title(f"Predicted Trend for {metric}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    return trend_fig

def render_fig_in_chat(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"![plot](data:image/png;base64,{img_base64})"

def _extract_llm_text(response_obj) -> str:
    """
    Groq LangChain objects may vary; normalize to a clean string.
    """
    # Try common attributes first
    if hasattr(response_obj, "content") and isinstance(response_obj.content, str):
        return response_obj.content.strip()
    # Try .message.content
    if hasattr(response_obj, "message") and hasattr(response_obj.message, "content"):
        return str(response_obj.message.content).strip()
    # Try JSON fallback
    try:
        data = json.loads(getattr(response_obj, "json", lambda: "{}")())
        if isinstance(data, dict) and "content" in data:
            return str(data["content"]).strip()
    except Exception:
        pass
    # Fallback: cast to str
    return str(response_obj).strip()

# ---------------- Scenario Classifier Function ----------------
def classify_scenario(user_text: str, last_scenario: str = None) -> str:
    """
    Use Groq to classify user input into one of the known scenarios.
    Returns one of: Risk Synthesis, Opportunity Discovery, Edge Case,
    Stretch Scenario, Feature Health, Unknown

    If the user input is a continuation (e.g., "continue", "complete"),
    it will reuse the last_scenario instead of switching.
    """

    # --- Continuation Guard ---
    continuation_phrases = ["continue", "go on", "complete", "carry on", "elaborate", "finish", "keep going", "more", "next"]
    if any(p in user_text.lower() for p in continuation_phrases) and last_scenario:
        return last_scenario

    # --- Groq Classification Prompt ---
    classification_prompt = f"""
You are a strict scenario classifier for a Business Intelligence Agent.

Task:
Classify the user's request into exactly ONE of these scenarios:
- Risk Synthesis
- Opportunity Discovery
- Edge Case
- Stretch Scenario
- Feature Health
- Unknown  (use ONLY if it clearly does not fit any category)

Supporting hints and examples:
- Risk Synthesis: "Surface internal usage patterns, reliability issues, adoption blockers", "recommend steps to improve reliability".
- Opportunity Discovery: "Where can we launch a new product?", "top features to double-down on".
- Edge Case: Data ambiguity / conflicting signals. e.g., "Surface any insights about new product" with unclear context.
- Stretch Scenario: Challenge internal knowledge / bold predictions. e.g., "Predict a feature to leapfrog competition", "bold GTM plan + risks".
- Feature Health: Any ask about feature health, adoption, sentiment in general.

Rules:
- Never invent a new label.
- If the user message is vague or ambiguous, return "Unknown".
- Do not output explanations, only the label.
- If the user asks to continue a previous response, return the previous scenario (already handled in code).

User request: "{user_text}"
"""
    try:
        response = groq_chat.invoke(classification_prompt)
        label = _extract_llm_text(response)
        valid_labels = {
            "Risk Synthesis",
            "Opportunity Discovery",
            "Edge Case",
            "Stretch Scenario",
            "Feature Health",
            "Unknown"
        }
        return label if label in valid_labels else "Unknown"
    except Exception:
        return "Unknown"

# ---------------- Follow-up Q&A Mode ----------------
def continue_conversation(user_text: str, last_structured_answer: str, scenario: str, context: str = "") -> str:
    """
    Use Groq to continue the conversation based on the last structured answer
    and the user's new question. Avoid repeating the full block.
    """
    followup_prompt = f"""
You are a helpful, concise BI Analyst Agent.

Recent conversation context:
{context}

Current scenario: {scenario}

The last structured answer (DO NOT repeat it; use it as context):
{last_structured_answer}

The user now asks: "{user_text}"

Instructions:
- Answer the exact question asked, using the last structured answer and the dataset context implied.
- Do NOT paste the entire previous block again.
- If the user asks about a specific term (e.g., "risk candidate"), define it simply and tie it to this dataset.
- If the user asks "why", give 2-4 crisp bullets grounded in the previous reasoning.
- If information is insufficient, say what is missing and suggest a next step (e.g., "filter by region", "pull top 5 features by dynamic_score").
- Keep your response short, precise, and conversational.
"""
    try:
        response = groq_chat.invoke(followup_prompt)
        return _extract_llm_text(response)
    except Exception as e:
        return f"⚠️ Follow-up generation failed: {e}"

# ---------------- Core Function with Groq Integration ----------------
def summarize_and_tabulate(scenario, df, context=""):
    summary, table, extra_outputs, structured, figures = "", pd.DataFrame(), {}, "", []
    df = compute_dynamic_scores(df)
    df = detect_anomalies(df)

    def with_context(prompt):
        return f"""
You are an Autonomous BI Agent.

Recent conversation context:
{context}

User has triggered scenario: {scenario}.
Your job: answer based on context + dataset insights.

{prompt}
"""

    # ---------------- Scenario Analysis ----------------
    if scenario == "Risk Synthesis":
        risk_df = df[(df['anomaly_flag']==1)|((df['support_tickets']>10)&(df['sentiment']<0.5))]
        summary = f"⚠️ {len(risk_df)} risky features detected across {risk_df['region'].nunique()} regions."
        table = (risk_df.sort_values('dynamic_score', ascending=False)
                 [['product','feature','region','support_tickets','sentiment','dynamic_score']].head(5))

        fig_scatter, ax1 = plt.subplots()
        sns.scatterplot(data=risk_df, x="support_tickets", y="sentiment", hue="region",
                        size="dynamic_score", ax=ax1, s=100)
        ax1.set_title("Support Tickets vs Sentiment (Risk Features)")
        figures.append(fig_scatter)
        figures.append(predict_trends(risk_df))

        groq_prompt = with_context(f"""
Dataset risk candidates:
{table.to_dict(orient='records')}

Tasks:
1) Name the 2–3 most concerning risks and WHY (one line each).
2) Call out any surprising correlations (region, product, adoption).
3) Predict near-term implications if ignored (one line each).
4) Give 3–5 concrete actions (triage, outreach, product fix, docs).
Format in short sections; keep it executive and actionable.
""")
        try:
            response = groq_chat.invoke(groq_prompt)
            llm_text = _extract_llm_text(response)
        except Exception as e:
            llm_text = f"⚠️ LLM insight generation failed: {e}"

        structured = f"### 📌 AI Risk Insights\n{llm_text}"

    elif scenario == "Opportunity Discovery":
        filtered = df[(df['usage']>120)&(df['sentiment']>0.8)&(df['support_tickets']<3)]
        summary = f"🚀 {len(filtered)} high adoption features detected. Showing top 5."
        table = filtered[['product','feature','region','dynamic_score']].sort_values("dynamic_score", ascending=False).head(5)

        groq_prompt = with_context(f"""
Dataset opportunity candidates:
{table.to_dict(orient='records')}

Tasks:
1) Identify top 2–3 growth levers and why.
2) Predict scaling implications.
3) Recommend 2–3 specific bets (campaigns, partnerships, feature doubling).
Be concise and specific.
""")
        try:
            response = groq_chat.invoke(groq_prompt)
            llm_text = _extract_llm_text(response)
        except Exception as e:
            llm_text = f"⚠️ LLM insight generation failed: {e}"

        structured = f"### 📌 AI Opportunity Insights\n{llm_text}"

    elif scenario == "Edge Case":
        filtered = df[(df['usage'] > 100) & (df['sentiment'] < 0.5)]
        if filtered.empty:
            summary = "✅ No unusual patterns detected in the current dataset."
            structured = ""
        else:
            summary = f"⚖️ {len(filtered)} feature(s) show unusual patterns."
            table = filtered[['product','feature','region','usage','sentiment']].head(5)

            groq_prompt = with_context(f"""
Edge case features:
{table.to_dict(orient='records')}

Tasks:
1) Explain what makes these patterns unusual.
2) Give 2–3 plausible causes.
3) Suggest 2 quick validation steps.
4) Recommend: prioritize vs. monitor.
Short, crisp bullets.
""")
            try:
                response = groq_chat.invoke(groq_prompt)
                llm_text = _extract_llm_text(response)
            except Exception as e:
                llm_text = f"⚠️ LLM insight generation failed: {e}"

            structured = f"### 📌 AI Edge Case Insights\n{llm_text}"

            fig_scatter, ax1 = plt.subplots(figsize=(8, 5))
            sns.scatterplot(
                data=filtered, x="usage", y="sentiment", hue="region",
                size="support_tickets", palette="tab10", ax=ax1, s=100, alpha=0.8
            )
            ax1.set_title("Usage vs Sentiment (Unusual Features)")
            figures.append(fig_scatter)

    elif scenario == "Stretch Scenario":
        candidates = df[(df['usage']>110)&(df['sentiment']>0.7)].sort_values("dynamic_score", ascending=False).head(3)
        feature_ideas = candidates['feature'].tolist()
        search_results = search("Azure competitors AWS GCP disruptive cloud features developer forum trends 2025")
        external_trends = "\n".join([f"- {r['title']}: {r['snippet']}" for r in search_results.get("results", [])[:5]])

        groq_prompt = with_context(f"""
Internal candidates: {feature_ideas}
External competitor trends:
{external_trends}

Tasks:
1) Connect internal strengths with external gaps (2–3 bullets).
2) Predict likely disruption areas (12 months).
3) Propose 2 bold initiatives to leapfrog.
Keep it visionary but grounded.
""")
        try:
            response = groq_chat.invoke(groq_prompt)
            llm_text = _extract_llm_text(response)
        except Exception as e:
            llm_text = f"⚠️ LLM insight generation failed: {e}"

        summary = f"🌍 Internal Top Features: {', '.join(feature_ideas)}\n\nExternal Trends:\n{external_trends}"
        structured = f"### 📌 AI Trend Insights\n{llm_text}"

    elif scenario == "Feature Health":
        # Simple health rollup: top features by dynamic_score
        health = df[['product','feature','region','usage','support_tickets','sentiment','dynamic_score']].copy()
        health = health.sort_values("dynamic_score", ascending=False).head(5)
        summary = f"🩺 Showing top 5 features by overall dynamic_score."
        table = health

        groq_prompt = with_context(f"""
Feature health snapshot (top 5 by dynamic_score):
{table.to_dict(orient='records')}

Tasks:
1) Summarize overall health in 3 bullets.
2) Call out 1–2 concerns and 1–2 bright spots.
3) Suggest next diagnostic cuts to run.
""")
        try:
            response = groq_chat.invoke(groq_prompt)
            llm_text = _extract_llm_text(response)
        except Exception as e:
            llm_text = f"⚠️ LLM insight generation failed: {e}"

        structured = f"### 📌 AI Feature Health\n{llm_text}"

    else:
        summary = "🤔 Scenario not recognized."
        structured = ""

    return summary, table, extra_outputs, structured, figures

# ---------------- Streamlit Chat UI ----------------
st.title("Autonomous BI Agent with Groq AI")
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 18px;
        padding: 10px 15px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    .user-bubble {
        background-color: #0078D4;
        color: white;
        margin-left: auto;
    }
    .agent-bubble {
        background-color: #F3F2F1;
        color: black;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

user_input = st.chat_input("Ask about risks, opportunities, feature health, edge cases, or trends...")

if user_input:
    # Save user message
    st.session_state.history.append(("user", user_input))

    # Classify scenario (with continuation guard)
    new_scenario = classify_scenario(
        user_input,
        last_scenario=st.session_state.get("current_scenario")
    )

    # Detect explicit refresh intent
    wants_refresh = any(k in user_input.lower() for k in ["refresh", "recompute", "rerun", "start over", "full analysis"])

    # If scenario changed → announce and generate fresh analysis
    if new_scenario != st.session_state.get("current_scenario"):
        st.session_state.current_scenario = new_scenario
        if new_scenario != "Unknown":
            st.session_state.history.append(("agent", f"🔄 New topic detected → switching to **{new_scenario}**"))

    # Build recent context (last 6 turns)
    recent_context = "\n".join([
        f"{speaker}: {msg}" for speaker, msg in st.session_state.history[-6:]
        if speaker in ["user", "agent"]
    ])

    # Decide between full scenario analysis vs follow-up
    last_structured_answer = st.session_state.last_structured_by_scenario.get(new_scenario, "")

    is_followup_like = any(p in user_input.lower() for p in [
        "what do you mean", "what is", "why", "how", "can you explain", "clarify",
        "drill", "deeper", "more details", "example", "such as", "expand", "elaborate",
        "complete", "continue", "carry on", "go on", "next", "more"
    ]) or user_input.strip().endswith("?")

    if new_scenario and new_scenario != "Unknown":
        if (st.session_state.get("current_scenario") == new_scenario) and last_structured_answer and is_followup_like and not wants_refresh:
            # ---- Follow-up Q&A mode (DON'T regenerate the big block) ----
            reply = continue_conversation(
                user_text=user_input,
                last_structured_answer=last_structured_answer,
                scenario=new_scenario,
                context=recent_context
            )
            st.session_state.history.append(("agent", reply))
        else:
            # ---- Full scenario (first time in scenario, or explicit refresh) ----
            summary, table, extra_outputs, structured, figures = summarize_and_tabulate(
                new_scenario, df, context=recent_context
            )

            # Save the structured answer for this scenario to ground follow-ups
            st.session_state.last_structured_by_scenario[new_scenario] = structured

            st.session_state.history.append(("agent", f"**Scenario:** {new_scenario}\n\n{summary}\n\n{structured}"))

            if not table.empty:
                st.session_state.history.append(("agent_table", table))
            if figures:
                st.session_state.history.append(("agent_figures", figures))
    else:
        st.session_state.history.append(("agent", "🤔 I’m not sure which scenario to explore. Try rephrasing."))

# ---------------- Display Chat ----------------
for i, (speaker, message) in enumerate(st.session_state.history):
    if speaker == "user":
        st.markdown(f"<div class='stChatMessage user-bubble'>{message}</div>", unsafe_allow_html=True)
    elif speaker == "agent":
        st.markdown(f"<div class='stChatMessage agent-bubble'>{message}</div>", unsafe_allow_html=True)
    elif speaker == "agent_table":
        st.table(message)
    elif speaker == "agent_figures":
        for j, fig in enumerate(message):
            st.pyplot(fig, clear_figure=False, use_container_width=True)
