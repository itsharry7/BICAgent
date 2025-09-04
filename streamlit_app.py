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
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ---------------- Load Groq API ----------------
load_dotenv()

groq_chat = ChatGroq(
    groq_api_key="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",
    model_name="llama-3.1-8b-instant",
    temperature=0.9,
    max_tokens=1000
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

# ---------------- Session State Initialization ----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = None

# ---------------- Autonomous Introduction on First Run ----------------
if not st.session_state.history:  # only on very first load
    intro_message = """
👋 Hello, I’m your **Autonomous BI Agent** powered by Groq AI.  
Here’s what I can do for you:  

### 🔎 Capabilities
- **Risk Synthesis** → Detect reliability issues, anomalies, adoption blockers.  
- **Opportunity Discovery** → Identify high-adoption features and growth levers.  
- **Edge Case Analysis** → Surface unusual usage or sentiment patterns.  
- **Stretch Scenarios** → Explore bold, forward-looking disruptive ideas.  
- **Feature Health** → Track adoption, reliability, and customer sentiment trends.  

### 💡 Sample Prompts
- *“Show me any risky features in our dataset.”*  
- *“What new opportunities should we double down on?”*  
- *“Are there any edge cases hidden in the data?”*  
- *“Predict a disruptive feature we could launch in Azure.”*  
- *“Give me a health report on Copilot features.”*  

---

✨ What can I do for you today?
"""
    st.session_state.history.append(("agent", intro_message))

if "user_df" not in st.session_state:
    st.session_state.user_df = None

uploaded_file = st.file_uploader("Upload your own enterprise data (CSV)", type="csv")
if uploaded_file:
    st.session_state.user_df = pd.read_csv(uploaded_file)
    st.success("Custom data uploaded! Agent will use this data.")

df = st.session_state.user_df if st.session_state.user_df is not None else load_data()

if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = None

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

def auto_detect_scenario(df):
    """
    Check the dataset and suggest a preliminary scenario trigger.
    Priority order: Risk > Edge Case > Opportunity > Feature Health.
    Returns scenario label or None.
    """
    df = compute_dynamic_scores(df)
    df = detect_anomalies(df)

    # Risk condition
    risky = df[(df['anomaly_flag'] == 1) | ((df['support_tickets'] > 10) & (df['sentiment'] < 0.5))]
    if not risky.empty:
        return "Risk Synthesis"

    # Edge Case
    edge = df[(df['usage'] > 100) & (df['sentiment'] < 0.5)]
    if not edge.empty:
        return "Edge Case"

    # Opportunity
    opp = df[(df['usage'] > 120) & (df['sentiment'] > 0.8) & (df['support_tickets'] < 3)]
    if not opp.empty:
        return "Opportunity Discovery"

    # Feature Health fallback (if nothing else triggered but usage exists)
    if df['usage'].mean() > 0:
        return "Feature Health"

    return None

# Scenario Classifier Function
def classify_scenario(user_text: str, last_scenario: str = None) -> str:
    """
    Use Groq to classify user input into one of the known scenarios.
    Returns one of: Risk Synthesis, Opportunity Discovery, Edge Case,
    Stretch Scenario, Feature Health, Unknown

    If the user input is a continuation (e.g., "continue", "complete"),
    it will reuse the last_scenario instead of switching.
    """

    # --- Continuation Guard ---
    continuation_phrases = ["continue", "go on", "complete", "carry on", "elaborate", "finish"]
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

Supporting Data for you regarding the scenarios so that you can make classifications easily:
- Strategy Analysis & Risk Synthesis: Sample Prompt : Surface any internal usage patterns, reliability issues, or adoption blockers in our Copilot-first services. OR Recommend actionable steps to accelerate product-market fit and reliability before external launch. This may also include insights about the business.
- Opportunity Discovery: Sample Prompt: What, where and when we can launch a new product.
- Edge Case: This is when we have Data Ambiguity. Sample Prompt: “Surface any insights about new product.”
- Stretch Scenario: Scenarios where your internal knwoledge is challenged. Sample prompt: Predict a feature we could launch in Azure that would leapfrog the competition. OR Outline a bold go-to-market plan and potential risks.
- Feature Health: when some asks about feature health etc. Like updates about product or service.

Rules:
- Never invent a new label.
- If the user message is vague or ambiguous, return "Unknown".
- Do not output explanations, only the label.
- If the user asks to continue a previous response, you should return the previous scenario (handled above).

User request: "{user_text}"
"""

    try:
        response = groq_chat.invoke(classification_prompt)
        label = getattr(response, "content", str(response)).strip()

        # sanitize
        valid_labels = {
            "Risk Synthesis",
            "Opportunity Discovery",
            "Edge Case",
            "Stretch Scenario",
            "Feature Health",
            "Unknown"
        }
        if label not in valid_labels:
            return "Unknown"
        return label

    except Exception:
        return "Unknown"
        
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

# ---------------- Core Function with Groq Integration ----------------
def summarize_and_tabulate(scenario, df, context=""):
    summary, table, extra_outputs, structured, figures = "", pd.DataFrame(), {}, "", []
    df = compute_dynamic_scores(df)
    df = detect_anomalies(df)

    # Helper to add conversation context
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
        
        # Visualization
        fig_scatter, ax1 = plt.subplots()
        sns.scatterplot(data=risk_df, x="support_tickets", y="sentiment", hue="region",
                        size="dynamic_score", ax=ax1, s=100)
        ax1.set_title("Support Tickets vs Sentiment (Risk Features)")
        figures.append(fig_scatter)
        figures.append(predict_trends(risk_df))

        groq_prompt = with_context(f"""
Dataset area of concern candidates:
{table.to_dict(orient='records')}

Your tasks: 
1. Identify the 2-3 most concerning areas and WHY they matter. 
2. Highlight any surprising or hidden correlations (region, product, adoption). 
3. Predict near-term implications if ignored. 
4. Suggest concrete responses: e.g., triage workflow, customer outreach, incident flag, product fix. 
5. Output Format: 
• Summary Table: 
• Key internal usage metrics vs. external customer metrics (adoption, reliability, feature engagement) 
• Divergence analysis: Where Microsoft’s internal usage or feedback differs from external customers
• Reliability & Adoption Insights: 
• List of top reliability issues or blockers found in internal “Microsoft running on Microsoft” scenarios 
• Prioritized recommendations for engineering or go-to-market teams 
6. Actionable Steps: 
• Concrete actions to close gaps (e.g., feature improvements, documentation, support readiness) 
• Links to supporting telemetry, feedback, and escalation contacts 
7. Confidence & Traceability: 
• Confidence scores for each insight, with full data lineage and citations 

Your answer/response MUST cover the following concerns:
•	Delayed strategic responses due to slow insight discovery
•	Missed opportunities hidden in data complexity
•	Decision paralysis from conflicting or incomplete information
•	Competitive disadvantage from reactive rather than predictive intelligence

Style & Tone: 
• Executive, strategic, and actionable 
• Transparent about data sources, confidence, and rationale 
• Focused on accelerating Copilot-first product excellence and customer alignment
""")
        try:
            response = groq_chat.invoke(groq_prompt)
            llm_text = getattr(response, "content", str(response))
        except Exception as e:
            llm_text = f"⚠️ LLM insight generation failed: {e}"

        structured = f"### 📌 AI Insights\n{llm_text}"

    elif scenario == "Opportunity Discovery":
        filtered = df[(df['usage']>120)&(df['sentiment']>0.8)&(df['support_tickets']<3)]
        summary = f"🚀 {len(filtered)} high adoption features detected. Showing top 5."
        table = filtered[['product','feature','region','dynamic_score']].sort_values("dynamic_score", ascending=False).head(5)

        groq_prompt = with_context(f"""
Dataset opportunity candidates:
{table.to_dict(orient='records')}

Tasks:
1. Identify the top growth levers and why.
2. Predict scaling implications.
3. Recommend 2-3 specific bets (campaigns, partnerships, features).

Your answer/response MUST cover the following concerns:
•	Delayed strategic responses due to slow insight discovery
•	Missed opportunities hidden in data complexity
•	Decision paralysis from conflicting or incomplete information
•	Competitive disadvantage from reactive rather than predictive intelligence
""")
        try:
            response = groq_chat.invoke(groq_prompt)
            llm_text = getattr(response, "content", str(response))
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
1. Explain what makes these patterns unusual.
2. Hypothesize plausible causes.
3. Suggest how to validate (experiments, interviews, deeper data cuts).
4. Recommend whether to prioritize or monitor quietly.

Your answer/response should address the following concerns, if applicable:
•	Delayed strategic responses due to slow insight discovery
•	Missed opportunities hidden in data complexity
•	Decision paralysis from conflicting or incomplete information
•	Competitive disadvantage from reactive rather than predictive intelligence
""")
            try:
                response = groq_chat.invoke(groq_prompt)
                llm_text = getattr(response, "content", str(response))
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
1. Connect internal strengths with external gaps.
2. Predict next 12 months disruptions.
3. Propose bold initiatives to leapfrog competitors.

Your answer/response should cover the following concerns, if applicable:
•	Delayed strategic responses due to slow insight discovery
•	Missed opportunities hidden in data complexity
•	Decision paralysis from conflicting or incomplete information
•	Competitive disadvantage from reactive rather than predictive intelligence
""")
        try:
            response = groq_chat.invoke(groq_prompt)
            llm_text = getattr(response, "content", str(response))
        except Exception as e:
            llm_text = f"⚠️ LLM insight generation failed: {e}"

        summary = f"🌍 Internal Top Features: {', '.join(feature_ideas)}\n\nExternal Trends:\n{external_trends}"
        structured = f"### 📌 AI Trend Insights\n{llm_text}"

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

if "history" not in st.session_state:
    st.session_state.history = []
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = None

# ---------------- Autonomous Preliminary Response ----------------
if not st.session_state.history:  # only on first run
    auto_scenario = auto_detect_scenario(df)
    if auto_scenario:
        st.session_state.current_scenario = auto_scenario
        st.session_state.history.append(
            ("agent", f"🤖 Preliminary analysis suggests looking at **{auto_scenario}** first.")
        )

        summary, table, extra_outputs, structured, figures = summarize_and_tabulate(
            auto_scenario, df, context="Autonomous scan (no user input)"
        )

        st.session_state.history.append(("agent", f"**Scenario:** {auto_scenario}\n\n{summary}\n\n{structured}"))

        if not table.empty:
            st.session_state.history.append(("agent_table", table))
        if figures:
            st.session_state.history.append(("agent_figures", figures))
user_input = st.chat_input("Ask about risks, opportunities, feature health, edge cases, or trends...")

if user_input:
    st.session_state.history.append(("user", user_input))

    # ---------------- Risk Synthesis Visual Choice ----------------
    if st.session_state.get("pending_visual_choice", False):
        intent = "none"
        try:
            classify_prompt = f"""
You are an intent classifier for a BI assistant. 
The user replied: "{user_input}"

Classify their intent into one of:
- "table"
- "graphs"
- "both"
- "none"

Only return the label.
"""
            classification = groq_chat.invoke(classify_prompt)
            intent = getattr(classification, "content", str(classification)).lower().strip()
        except Exception:
            intent = "none"

        # Show requested visuals
        if intent == "table" and "last_table" in st.session_state:
            st.session_state.history.append(("agent_table", st.session_state.last_table))
        elif intent == "graphs" and "last_figures" in st.session_state:
            st.session_state.history.append(("agent_figures", st.session_state.last_figures))
        elif intent == "both":
            if "last_table" in st.session_state:
                st.session_state.history.append(("agent_table", st.session_state.last_table))
            if "last_figures" in st.session_state:
                st.session_state.history.append(("agent_figures", st.session_state.last_figures))
        elif intent == "none":
            st.session_state.history.append(("agent", "👍 Skipping visuals as requested."))

        # Clear flag and stop further processing
        st.session_state.pending_visual_choice = False
        # ✅ Return early → do not classify new scenario
        st.stop()

    # ---------------- Normal Scenario Flow ----------------
    else:
        new_scenario = classify_scenario(
            user_input,
            last_scenario=st.session_state.get("current_scenario")
        )

        # Update scenario state
        if new_scenario != st.session_state.get("current_scenario"):
            st.session_state.current_scenario = new_scenario
            if new_scenario != "Unknown":
                st.session_state.history.append(
                    ("agent", f"🔄 New topic detected → switching to **{new_scenario}**")
                )

        # Process recognized scenarios
        if new_scenario and new_scenario != "Unknown":
            recent_context = "\n".join([
                f"{speaker}: {msg}" for speaker, msg in st.session_state.history[-5:]
                if speaker in ["user", "agent"]
            ])

            summary, table, extra_outputs, structured, figures = summarize_and_tabulate(
                new_scenario, df, context=recent_context
            )

            st.session_state.history.append(
                ("agent", f"**Scenario:** {new_scenario}\n\n{summary}\n\n{structured}")
            )

            # Store last visuals
            st.session_state.last_table = table
            st.session_state.last_figures = figures

            # ---------------- Risk Synthesis Special Prompt ----------------
            if new_scenario == "Risk Synthesis" and (not table.empty or figures):
                if not st.session_state.get("risk_prompt_sent", False):
                    followup_viz_msg = """
📊 I’ve prepared supporting visuals:
- Table of metrics
- Graphs showing patterns

Would you like me to show them?  
(Reply naturally: "show me graphs", "just the table", "both", or "skip")
"""
                    st.session_state.history.append(("agent", followup_viz_msg))
                    st.session_state.risk_prompt_sent = True
                    st.session_state.pending_visual_choice = True

            # ---------------- Normal follow-up suggestions ----------------
            else:
                followup_msg = """
🤖 Would you like me to go deeper? For example:
- 📊 Drill down into anomalies
- 🔮 Predict future trends
- 💬 Summarize user complaints
- 🚀 Suggest actions to take next

Reply with 'yes + option' (e.g., 'yes, drill down') or type your own request.
"""
                st.session_state.history.append(("agent", followup_msg))

        # Unknown scenario
        else:
            st.session_state.history.append(
                ("agent", "🤔 I’m not sure which scenario to explore. Try rephrasing.")
            )


            
#-------------------# Add follow-up suggestions after a normal answer
            
        if new_scenario and new_scenario != "Unknown":
    # ✅ Handle Risk Synthesis differently
            if new_scenario == "Risk Synthesis":
        # ✅ Only trigger once
                if not st.session_state.get("risk_prompt_sent", False):
                    st.session_state.pending_followup = True
                    if not table.empty or figures:
                        followup_viz_msg = """
        📊 I’ve prepared supporting visuals:
        - Table of metrics
        - Graphs showing patterns
        
        Would you like me to show them?  
        (You can reply naturally, e.g. "show me graphs", "just the table", "both", or "skip")
        """
                        st.session_state.history.append(("agent", followup_viz_msg))
                    st.session_state.risk_prompt_sent = True  # mark as sent
                    st.session_state.pending_followup = True
            else:
                # Normal scenarios → immediate follow-up
                followup_msg = """
        🤖 Would you like me to go deeper? For example:
        - 📊 Drill down into anomalies
        - 🔮 Predict future trends
        - 💬 Summarize user complaints
        - 🚀 Suggest actions to take next
        
        Reply with 'yes + option' (e.g., 'yes, drill down') or type your own request.
        """
                st.session_state.history.append(("agent", followup_msg))
        else:
            # Unknown scenario case
            st.session_state.history.append(
                ("agent", "🤔 I’m not sure which scenario to explore. Try rephrasing.")
            )
        # ---------------- Autonomous Follow-up Scan ----------------
        auto_scenario = auto_detect_scenario(df)
        if auto_scenario and auto_scenario != st.session_state.current_scenario:
            st.session_state.current_scenario = auto_scenario
            st.session_state.history.append(
                ("agent", f"🤖 While analyzing, I also detected a potential **{auto_scenario}** situation.")
            )

            summary, table, extra_outputs, structured, figures = summarize_and_tabulate(
                auto_scenario, df, context="Autonomous follow-up scan"
            )

            st.session_state.history.append(("agent", f"**Scenario:** {auto_scenario}\n\n{summary}\n\n{structured}"))

            if not table.empty:
                st.session_state.history.append(("agent_table", table))
            if figures:
                st.session_state.history.append(("agent_figures", figures))


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
