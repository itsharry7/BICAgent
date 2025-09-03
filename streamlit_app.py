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
    
# Scenario Classifier Function
def classify_scenario(user_input: str) -> str:
    """
    Uses Groq LLM to classify user query into one of the known scenarios.
    """
    classify_prompt = f"""
Classify the following query into exactly one of these categories:
["Risk Synthesis", "Opportunity Discovery", "Edge Case", "Stretch Scenario", "Feature Health"]

Query: "{user_input}"

Respond with ONLY the scenario name, nothing else.
"""
    try:
        response = groq_chat.invoke(classify_prompt)
        scenario = getattr(response, "content", str(response)).strip()

        # Clean up common issues (extra quotes, punctuation, explanations)
        scenario = scenario.replace('"', '').replace("'", "").strip()

        if scenario not in ["Risk Synthesis", "Opportunity Discovery", "Edge Case", "Stretch Scenario", "Feature Health"]:
            scenario = "Unknown"
        return scenario
    except Exception as e:
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
def summarize_and_tabulate(scenario, df):
    summary, table, extra_outputs, structured, figures = "", pd.DataFrame(), {}, "", []
    df = compute_dynamic_scores(df)
    df = detect_anomalies(df)

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

        # 🔥 Upgraded Groq Prompt
        groq_prompt = f"""
You are analyzing enterprise product telemetry + customer signals.

Dataset risk candidates:
{table.to_dict(orient='records')}

Your tasks:
1. Identify the 2-3 most concerning risks and WHY they matter.
2. Highlight any surprising or hidden correlations (region, product, adoption).
3. Predict near-term implications if ignored.
4. Suggest concrete responses: e.g., triage workflow, customer outreach, incident flag, product fix.
5. Output Format:
•	Summary Table: 
•	Key internal usage metrics vs. external customer metrics (adoption, reliability, feature engagement)
•	Divergence analysis: Where Microsoft’s internal usage or feedback differs from external customers
•	Reliability & Adoption Insights: 
•	List of top reliability issues or blockers found in internal “Microsoft running on Microsoft” scenarios
•	Prioritized recommendations for engineering or go-to-market teams
6. Actionable Steps: 
•	Concrete actions to close gaps (e.g., feature improvements, documentation, support readiness)
•	Links to supporting telemetry, feedback, and escalation contacts
7. Confidence & Traceability: 
•	Confidence scores for each insight, with full data lineage and citations
Style & Tone:
•	Executive, strategic, and actionable
•	Transparent about data sources, confidence, and rationale
•	Focused on accelerating Copilot-first product excellence and customer alignment

Output a structured summary stakeholders can act on immediately.
"""
        try:
            response = groq_chat.invoke(groq_prompt)
            llm_text = getattr(response, "content", str(response))
        except Exception as e:
            llm_text = f"⚠️ LLM insight generation failed: {e}"

        structured = f"### 📌 AI Risk Insights\n{llm_text}"

    elif scenario == "Opportunity Discovery":
        filtered = df[(df['usage']>120)&(df['sentiment']>0.8)&(df['support_tickets']<3)]
        summary = f"🚀 {len(filtered)} high adoption features detected. Showing top 5."
        table = filtered[['product','feature','region','dynamic_score']].sort_values("dynamic_score", ascending=False).head(5)

        groq_prompt = f"""
You are a product growth strategist reviewing adoption + sentiment metrics.

Dataset opportunity candidates:
{table.to_dict(orient='records')}

Your tasks:
1. Identify the top growth levers and explain WHY they stand out.
2. Predict scaling implications if invested in now.
3. Recommend 2-3 specific bets (campaigns, partnerships, feature doubling).

Keep it concise but action-oriented.
"""
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

            groq_prompt = f"""
You are scanning for unusual or conflicting signals.

Edge case features:
{table.to_dict(orient='records')}

Your tasks:
1. Explain what makes these patterns unusual.
2. Hypothesize plausible causes (data issue, adoption shift, misaligned expectations).
3. Suggest how to validate (experiments, customer interviews, deeper data cuts).
4. Recommend whether these should be prioritized or monitored quietly.
"""
            try:
                response = groq_chat.invoke(groq_prompt)
                llm_text = getattr(response, "content", str(response))
            except Exception as e:
                llm_text = f"⚠️ LLM insight generation failed: {e}"

            structured = f"### 📌 AI Edge Case Insights\n{llm_text}"

            # Add visualization
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

        groq_prompt = f"""
You are synthesizing internal telemetry + external market signals.

Internal candidates: {feature_ideas}
External competitor trends:
{external_trends}

Your tasks:
1. Connect internal strengths with external gaps.
2. Predict where disruption is most likely in the next 12 months.
3. Propose bold initiatives Microsoft could take to leapfrog competitors.

Keep tone visionary but backed by evidence.
"""
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

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask about risks, opportunities, feature health, edge cases, or trends...")

if user_input:
    # Save user message
    st.session_state.history.append(("user", user_input))

    prompt = user_input.lower()
    scenario = classify_scenario(user_input)

    if scenario and scenario != "Unknown":
        summary, table, extra_outputs, structured, figures = summarize_and_tabulate(scenario, df)
        st.session_state.history.append(("agent", f"**Scenario detected:** {scenario}\n\n{summary}\n\n{structured}"))
        if not table.empty:
            st.session_state.history.append(("agent_table", table))
        if figures:
            st.session_state.history.append(("agent_figures", figures))
    else:
        st.session_state.history.append(("agent", "🤔 I’m not sure which scenario to explore. Try rephrasing."))

                  
# ---------------- Display Chat ----------------
for i, (speaker, message) in enumerate(st.session_state.history):
    if speaker == "user":
        st_message(message, is_user=True, key=f"user_{i}")
    elif speaker == "agent":
        st_message(message, key=f"agent_{i}")
    elif speaker == "agent_table":
        st.table(message)
    elif speaker == "agent_figures":
        for j, fig in enumerate(message):
            st.pyplot(fig, clear_figure=False, use_container_width=True)
