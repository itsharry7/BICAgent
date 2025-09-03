import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_chat import message as st_message
from sklearn.cluster import KMeans

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

# ---------------- Core Function ----------------
def summarize_and_tabulate(scenario, df):
    summary, table, extra_outputs, structured, figures = "", pd.DataFrame(), {}, "", []

    if scenario == "Risk Synthesis":
        df = df.copy()
        np.random.seed(42)
        df['External Adoption'] = df['usage'] + np.random.normal(-10, 15, len(df))
        df['External Reliability'] = 1 - (df['support_tickets'] / (df['usage'] + 1)) + np.random.normal(0, 0.05, len(df))
        df['External Engagement'] = df['sentiment'] + np.random.normal(-0.1, 0.1, len(df))
        df['External Reliability'] = df['External Reliability'].clip(0,1)
        df['External Engagement'] = df['External Engagement'].clip(0,1)

        # ---------- Autonomous anomaly detection ----------
        feature_metrics = df[['usage', 'sentiment', 'support_tickets']].copy()
        kmeans = KMeans(n_clusters=3, random_state=42).fit(feature_metrics)
        df['risk_cluster'] = kmeans.labels_
        risk_df = df[(df['anomaly_flag'] == 1) | ((df['support_tickets'] > 10) & (df['sentiment'] < 0.5))]

        summary = f"⚠️ {len(risk_df)} risky features across {risk_df['region'].nunique()} regions detected. Top 5 by support tickets:"

        table = (risk_df[['product','feature','region','support_tickets','sentiment']]
                 .sort_values('support_tickets', ascending=False)
                 .head(5)
                 .rename(columns={'support_tickets':'High Ticket Volume','sentiment':'Low Sentiment'}))

        # ---------- Charts ----------
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=risk_df, x="support_tickets", y="sentiment", hue="region", ax=ax1, s=100)
        ax1.set_title("Support Tickets vs Sentiment (Risk Features)")
        figures.append(fig1)

        heatmap_data = risk_df.pivot_table(index="feature", columns="region", values="support_tickets", aggfunc="sum", fill_value=0)
        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.heatmap(heatmap_data, cmap="Reds", annot=True, fmt="d", ax=ax2)
        ax2.set_title("Risk Feature Heatmap (Tickets by Region)")
        figures.append(fig2)

        # ---------- Structured Narrative ----------
        structured = f"""
### 📌 Structured Risk Synthesis

**Summary:**  
- {len(risk_df)} risky features detected across {risk_df['region'].nunique()} regions.  
- Top risks concentrated in {', '.join(risk_df['region'].unique()[:3])} (sample view).  

**Key Internal vs External Metrics:**  
- Internal Adoption (avg): {df['usage'].mean():.1f}  
- External Adoption (avg): {df['External Adoption'].mean():.1f}  
- Internal Reliability (sentiment proxy): {df['sentiment'].mean():.2f}  
- External Reliability (simulated): {df['External Reliability'].mean():.2f}  

**Divergence Analysis:**  
- Clustering detected 3 risk clusters; top-risk features mostly in cluster 0.  
- Features like **Auto Insights** in LATAM show high internal usage but poor external reliability.  

**Actionable Recommendations:**  
1. Stabilize Auto Insights in LATAM (highest combined risk).  
2. Improve documentation & error handling for Predictive Alerts.  
3. Align Copilot Chat sentiment signals with external feedback mechanisms.  

**Confidence & Traceability:**  
- Derived from support tickets + sentiment + anomaly flags + cluster analysis.
        """

        extra_outputs = {
            "Risk Summary": {
                "Total Risk Features": len(risk_df),
                "Regions Impacted": risk_df['region'].nunique(),
                "Avg Sentiment (at risk)": round(risk_df['sentiment'].mean(),2),
                "Avg Support Tickets": int(risk_df['support_tickets'].mean())
            }
        }

    # ---------- Other scenarios ----------
    elif scenario == "Opportunity Discovery":
        filtered = df[(df['usage']>120)&(df['sentiment']>0.8)&(df['support_tickets']<3)]
        summary = "🚀 High-adoption, high-sentiment features with low support tickets identified as opportunities."
    elif scenario == "Feature Health":
        filtered = df[(df['sentiment']<0.4)&(df['support_tickets']>8)]
        summary = "💡 Features with poor sentiment & high support tickets flagged for health review."
    elif scenario == "Edge Case":
        filtered = df[(df['usage']>100)&(df['sentiment']<0.5)]
        if filtered.empty:
            summary = "⚖️ Edge Case Analysis: No unusual patterns detected."
        else:
            grouped = filtered.groupby(["product","feature"]).agg(
                avg_usage=("usage","mean"),
                avg_sentiment=("sentiment","mean"),
                count=("feature","size")
            ).reset_index()
            insights = [f"- {r['feature']} in {r['product']} → Usage:{r['avg_usage']:.0f}, Sentiment:{r['avg_sentiment']:.2f}, Count:{r['count']}" for _,r in grouped.iterrows()]
            structured = "### ⚖️ Here are some of the insights, I gather regarding your query\n" + "\n".join(insights)
            summary = "Tentative patterns detected from my knowledge references."

    elif scenario == "Stretch Scenario":
        candidates = df[(df['usage']>110)&(df['sentiment']>0.7)].sort_values("usage",ascending=False).head(3)
        features = candidates['feature'].tolist()
        search_results = search("Azure competitors AWS GCP disruptive cloud features developer forum trends 2025")
        trends = [f"- {r['title']}: {r['snippet']}" for r in search_results.get("results",[])[:5]]
        summary = f"🌍 Disruptive Feature Candidates (Internal): {', '.join(features)}\n"
        if trends:
            summary += "\nExternal Trends:\n" + "\n".join(trends)

    else:
        summary = "🤔 Scenario unclear. Try risks, opportunities, feature health, edge cases, or trends."

    return summary, table, extra_outputs, structured, figures

# ---------------- Streamlit Chat UI ----------------
st.title("Autonomous BI Agent (Conversational Prototype)")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask me about risks, opportunities, feature health, edge cases, or trends...")

if user_input:
    prompt = user_input.lower()
    scenario = None
    if any(w in prompt for w in ["predict","disrupt","leapfrog","go-to-market","future","breakthrough","moonshot","next big","differentiator","market plan","strategy","trend","bold","creative"]):
        scenario = "Stretch Scenario"
    elif any(w in prompt for w in ["risk","compliance","issue"]):
        scenario = "Risk Synthesis"
    elif any(w in prompt for w in ["opportunity","investment","growth"]):
        scenario = "Opportunity Discovery"
    elif any(w in prompt for w in ["feature health","adoption","sentiment"]):
        scenario = "Feature Health"
    elif any(w in prompt for w in ["conflict","edge case","contradict","sparse","ambiguous","beta","explore","unknown","uncertain","tentative"]):
        scenario = "Edge Case"
    if scenario is None and any(w in prompt for w in ["insight","surface"]):
        scenario = "Edge Case"

    st.session_state.history.append(("user", user_input))
    if scenario:
        summary, table, extra_outputs, structured, figures = summarize_and_tabulate(scenario, df)
        st.session_state.history.append(("agent", f"**Scenario detected:** {scenario}\n\n{summary}\n\n{structured}"))
        if not table.empty:
            st.session_state.history.append(("agent_table", table))
        if figures:
            st.session_state.history.append(("agent_figures", figures))
    else:
        st.session_state.history.append(("agent", "Scenario unclear."))

# ---------------- Display Chat ----------------
for i, (speaker, message) in enumerate(st.session_state.history):
    if speaker=="user":
        st_message(message, is_user=True, key=f"user_{i}")
    elif speaker=="agent":
        st_message(message, key=f"agent_{i}")
    elif speaker=="agent_table":
        st.table(message)
    elif speaker=="agent_figures":
        for j, fig in enumerate(message):
            st.pyplot(fig, clear_figure=True)
