import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_chat import message as st_message

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

# File uploader
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
        df['External Adoption'] = df['usage'] + np.random.normal(loc=-10, scale=15, size=len(df))
        df['External Reliability'] = 1 - (df['support_tickets'] / (df['usage'] + 1)) + np.random.normal(loc=0, scale=0.05, size=len(df))
        df['External Engagement'] = df['sentiment'] + np.random.normal(loc=-0.1, scale=0.1, size=len(df))
        df['External Reliability'] = df['External Reliability'].clip(0, 1)
        df['External Engagement'] = df['External Engagement'].clip(0, 1)

        risk_df = df[(df['anomaly_flag'] == 1) | ((df['support_tickets'] > 10) & (df['sentiment'] < 0.5))]

        summary = (
            f"⚠️ {len(risk_df)} risky features across {risk_df['region'].nunique()} regions detected. "
            "Here are the top 5 by support tickets:"
        )

        table = (
            risk_df[['product', 'feature', 'region', 'support_tickets', 'sentiment']]
            .sort_values(by="support_tickets", ascending=False)
            .head(5)
            .rename(columns={
                'support_tickets': 'High Ticket Volume',
                'sentiment': 'Low Sentiment'
            })
        )

        # ----------- Charts -----------
        # Scatter: Tickets vs Sentiment
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=risk_df, x="support_tickets", y="sentiment", hue="region", ax=ax1, s=100)
        ax1.set_title("Support Tickets vs Sentiment (Risk Features)")
        ax1.set_xlabel("Support Tickets")
        ax1.set_ylabel("Sentiment Score")
        figures.append(fig1)

        # Heatmap: Features vs Regions
        heatmap_data = risk_df.pivot_table(index="feature", columns="region", values="support_tickets",
                                           aggfunc="sum", fill_value=0)
        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.heatmap(heatmap_data, cmap="Reds", annot=True, fmt="d", ax=ax2)
        ax2.set_title("Risk Feature Heatmap (Tickets by Region)")
        figures.append(fig2)

        # ----------- Structured Narrative -----------
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
- Features like **Auto Insights** in LATAM show high internal usage but poor external reliability.  
- **Copilot Chat** in Dynamics 365 diverges strongly (internal satisfaction high, external reports lower).  

**Reliability & Adoption Insights:**  
- Top reliability issues flagged in internal “Microsoft running on Microsoft” include:  
  1. Latency in Predictive Alerts (Azure AI, Europe).  
  2. Workflow Automation stability issues (Power Platform, Asia).  
  3. Support escalations in Auto Insights (Dynamics 365, LATAM).  

**Prioritized Recommendations:**  
1. Stabilize Auto Insights in LATAM (highest combined risk).  
2. Improve documentation & error handling for Predictive Alerts.  
3. Align Copilot Chat sentiment signals with external feedback mechanisms.  

**Actionable Steps:**  
- Engineering: Patch Auto Insights failure path; run chaos tests.  
- PMM: Publish clear reliability roadmap before Ignite launch.  
- Support: Create LATAM escalation channel for Workflow Automation.  
- Data: Add instrumentation to capture feature-level drop-offs.  

**Confidence & Traceability:**  
- Confidence scores: Moderate (data anomalies present).  
- Traceability: Derived from support tickets + sentiment + anomaly flags in telemetry.  
- Full lineage available in `synthetic_enterprise_data.csv` uploaded dataset.  
        """

        extra_outputs = {
            "Risk Summary": {
                "Total Risk Features": len(risk_df),
                "Regions Impacted": risk_df['region'].nunique(),
                "Avg Sentiment (at risk)": round(risk_df['sentiment'].mean(), 2),
                "Avg Support Tickets": int(risk_df['support_tickets'].mean())
            }
        }

    # ---------------- Other Scenarios ----------------
    elif scenario == "Opportunity Discovery":
        filtered = df[(df['usage'] > 120) & (df['sentiment'] > 0.8) & (df['support_tickets'] < 3)]
        summary = "🚀 Some features are highly used and loved by users, with minimal support issues — potential opportunities for deeper investment or expansion."

    elif scenario == "Feature Health":
        filtered = df[(df['sentiment'] < 0.4) & (df['support_tickets'] > 8)]
        summary = "💡 Certain features are experiencing poor sentiment and high support demand, indicating possible health issues that need investigation."

    elif scenario == "Edge Case":
        filtered = df[(df['usage'] > 100) & (df['sentiment'] < 0.5)]
        if filtered.empty:
            summary = "⚖️ Edge Case Analysis\n\nNo strong edge case patterns detected. Data may be sparse — continue monitoring.\n"
        else:
            group_cols = ["product", "feature"]
            if "stage" in df.columns:
                group_cols.append("stage")

            grouped = filtered.groupby(group_cols).agg(
                avg_usage=("usage","mean"),
                avg_sentiment=("sentiment","mean"),
                regions=("region", lambda x: ", ".join(sorted(set(x)))),
                count=("feature","size")
            ).reset_index()

            tentative_insights = []
            for _, row in grouped.iterrows():
                if "stage" in row and str(row.get("stage","")).lower() == "beta":
                    confidence = "Medium" if row["avg_sentiment"] < 0.4 else "High"
                else:
                    confidence = "Low (no stage info – assuming incomplete beta signals)"
                tentative_insights.append(
                    f"- **{row['feature']}** in {row['product']} (regions: {row['regions']}, samples: {row['count']}) "
                    f"→ Avg Usage: {row['avg_usage']:.0f}, Avg Sentiment: {row['avg_sentiment']:.2f}. ⚠️ Confidence: {confidence}"
                )

            structured = (
                "⚖️ Edge Case Analysis\n\n### Tentative Insights (confidence flagged)\n" +
                "\n".join(tentative_insights) +
                "\n\n### Data Limitations\n- Sparse signals may not fully represent all users.\n"
                "- Ambiguity: usage may be driven by forced adoption or lack of alternatives.\n"
                "- Regional variations could skew interpretation.\n"
                "\n### Recommendations\n1. Collect qualitative feedback from beta testers.\n"
                "2. Add instrumentation for drop-off, error rates, and friction points.\n"
                "3. Validate with small user surveys to confirm whether low sentiment reflects true dissatisfaction.\n"
            )

    elif scenario == "Stretch Scenario":
        candidates = df[(df['usage'] > 110) & (df['sentiment'] > 0.7)].sort_values("usage", ascending=False).head(3)
        feature_ideas = candidates['feature'].unique().tolist()
        search_results = search("Azure competitors AWS GCP disruptive cloud features developer forum trends 2025")
        external_trends = []
        if "results" in search_results:
            for r in search_results["results"][:5]:
                external_trends.append(f"- {r['title']}: {r['snippet']}")

        summary = "🌍 This request goes beyond internal telemetry and requires external research.\n\n"
        summary += "### Potential Disruptive Feature Directions (Internal)\n"
        for feat in feature_ideas:
            summary += f"- **{feat}** → Strong adoption & sentiment; candidate for next-gen innovation.\n"

        if external_trends:
            summary += "\n### External Trends & Signals (forums, competitors, reports)\n"
            summary += "\n".join(external_trends)

        summary += "\n\n### Bold Go-To-Market Plan\n1. **Research & Validation** – developer surveys, GitHub/forum scanning.\n"
        summary += "2. **Pilot** – private preview with hackathon winners & early adopters.\n"
        summary += "3. **Public Launch** – Azure Marketplace integration + dev-first campaigns.\n"
        summary += "4. **Scale** – enterprise co-sell, bundled pricing, open-source contributions.\n\n"
        summary += "### Potential Risks\n- Execution: Stability may lag vision.\n- Adoption: Complex UX/tooling slows migration.\n- Competition: AWS/GCP may fast-follow.\n"

    else:
        summary = "🤔 I'm not sure what scenario you want. Try risks, opportunities, feature health, edge cases, or trends."

    return summary, table, extra_outputs, structured, figures

# ---------------- Streamlit Chat UI ----------------
st.title("Autonomous BI Agent (Conversational Prototype)")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask me about risks, opportunities, feature health, edge cases, or trends...")

if user_input:
    prompt = user_input.lower()
    scenario = None

    if any(word in prompt for word in ["predict","disrupt","leapfrog","go-to-market","future","breakthrough","moonshot","next big","differentiator","market plan","strategy","trend","bold","creative"]):
        scenario = "Stretch Scenario"
    elif any(word in prompt for word in ["risk","compliance","issue"]):
        scenario = "Risk Synthesis"
    elif any(word in prompt for word in ["opportunity","investment","growth"]):
        scenario = "Opportunity Discovery"
    elif any(word in prompt for word in ["feature health","adoption","sentiment"]):
        scenario = "Feature Health"
    elif any(word in prompt for word in ["conflict","edge case","contradict","sparse","ambiguous","beta","explore","unknown","uncertain","tentative"]):
        scenario = "Edge Case"
    if scenario is None and any(word in prompt for word in ["insight","surface"]):
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
        st.session_state.history.append(("agent", "I'm not sure what scenario you want to explore."))

# ---------------- Display Chat ----------------
for i, (speaker, message) in enumerate(st.session_state.history):
    if speaker == "user":
        st_message(message, is_user=True, key=f"user_{i}")
    elif speaker == "agent":
        st_message(message, key=f"agent_{i}")
    elif speaker == "agent_table":
        st.table(message)  # st.table does not need a key
    elif speaker == "agent_figures":
        for j, fig in enumerate(message):
            st.pyplot(fig, key=f"fig_{i}_{j}")
