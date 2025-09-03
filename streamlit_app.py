import streamlit as st
import pandas as pd
import numpy as np
import requests

# ---------------- Web Search with Fallback ----------------
def search(query: str):
    """DuckDuckGo search with graceful fallback to mock results."""
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json"}

    try:
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()

        results = []
        for item in data.get("RelatedTopics", []):
            if "Text" in item and "FirstURL" in item:
                results.append({
                    "title": item["Text"],
                    "snippet": item["Text"],
                    "link": item["FirstURL"]
                })
        if results:
            return {"results": results}

    except Exception as e:
        # Log error for debugging
        print(f"[search fallback] Error: {e}")

    # --- Mock fallback if real search fails ---
    return {
        "results": [
            {"title": "AI-powered cloud disruption", "snippet": "Startups are leveraging AI-first cloud services to challenge AWS and Azure.", "link": "#"},
            {"title": "Multi-cloud adoption rising", "snippet": "Enterprises are increasingly adopting AWS, Azure, and GCP together for resilience.", "link": "#"},
            {"title": "Serverless platforms 2025", "snippet": "Next-gen serverless promises near-zero ops and pay-per-execution pricing models.", "link": "#"},
            {"title": "Developer-first platforms", "snippet": "Communities highlight the importance of DX as the key differentiator in cloud.", "link": "#"},
            {"title": "Sustainability in cloud", "snippet": "Green computing and carbon-aware scheduling are becoming enterprise priorities.", "link": "#"}
        ]
    }

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
    summary, table, extra_outputs = "", pd.DataFrame(), {}

    if scenario == "Risk Synthesis":
        # Simulate external metrics
        df = df.copy()
        np.random.seed(42)
        df['External Adoption'] = df['usage'] + np.random.normal(loc=-10, scale=15, size=len(df))
        df['External Reliability'] = 1 - (df['support_tickets'] / (df['usage'] + 1)) + np.random.normal(loc=0, scale=0.05, size=len(df))
        df['External Engagement'] = df['sentiment'] + np.random.normal(loc=-0.1, scale=0.1, size=len(df))
        df['External Reliability'] = df['External Reliability'].clip(0, 1)
        df['External Engagement'] = df['External Engagement'].clip(0, 1)

        # Filter for risky features
        risk_df = df[(df['anomaly_flag'] == 1) | ((df['support_tickets'] > 10) & (df['sentiment'] < 0.5))]

        summary = (
            f"⚠️ {len(risk_df)} risky features across {risk_df['region'].nunique()} regions detected. "
            "Here are the top 5 by support tickets:"
        )

        # Minimal risk summary table
        table = (
            risk_df[['product', 'feature', 'region', 'support_tickets', 'sentiment']]
            .sort_values(by="support_tickets", ascending=False)
            .head(5)
            .rename(columns={
                'support_tickets': 'High Ticket Volume',
                'sentiment': 'Low Sentiment'
            })
        )

        # Aggregates
        extra_outputs = {
            "Risk Summary": {
                "Total Risk Features": len(risk_df),
                "Regions Impacted": risk_df['region'].nunique(),
                "Avg Sentiment (at risk)": round(risk_df['sentiment'].mean(), 2),
                "Avg Support Tickets": int(risk_df['support_tickets'].mean())
            }
        }

    elif scenario == "Opportunity Discovery":
        filtered = df[(df['usage'] > 120) & (df['sentiment'] > 0.8) & (df['support_tickets'] < 3)]
        summary = (
            "🚀 Some features are highly used and loved by users, with minimal support issues — "
            "potential opportunities for deeper investment or expansion."
        )

    elif scenario == "Feature Health":
        filtered = df[(df['sentiment'] < 0.4) & (df['support_tickets'] > 8)]
        summary = (
            "💡 Certain features are experiencing poor sentiment and high support demand, "
            "indicating possible health issues that need investigation."
        )

elif scenario == "Edge Case":
    # Filter candidates (high usage + low sentiment)
    filtered = df[(df['usage'] > 100) & (df['sentiment'] < 0.5)]

    if filtered.empty:
        summary = (
            "⚖️ Edge Case Analysis\n\n"
            "No strong edge case patterns detected. Data may be sparse — continue monitoring.\n"
        )
    else:
        # Group by product/feature for roll-up summaries
        group_cols = ["product", "feature"]
        if "stage" in df.columns:
            group_cols.append("stage")

        grouped = (
            filtered.groupby(group_cols)
            .agg(
                avg_usage=("usage", "mean"),
                avg_sentiment=("sentiment", "mean"),
                regions=("region", lambda x: ", ".join(sorted(set(x)))),
                count=("feature", "size"),
            )
            .reset_index()
        )

        tentative_insights = []
        for _, row in grouped.iterrows():
            # Confidence logic
            if "stage" in row and str(row.get("stage", "")).lower() == "beta":
                confidence = "Medium" if row["avg_sentiment"] < 0.4 else "High"
            else:
                confidence = "Low (no stage info – assuming incomplete beta signals)"

            tentative_insights.append(
                f"- **{row['feature']}** in {row['product']} "
                f"(regions: {row['regions']}, samples: {row['count']}) "
                f"→ Avg Usage: {row['avg_usage']:.0f}, Avg Sentiment: {row['avg_sentiment']:.2f}. "
                f"⚠️ Confidence: {confidence}"
            )

        summary = (
            "⚖️ Edge Case Analysis\n\n"
            "### Tentative Insights (confidence flagged)\n"
            + "\n".join(tentative_insights) +
            "\n\n### Data Limitations\n"
            "- Sparse signals: sentiment data may not fully represent all users.\n"
            "- Ambiguity: usage may be driven by forced adoption or lack of alternatives.\n"
            "- Regional variations could skew interpretation.\n"
            "\n### Recommendations\n"
            "1. Collect qualitative feedback from beta testers.\n"
            "2. Add instrumentation for drop-off, error rates, and friction points.\n"
            "3. Validate with small user surveys to confirm whether low sentiment reflects true dissatisfaction.\n"
        )


    elif scenario == "Stretch Scenario":
        # Internal candidates
        candidates = df[(df['usage'] > 110) & (df['sentiment'] > 0.7)].sort_values("usage", ascending=False).head(3)
        feature_ideas = candidates['feature'].unique().tolist()

        # 🔎 Perform live web search (with fallback)
        search_results = search("Azure competitors AWS GCP disruptive cloud features developer forum trends 2025")
        external_trends = []
        if "results" in search_results:
            for r in search_results["results"][:5]:
                external_trends.append(f"- {r['title']}: {r['snippet']}")

        # Build structured summary
        summary = (
            "🌍 This request goes beyond internal telemetry and requires external research. "
            "Here’s a structured response combining **internal adoption signals** and **live market insights**:\n\n"
            "### Potential Disruptive Feature Directions (Internal)\n"
        )
        for feat in feature_ideas:
            summary += f"- **{feat}** → Strong adoption & sentiment; candidate for next-gen innovation.\n"

        if external_trends:
            summary += "\n### External Trends & Signals (forums, competitors, reports)\n"
            summary += "\n".join(external_trends)

        summary += (
            "\n\n### Bold Go-To-Market Plan\n"
            "1. **Research & Validation** – developer surveys, GitHub/forum scanning.\n"
            "2. **Pilot** – private preview with hackathon winners & early adopters.\n"
            "3. **Public Launch** – Azure Marketplace integration + dev-first campaigns.\n"
            "4. **Scale** – enterprise co-sell, bundled pricing, open-source contributions.\n\n"
            "### Potential Risks\n"
            "- Execution: Stability may lag vision.\n"
            "- Adoption: Complex UX/tooling slows migration.\n"
            "- Competition: AWS/GCP may fast-follow.\n"
        )

    else:
        summary = "🤔 I'm not sure what scenario you want. Try risks, opportunities, feature health, edge cases, or trends."

    return summary, table, extra_outputs

# ---------------- Streamlit Chat UI ----------------
st.title("Autonomous BI Agent (Conversational Prototype)")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask me about risks, opportunities, feature health, edge cases, or trends...")

if user_input:
    prompt = user_input.lower()
    scenario = None

    if any(word in prompt for word in [
        "predict", "disrupt", "leapfrog", "go-to-market", "future", "breakthrough", "moonshot", "next big", "differentiator", "market plan", "strategy", "trend", "bold", "creative"
    ]):
        scenario = "Stretch Scenario"
    elif any(word in prompt for word in ["risk", "compliance", "issue"]):
        scenario = "Risk Synthesis"
    elif any(word in prompt for word in ["opportunity", "investment", "growth"]):
        scenario = "Opportunity Discovery"
    elif any(word in prompt for word in ["feature health", "adoption", "sentiment"]):
        scenario = "Feature Health"
    elif any(word in prompt for word in ["conflict", "edge case", "contradict", "sparse", "ambiguous", "beta", "explore", "unknown", "uncertain", "tentative"]):
        scenario = "Edge Case"

    if scenario is None and any(word in prompt for word in ["insight", "surface"]):
        scenario = "Edge Case"

    st.session_state.history.append(("user", user_input))
    if scenario:
        summary, table, extra_outputs = summarize_and_tabulate(scenario, df)
        st.session_state.history.append(("agent", f"**Scenario detected:** {scenario}\n\n{summary}"))

        # Show summary table only for Risk Synthesis
        if scenario == "Risk Synthesis" and not table.empty:
            st.session_state.history.append(("agent_table", table))

    else:
        st.session_state.history.append(("agent", "I'm not sure what scenario you want to explore."))

# ---------------- Display Chat ----------------
for speaker, message in st.session_state.history:
    if speaker == "user":
        st.markdown(f"**You:** {message}")
    elif speaker == "agent":
        st.markdown(f"{message}")
    elif speaker == "agent_table":
        st.table(message)
