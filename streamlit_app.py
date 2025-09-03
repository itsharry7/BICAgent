import streamlit as st
import pandas as pd
import numpy as np

# Load default data
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_enterprise_data.csv")

if "user_df" not in st.session_state:
    st.session_state.user_df = None

# File uploader for user data
uploaded_file = st.file_uploader("Upload your own enterprise data (CSV)", type="csv")
if uploaded_file:
    st.session_state.user_df = pd.read_csv(uploaded_file)
    st.success("Custom data uploaded! Agent will use this data.")

df = st.session_state.user_df if st.session_state.user_df is not None else load_data()


# ---------------- FIXED FUNCTION ----------------
def summarize_and_tabulate(scenario, df):
    summary = ""
    table = pd.DataFrame()
    extra_outputs = {}

    if scenario == "Risk Synthesis":
        # Simulate external metrics
        df = df.copy()
        np.random.seed(42)
        df['External Adoption'] = df['usage'] + np.random.normal(loc=-10, scale=15, size=len(df))
        df['External Reliability'] = 1 - (df['support_tickets'] / (df['usage'] + 1)) + np.random.normal(loc=0, scale=0.05, size=len(df))
        df['External Engagement'] = df['sentiment'] + np.random.normal(loc=-0.1, scale=0.1, size=len(df))
        df['External Reliability'] = df['External Reliability'].clip(0, 1)
        df['External Engagement'] = df['External Engagement'].clip(0, 1)

        # Filter for risk
        risk_df = df[(df['anomaly_flag'] == 1) | ((df['support_tickets'] > 10) & (df['sentiment'] < 0.5))]

        summary = (
            "Several products and features across regions show anomalies or high support demand with low sentiment, "
            "indicating urgent risks that require attention. Below is a comparison of internal and simulated external metrics."
        )

        # Summary Table
        table = risk_df[['product', 'feature', 'region', 'team', 'role', 'usage',
                         'External Adoption', 'support_tickets',
                         'External Reliability', 'sentiment',
                         'External Engagement']].copy()
        table.rename(columns={
            'usage': 'Internal Adoption',
            'support_tickets': 'Internal Reliability (Tickets)',
            'sentiment': 'Internal Engagement'
        }, inplace=True)

        # Divergence Analysis
        divergence = []
        for _, row in table.iterrows():
            issues = []
            if row['Internal Adoption'] > row['External Adoption']:
                issues.append("Higher internal adoption")
            if row['Internal Reliability (Tickets)'] > 10 and row['External Reliability'] > 0.8:
                issues.append("Internal reliability issues not reflected externally")
            if row['Internal Engagement'] < row['External Engagement']:
                issues.append("Lower internal engagement")
            if issues:
                divergence.append(", ".join(issues))
            else:
                divergence.append("No significant divergence")
        table['Divergence'] = divergence

        # Reliability & Adoption Insights
        reliability_issues = []
        for _, row in risk_df.iterrows():
            if row['support_tickets'] > 10:
                reliability_issues.append({
                    "Product": row['product'],
                    "Feature": row['feature'],
                    "Region": row['region'],
                    "Insight": "High support ticket volume in internal usage",
                    "Confidence": "High"
                })

        # Prioritized Recommendations
        recommendations = [
            {
                "Product": issue["Product"],
                "Feature": issue["Feature"],
                "Region": issue["Region"],
                "Recommendation": "Improve reliability through bug fixes and support automation",
                "Confidence": "High"
            }
            for issue in reliability_issues
        ]

        # Actionable Steps
        actions = [
            {
                "Product": issue["Product"],
                "Feature": issue["Feature"],
                "Region": issue["Region"],
                "Action": "Review support logs, update documentation, and prioritize engineering fixes",
                "Confidence": "High"
            }
            for issue in reliability_issues
        ]

        extra_outputs = {
            "Reliability & Adoption Insights": reliability_issues,
            "Prioritized Recommendations": recommendations,
            "Actionable Steps": actions
        }

    elif scenario == "Opportunity Discovery":
        filtered = df[(df['usage'] > 120) & (df['sentiment'] > 0.8) & (df['support_tickets'] < 3)]
        summary = (
            "Some features are highly used and loved by users, with minimal support issues—potential opportunities for deeper investment or expansion."
        )
        table = filtered.head(5)[['product', 'feature', 'region', 'team', 'role']]
        table['Insight'] = "High engagement and satisfaction, low friction"

    elif scenario == "Feature Health":
        filtered = df[(df['sentiment'] < 0.4) & (df['support_tickets'] > 8)]
        summary = (
            "Certain features are experiencing poor sentiment and high support demand, indicating possible health issues that need investigation."
        )
        table = filtered.head(5)[['product', 'feature', 'region', 'team', 'role']]
        table['Insight'] = "Poor sentiment and high support demand"

    elif scenario == "Edge Case":
        filtered = df[(df['usage'] > 100) & (df['sentiment'] < 0.5)]
        summary = (
            "Some features are heavily used but poorly rated, suggesting possible forced adoption, hidden friction, or ambiguous/sparse data."
        )
        table = filtered.head(5)[['product', 'feature', 'region', 'team', 'role']]
        table['Insight'] = "High usage but low sentiment—possible forced adoption or ambiguity"

    elif scenario == "Stretch Scenario":
        filtered = df[(df['usage'] > 110) & (df['support_tickets'] > 8) & (df['sentiment'] > 0.7)]
        summary = (
            "Emerging patterns show popular features with rising support load—opportunities for automation or bold innovation."
        )
        table = filtered.head(5)[['product', 'feature', 'region', 'team', 'role']]
        table['Insight'] = "Popular feature, rising support—opportunity for innovation"

    else:
        summary = "I'm not sure what scenario you want to explore. Try asking about risks, opportunities, feature health, edge cases, or trends."

    return summary, table, extra_outputs
# ------------------------------------------------


# Streamlit chat UI
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
    elif any(word in prompt for word in [
        "conflict", "edge case", "contradict", "sparse", "ambiguous", "beta", "explore", "unknown", "uncertain", "tentative"
    ]):
        scenario = "Edge Case"
    if scenario is None and any(word in prompt for word in ["insight", "surface"]):
        scenario = "Edge Case"

    st.session_state.history.append(("user", user_input))
    if scenario:
        summary, table, extra_outputs = summarize_and_tabulate(scenario, df)
        st.session_state.history.append(("agent", f"**Scenario detected:** {scenario}\n\n{summary}"))
        st.session_state.history.append(("agent_table", table))
        if scenario == "Risk Synthesis" and extra_outputs:
            st.session_state.history.append(("agent", "### Reliability & Adoption Insights"))
            for insight in extra_outputs["Reliability & Adoption Insights"]:
                st.session_state.history.append(("agent", f"- {insight['Product']} | {insight['Feature']} | {insight['Region']}: {insight['Insight']} (Confidence: {insight['Confidence']})"))
            st.session_state.history.append(("agent", "### Prioritized Recommendations"))
            for rec in extra_outputs["Prioritized Recommendations"]:
                st.session_state.history.append(("agent", f"- {rec['Product']} | {rec['Feature']} | {rec['Region']}: {rec['Recommendation']} (Confidence: {rec['Confidence']})"))
            st.session_state.history.append(("agent", "### Actionable Steps"))
            for act in extra_outputs["Actionable Steps"]:
                st.session_state.history.append(("agent", f"- {act['Product']} | {act['Feature']} | {act['Region']}: {act['Action']} (Confidence: {act['Confidence']})"))
        st.session_state.history.append(("agent", "Would you like me to visualize these insights? (yes/no)"))
        st.session_state.last_scenario = scenario
    else:
        st.session_state.history.append(("agent", "I'm not sure what scenario you want to explore. Try asking about risks, opportunities, feature health, edge cases, or trends."))

# Display chat history
for speaker, message in st.session_state.history:
    if speaker == "user":
        st.markdown(f"**You:** {message}")
    elif speaker == "agent":
        st.markdown(f"{message}")
    elif speaker == "agent_table":
        st.table(message)

# Visualization on user request
if st.session_state.history and st.session_state.history[-1][1].endswith("visualize these insights? (yes/no)"):
    vis_input = st.text_input("Type 'yes' to see a visualization, or 'no' to continue.", key="vis_input")
    if vis_input and vis_input.lower().startswith("y"):
        scenario = st.session_state.last_scenario
        if scenario == "Risk Synthesis":
            vis_df = df[(df['anomaly_flag'] == 1) | ((df['support_tickets'] > 10) & (df['sentiment'] < 0.5))]
        elif scenario == "Opportunity Discovery":
            vis_df = df[(df['usage'] > 120) & (df['sentiment'] > 0.8) & (df['support_tickets'] < 3)]
        elif scenario == "Feature Health":
            vis_df = df[(df['sentiment'] < 0.4) & (df['support_tickets'] > 8)]
        elif scenario == "Edge Case":
            vis_df = df[(df['usage'] > 100) & (df['sentiment'] < 0.5)]
        elif scenario == "Stretch Scenario":
            vis_df = df[(df['usage'] > 110) & (df['support_tickets'] > 8) & (df['sentiment'] > 0.7)]
        else:
            vis_df = df

        st.subheader("Insights by Region")
        region_counts = vis_df['region'].value_counts()
        st.bar_chart(region_counts)
        st.session_state.history.append(("agent", "Here’s a visualization of the insights by region."))
