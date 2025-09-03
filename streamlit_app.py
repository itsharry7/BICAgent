import streamlit as st
import pandas as pd

# Load the complex synthetic enterprise dataset
df = pd.read_csv("synthetic_enterprise_data.csv")

# Helper function to generate insights
def generate_insights(scenario, df):
    insights = []
    if scenario == "Risk Synthesis":
        # High anomaly, high support, low sentiment
        risk_df = df[(df['anomaly_flag'] == 1) | ((df['support_tickets'] > 10) & (df['sentiment'] < 0.5))]
        for _, row in risk_df.iterrows():
            insights.append({
                "Insight": f"Risk: {row['product']} - {row['feature']} in {row['region']} ({row['team']}, {row['role']}) shows anomaly or high support demand with low sentiment.",
                "Confidence": "High" if row['anomaly_flag'] else "Medium",
                "Explanation": "Detected anomaly or user pain point requiring urgent attention."
            })
    elif scenario == "Opportunity Discovery":
        # High usage, high sentiment, low support
        opp_df = df[(df['usage'] > 120) & (df['sentiment'] > 0.8) & (df['support_tickets'] < 3)]
        for _, row in opp_df.iterrows():
            insights.append({
                "Insight": f"Opportunity: {row['product']} - {row['feature']} in {row['region']} ({row['team']}, {row['role']}) is loved by users and has low friction.",
                "Confidence": "High",
                "Explanation": "High engagement and satisfaction—consider deeper investment or expansion."
            })
    elif scenario == "Feature Health":
        # Declining sentiment, rising support
        health_df = df[(df['sentiment'] < 0.4) & (df['support_tickets'] > 8)]
        for _, row in health_df.iterrows():
            insights.append({
                "Insight": f"Health Alert: {row['product']} - {row['feature']} in {row['region']} ({row['team']}, {row['role']}) has poor sentiment and high support.",
                "Confidence": "High",
                "Explanation": "Negative user experience and high support demand indicate feature health issues."
            })
    elif scenario == "Edge Case":
        # Conflicting signals: high usage, low sentiment
        edge_df = df[(df['usage'] > 100) & (df['sentiment'] < 0.5)]
        for _, row in edge_df.iterrows():
            insights.append({
                "Insight": f"Edge Case: {row['product']} - {row['feature']} in {row['region']} ({row['team']}, {row['role']}) is heavily used but poorly rated.",
                "Confidence": "Medium",
                "Explanation": "Possible forced adoption or hidden friction—requires deeper investigation."
            })
    elif scenario == "Stretch Scenario":
        # Subtle pattern: high usage, high support, good sentiment
        stretch_df = df[(df['usage'] > 110) & (df['support_tickets'] > 8) & (df['sentiment'] > 0.7)]
        for _, row in stretch_df.iterrows():
            insights.append({
                "Insight": f"Stretch: {row['product']} - {row['feature']} in {row['region']} ({row['team']}, {row['role']}) is popular but support load is rising—opportunity for automation or new feature.",
                "Confidence": "Medium",
                "Explanation": "Emerging pattern—consider bold investment or innovation."
            })
    return insights

# Streamlit UI
st.title("Autonomous BI Agent Prototype")
st.markdown("Select a scenario to generate actionable insights from complex enterprise data.")

scenario = st.selectbox(
    "Choose a scenario",
    [
        "Risk Synthesis",
        "Opportunity Discovery",
        "Feature Health",
        "Edge Case",
        "Stretch Scenario"
    ]
)

if st.button("Generate Insights"):
    insights = generate_insights(scenario, df)
    if insights:
        for i, insight in enumerate(insights, 1):
            st.subheader(f"Insight {i}")
            st.write(f"**Insight:** {insight['Insight']}")
            st.write(f"**Confidence:** {insight['Confidence']}")
            st.write(f"**Explanation:** {insight['Explanation']}")
    else:
        st.warning("No insights found for the selected scenario.")

st.markdown("---")
st.caption("Upload your own enterprise data to test the agent's capabilities.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    st.write(user_df.head())
    st.success("File uploaded! Rerun the scenario to use your data.")
