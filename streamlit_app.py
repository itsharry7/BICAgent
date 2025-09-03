import streamlit as st
import pandas as pd

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

# Agent logic
def agent_response(prompt, df):
    prompt = prompt.lower()
    insights = []
    scenario = None

    # Intent detection
    if "risk" in prompt or "compliance" in prompt or "issue" in prompt:
        scenario = "Risk Synthesis"
    elif "opportunity" in prompt or "investment" in prompt or "growth" in prompt:
        scenario = "Opportunity Discovery"
    elif "feature health" in prompt or "adoption" in prompt or "sentiment" in prompt:
        scenario = "Feature Health"
    elif "conflict" in prompt or "edge case" in prompt or "contradict" in prompt:
        scenario = "Edge Case"
    elif "trend" in prompt or "bold" in prompt or "creative" in prompt:
        scenario = "Stretch Scenario"

    if scenario == "Risk Synthesis":
        risk_df = df[(df['anomaly_flag'] == 1) | ((df['support_tickets'] > 10) & (df['sentiment'] < 0.5))]
        for _, row in risk_df.iterrows():
            insights.append(f"Risk: {row['product']} - {row['feature']} in {row['region']} ({row['team']}, {row['role']}) shows anomaly or high support demand with low sentiment.")
    elif scenario == "Opportunity Discovery":
        opp_df = df[(df['usage'] > 120) & (df['sentiment'] > 0.8) & (df['support_tickets'] < 3)]
        for _, row in opp_df.iterrows():
            insights.append(f"Opportunity: {row['product']} - {row['feature']} in {row['region']} ({row['team']}, {row['role']}) is loved by users and has low friction.")
    elif scenario == "Feature Health":
        health_df = df[(df['sentiment'] < 0.4) & (df['support_tickets'] > 8)]
        for _, row in health_df.iterrows():
            insights.append(f"Health Alert: {row['product']} - {row['feature']} in {row['region']} ({row['team']}, {row['role']}) has poor sentiment and high support.")
    elif scenario == "Edge Case":
        edge_df = df[(df['usage'] > 100) & (df['sentiment'] < 0.5)]
        for _, row in edge_df.iterrows():
            insights.append(f"Edge Case: {row['product']} - {row['feature']} in {row['region']} ({row['team']}, {row['role']}) is heavily used but poorly rated.")
    elif scenario == "Stretch Scenario":
        stretch_df = df[(df['usage'] > 110) & (df['support_tickets'] > 8) & (df['sentiment'] > 0.7)]
        for _, row in stretch_df.iterrows():
            insights.append(f"Stretch: {row['product']} - {row['feature']} in {row['region']} ({row['team']}, {row['role']}) is popular but support load is rising—opportunity for automation or new feature.")
    else:
        insights.append("I'm not sure what scenario you want to explore. Try asking about risks, opportunities, feature health, edge cases, or trends.")

    return scenario, insights

# Streamlit chat UI
st.title("Autonomous BI Agent (Conversational Prototype)")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask me about risks, opportunities, feature health, edge cases, or trends...")

if user_input:
    scenario, insights = agent_response(user_input, df)
    st.session_state.history.append(("user", user_input))
    if scenario:
        st.session_state.history.append(("agent", f"Scenario detected: **{scenario}**. Here are the top insights:"))
        for insight in insights[:5]:
            st.session_state.history.append(("agent", insight))
        st.session_state.history.append(("agent", "Would you like me to visualize these insights? (yes/no)"))
        st.session_state.last_scenario = scenario
    else:
        st.session_state.history.append(("agent", insights[0]))

# Display chat history
for speaker, message in st.session_state.history:
    if speaker == "user":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Agent:** {message}")

# Visualization on user request
if st.session_state.history and st.session_state.history[-1][1].endswith("visualize these insights? (yes/no)"):
    vis_input = st.text_input("Type 'yes' to see a visualization, or 'no' to continue.", key="vis_input")
    if vis_input and vis_input.lower().startswith("y"):
        import matplotlib.pyplot as plt

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
