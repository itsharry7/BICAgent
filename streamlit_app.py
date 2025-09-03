def summarize_and_tabulate(scenario, df):
    summary = ""
    table = pd.DataFrame()
    if scenario == "Risk Synthesis":
        filtered = df[(df['anomaly_flag'] == 1) | ((df['support_tickets'] > 10) & (df['sentiment'] < 0.5))]
        summary = (
            "Several products and features across regions show anomalies or high support demand with low sentiment, "
            "indicating urgent risks that require attention."
        )
        table = filtered.head(5)[['product', 'feature', 'region', 'team', 'role']]
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
    return summary, table
     table['Insight'] = "Anomaly or high support demand, low sentiment"
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
    return summary, table

# Streamlit chat UI
st.title("Autonomous BI Agent (Conversational Prototype)")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask me about risks, opportunities, feature health, edge cases, or trends...")

if user_input:
    # Improved intent detection for edge/ambiguous/exploratory prompts
    prompt = user_input.lower()
    scenario = None
    if any(word in prompt for word in ["risk", "compliance", "issue"]):
        scenario = "Risk Synthesis"
    elif any(word in prompt for word in ["opportunity", "investment", "growth"]):
        scenario = "Opportunity Discovery"
    elif any(word in prompt for word in ["feature health", "adoption", "sentiment"]):
        scenario = "Feature Health"
    elif any(word in prompt for word in [
        "conflict", "edge case", "contradict", "sparse", "ambiguous", "beta", "explore", "unknown", "uncertain", "tentative"
    ]):
        scenario = "Edge Case"
    elif any(word in prompt for word in ["trend", "bold", "creative"]):
        scenario = "Stretch Scenario"
    # Fallback for exploratory/insight prompts
    if scenario is None and any(word in prompt for word in ["insight", "surface"]):
        scenario = "Edge Case"

    st.session_state.history.append(("user", user_input))
    if scenario:
        summary, table = summarize_and_tabulate(scenario, df)
        st.session_state.history.append(("agent", f"**Scenario detected:** {scenario}\n\n{summary}"))
        st.session_state.history.append(("agent_table", table))
        st.session_state.history.append(("agent", "Would you like me to visualize these insights? (yes/no)"))
        st.session_state.last_scenario = scenario
    else:
        st.session_state.history.append(("agent", "I'm not sure what scenario you want to explore. Try asking about risks, opportunities, feature health, edge cases, or trends."))

# Display chat history
for speaker, message in st.session_state.history:
    if speaker == "user":
        st.markdown(f"**You:** {message}")
    elif speaker == "agent":
        st.markdown(f"**Agent:**\n{message}")
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
        st.session_state.history.append(("agent", "Here’s a visualization of the insights by region."))            "indicating urgent risks that require attention."
        )
        table = filtered.head(5)[['product', 'feature', 'region', 'team', 'role']]
        table['Insight'] = "Anomaly or high support demand, low sentiment"
    elif scenario == "Opportunity Discovery":
        filtered = df[(df['usage'] > 120) & (df['sentiment'] > 0.8) & (df['support_tickets'] < 3)]
