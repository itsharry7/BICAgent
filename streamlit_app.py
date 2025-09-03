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

# ---------------- Groq Integration ----------------
from groq import GroqClient

# Initialize Groq client
groq_client = GroqClient(api_key=st.secrets["GROQ_API_KEY"])

def generate_groq_insights(scenario, df_sample):
    """
    Generate AI-powered insights using Groq.
    """
    # Limit rows for context to avoid token overflow
    df_sample = df_sample.head(50)
    data_str = df_sample.to_dict(orient='records')

    prompt = f"""
You are an expert enterprise analyst. Analyze the following dataset based on the scenario: {scenario}.
Dataset sample (up to 50 rows): {data_str}

Provide:
1. Concise summary of insights.
2. Top 5 features or risks.
3. Recommendations for actions.
4. Structured output in markdown.
"""
    response = groq_client.generate(prompt=prompt, max_output_tokens=600)
    return response.output_text

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
    weights = np.array([0.4, 0.3, 0.2, 0.1])
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

# ---------------- Core Function ----------------
def summarize_and_tabulate(scenario, df):
    df = compute_dynamic_scores(df)
    df = detect_anomalies(df)
    
    summary, table, extra_outputs, structured, figures = "", pd.DataFrame(), {}, "", []

    # ---------------- Scenario Analysis ----------------
    if scenario == "Risk Synthesis":
        risk_df = df[(df['anomaly_flag'] == 1) | ((df['support_tickets'] > 10) & (df['sentiment'] < 0.5))]
        table = (risk_df.sort_values('dynamic_score', ascending=False)
                 [['product','feature','region','support_tickets','sentiment','dynamic_score']].head(5))
        fig_scatter, ax1 = plt.subplots()
        sns.scatterplot(data=risk_df, x="support_tickets", y="sentiment", hue="region", size="dynamic_score", ax=ax1, s=100)
        ax1.set_title("Support Tickets vs Sentiment (Risk Features)")
        figures.append(fig_scatter)
        trend_fig = predict_trends(risk_df, metric='usage')
        figures.append(trend_fig)
        
        # Groq AI Insights
        summary = generate_groq_insights("Risk Synthesis", risk_df)
        structured = f"### 🤖 Groq Insights (Risk Synthesis)\n{summary}"

    elif scenario == "Opportunity Discovery":
        filtered = df[(df['usage'] > 120) & (df['sentiment'] > 0.8) & (df['support_tickets'] < 3)]
        table = filtered[['product','feature','dynamic_score']].sort_values('dynamic_score', ascending=False).head(5)
        summary = generate_groq_insights("Opportunity Discovery", filtered)
        structured = f"### 🤖 Groq Insights (Opportunity Discovery)\n{summary}"

    elif scenario == "Feature Health":
        filtered = df[(df['sentiment'] < 0.4) & (df['support_tickets'] > 8)]
        table = filtered[['product','feature','dynamic_score']].sort_values('dynamic_score', ascending=False).head(5)
        summary = generate_groq_insights("Feature Health", filtered)
        structured = f"### 🤖 Groq Insights (Feature Health)\n{summary}"

    elif scenario == "Edge Case":
        filtered = df[(df['usage'] > 100) & (df['sentiment'] < 0.5)]
        if filtered.empty:
            summary = "✅ No unusual patterns detected."
        else:
            table = filtered[['product','feature','region','usage','sentiment']].head(5)
            fig_scatter, ax1 = plt.subplots()
            sns.scatterplot(data=filtered, x="usage", y="sentiment", hue="region", size="support_tickets", ax=ax1, s=100)
            ax1.set_title("Usage vs Sentiment (Unusual Features)")
            figures.append(fig_scatter)
            summary = generate_groq_insights("Edge Case", filtered)
            structured = f"### 🤖 Groq Insights (Edge Case)\n{summary}"
            csv = filtered.to_csv(index=False)
            st.download_button("Download Feature Insights", csv, "edge_case_features.csv")

    elif scenario == "Stretch Scenario":
        candidates = df[(df['usage'] > 110) & (df['sentiment'] > 0.7)].sort_values("dynamic_score", ascending=False).head(3)
        feature_ideas = candidates['feature'].tolist()
        search_results = search("Azure competitors AWS GCP disruptive cloud features developer forum trends 2025")
        external_trends = "\n".join([f"- {r['title']}: {r['snippet']}" for r in search_results.get("results", [])[:5]]) \
                          if search_results.get("results") else "- Unable to fetch external trends."
        summary = generate_groq_insights("Stretch Scenario", candidates)
        structured = f"### 🤖 Groq Insights (Stretch Scenario)\n**Internal Candidates:** {', '.join(feature_ideas)}\n\n{summary}\n\n**External Trends:**\n{external_trends}"

    else:
        summary = "🤔 Unknown scenario."
    
    return summary, table, extra_outputs, structured, figures

# ---------------- Streamlit Chat UI ----------------
st.title("Autonomous BI Agent (Enhanced Prototype with Groq)")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask about risks, opportunities, feature health, edge cases, or trends...")

if user_input:
    prompt = user_input.lower()
    scenario = None
    if any(word in prompt for word in ["predict","disrupt","go-to-market","trend","bold"]):
        scenario = "Stretch Scenario"
    elif any(word in prompt for word in ["risk","compliance","issue"]):
        scenario = "Risk Synthesis"
    elif any(word in prompt for word in ["opportunity","investment","growth"]):
        scenario = "Opportunity Discovery"
    elif any(word in prompt for word in ["feature health","adoption","sentiment"]):
        scenario = "Feature Health"
    elif any(word in prompt for word in ["conflict","edge case","contradict","beta"]):
        scenario = "Edge Case"

    st.session_state.history.append(("user", user_input))

    if scenario:
        summary, table, extra_outputs, structured, figures = summarize_and_tabulate(scenario, df)
        st.session_state.history.append(("agent", f"**Scenario detected:** {scenario}\n\n{structured}"))
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
        st.table(message)
    elif speaker == "agent_figures":
        for j, fig in enumerate(message):
            st.pyplot(fig, clear_figure=False, use_container_width=True)
