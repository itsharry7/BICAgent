import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_chat import message as st_message
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import io
import base64

# ---------------- OpenAI Integration ----------------
from openai import OpenAI
import os

# Set your OpenAI API key here or via environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_llm_insights(prompt: str, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a helpful enterprise insights assistant."},
                {"role":"user","content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM insight generation failed: {str(e)}"

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

def render_fig_in_chat(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"![plot](data:image/png;base64,{img_base64})"

# ---------------- Core Function with LLM ----------------
def summarize_and_tabulate(scenario, df):
    summary, table, extra_outputs, structured, figures = "", pd.DataFrame(), {}, "", []
    df = compute_dynamic_scores(df)
    df = detect_anomalies(df)

    if scenario == "Risk Synthesis":
        risk_df = df[(df['anomaly_flag'] == 1) | ((df['support_tickets'] > 10) & (df['sentiment'] < 0.5))]
        table = (risk_df.sort_values('dynamic_score', ascending=False)
                 [['product','feature','region','support_tickets','sentiment','dynamic_score']].head(5))
        
        # LLM-assisted summary
        prompt = f"Analyze the following risky features and provide a concise insight with recommendations:\n{table.to_dict(orient='records')}"
        summary = get_llm_insights(prompt)
        
        # Trend & charts
        fig_scatter, ax1 = plt.subplots()
        sns.scatterplot(data=risk_df, x="support_tickets", y="sentiment", hue="region", size="dynamic_score", ax=ax1, s=100)
        ax1.set_title("Support Tickets vs Sentiment (Risk Features)")
        figures.append(fig_scatter)

        fig_heatmap, ax2 = plt.subplots(figsize=(8,6))
        heatmap_data = risk_df.pivot_table(index="feature", columns="region", values="support_tickets", aggfunc="sum", fill_value=0)
        sns.heatmap(heatmap_data, cmap="Reds", annot=True, fmt="d", ax=ax2)
        ax2.set_title("Risk Feature Heatmap")
        figures.append(fig_heatmap)

        trend_fig = predict_trends(risk_df, metric='usage')
        figures.append(trend_fig)

        structured = f"### 📌 Top Risk Features Table\n{table.to_markdown(index=False)}"

    elif scenario == "Opportunity Discovery":
        filtered = df[(df['usage'] > 120) & (df['sentiment'] > 0.8) & (df['support_tickets'] < 3)]
        table = filtered[['feature','dynamic_score']].sort_values('dynamic_score',ascending=False).head(5)
        prompt = f"Identify opportunities and actionable insights from these high adoption features:\n{table.to_dict(orient='records')}"
        summary = get_llm_insights(prompt)
        structured = f"### 📌 Opportunity Table\n{table.to_markdown(index=False)}"

    elif scenario == "Feature Health":
        filtered = df[(df['sentiment'] < 0.4) & (df['support_tickets'] > 8)]
        table = filtered[['feature','dynamic_score']].sort_values('dynamic_score',ascending=False).head(5)
        prompt = f"Provide insights on low-sentiment features and suggest improvement actions:\n{table.to_dict(orient='records')}"
        summary = get_llm_insights(prompt)
        structured = f"### 📌 Feature Health Table\n{table.to_markdown(index=False)}"

    elif scenario == "Edge Case":
        filtered = df[(df['usage'] > 100) & (df['sentiment'] < 0.5)]
        if filtered.empty:
            summary = "✅ No unusual patterns detected."
        else:
            table = filtered[['product','feature','region','usage','sentiment']].head(5)
            prompt = f"Some features show unusual patterns. Explain what these edge patterns mean and suggest actions:\n{table.to_dict(orient='records')}"
            summary = get_llm_insights(prompt)
            structured = f"### 📌 Unusual Feature Table\n{table.to_markdown(index=False)}"
            csv = filtered.to_csv(index=False)
            st.download_button("Download Feature Insights", csv, "unusual_features.csv")
            # Optional scatter chart
            fig_scatter, ax1 = plt.subplots()
            sns.scatterplot(data=filtered, x="usage", y="sentiment", hue="region", size="support_tickets", ax=ax1, s=100)
            ax1.set_title("Usage vs Sentiment (Unusual Features)")
            figures.append(fig_scatter)

    elif scenario == "Stretch Scenario":
        candidates = df[(df['usage'] > 110) & (df['sentiment'] > 0.7)].sort_values("dynamic_score", ascending=False).head(3)
        table = candidates[['feature','dynamic_score']]
        prompt = f"Provide a forward-looking insight on these top internal features and external trends:\n{table.to_dict(orient='records')}"
        summary = get_llm_insights(prompt)
        structured = f"### 📌 Stretch Scenario Top Features\n{table.to_markdown(index=False)}"

    else:
        summary = "🤔 Unknown scenario."

    return summary, table, extra_outputs, structured, figures

# ---------------- Streamlit Chat UI ----------------
st.title("Autonomous BI Agent (LLM-Enhanced)")

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
        st.table(message)
    elif speaker == "agent_figures":
        for j, fig in enumerate(message):
            st.pyplot(fig, clear_figure=False, use_container_width=True)
