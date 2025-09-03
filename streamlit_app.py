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
groq_api_key = os.getenv("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")

groq_chat = ChatGroq(
    groq_api_key=groq_api_key,
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
        fig_scatter, ax1 = plt.subplots()
        sns.scatterplot(data=risk_df, x="support_tickets", y="sentiment", hue="region", size="dynamic_score", ax=ax1, s=100)
        ax1.set_title("Support Tickets vs Sentiment (Risk Features)")
        figures.append(fig_scatter)
        trend_fig = predict_trends(risk_df)
        figures.append(trend_fig)

        # ---------------- Groq LLM Narrative ----------------
        groq_prompt = (
            f"Analyze this enterprise dataset risk scenario. "
            f"Top 5 risk features:\n{table.to_dict(orient='records')}\n"
            f"Provide actionable recommendations for stakeholders in a concise manner."
        )
        response = groq_chat.invoke(groq_prompt)
        try:
            llm_text = json.loads(response.json())['content']
        except:
            llm_text = response

        structured = f"### 📌 Structured LLM Insights\n{llm_text}"

    elif scenario == "Opportunity Discovery":
        filtered = df[(df['usage']>120)&(df['sentiment']>0.8)&(df['support_tickets']<3)]
        summary = f"🚀 {len(filtered)} high adoption features detected. Top opportunities:\n{filtered[['feature','dynamic_score']].head(5).to_markdown(index=False)}"

        # LLM opportunity enhancement
        groq_prompt = (
            f"Here are top opportunity features:\n{filtered[['feature','dynamic_score']].to_dict(orient='records')}\n"
            f"Generate a concise growth narrative and recommendations."
        )
        response = groq_chat.invoke(groq_prompt)
        try:
            llm_text = json.loads(response.json())['content']
        except:
            llm_text = response
        structured = f"### 📌 LLM Opportunity Insights\n{llm_text}"

    elif scenario == "Stretch Scenario":
        candidates = df[(df['usage']>110)&(df['sentiment']>0.7)].sort_values("dynamic_score", ascending=False).head(3)
        feature_ideas = candidates['feature'].tolist()
        search_results = search("Azure competitors AWS GCP disruptive cloud features developer forum trends 2025")
        external_trends = "\n".join([f"- {r['title']}: {r['snippet']}" for r in search_results.get("results", [])[:5]])

        # LLM to synthesize internal + external trends
        groq_prompt = (
            f"Internal top features: {feature_ideas}\n"
            f"External trends: {external_trends}\n"
            f"Generate an executive summary highlighting opportunities and recommendations."
        )
        response = groq_chat.invoke(groq_prompt)
        try:
            llm_text = json.loads(response.json())['content']
        except:
            llm_text = response
        summary = f"🌍 Internal Top Features: {', '.join(feature_ideas)}\n\nExternal Trends:\n{external_trends}"
        structured = f"### 📌 LLM Trend Insights\n{llm_text}"

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
    prompt = user_input.lower()
    scenario = None
    if any(word in prompt for word in ["predict","disrupt","trend","bold"]):
        scenario = "Stretch Scenario"
    elif any(word in prompt for word in ["risk","compliance","issue"]):
        scenario = "Risk Synthesis"
    elif any(word in prompt for word in ["opportunity","investment","growth"]):
        scenario = "Opportunity Discovery"

    st.session_state.history.append(("user", user_input))

    if scenario:
        summary, table, extra_outputs, structured, figures = summarize_and_tabulate(scenario, df)
        st.session_state.history.append(("agent", f"**Scenario detected:** {scenario}\n\n{summary}\n\n{structured}"))
        if not table.empty:
            st.session_state.history.append(("agent_table", table))
        if figures:
            st.session_state.history.append(("agent_figures", figures))
    else:
        st.session_state.history.append(("agent", "I'm not sure which scenario to explore."))

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
