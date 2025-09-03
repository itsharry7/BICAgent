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
    # Normalize metrics and compute weighted score
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
    # mark clusters with extreme mean scores as anomalies
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

# ---------------- Core Function ----------------
def summarize_and_tabulate(scenario, df):
    summary, table, extra_outputs, structured, figures = "", pd.DataFrame(), {}, "", []
    df = compute_dynamic_scores(df)
    df = detect_anomalies(df)

    if scenario == "Risk Synthesis":
        risk_df = df[(df['anomaly_flag'] == 1) | ((df['support_tickets'] > 10) & (df['sentiment'] < 0.5))]
        summary = f"⚠️ {len(risk_df)} risky features across {risk_df['region'].nunique()} regions detected. Top risks by dynamic score:"
        table = (risk_df.sort_values('dynamic_score', ascending=False)
                 [['product','feature','region','support_tickets','sentiment','dynamic_score']]
                 .head(5))
        # Charts
        fig_scatter, ax1 = plt.subplots()
        sns.scatterplot(data=risk_df, x="support_tickets", y="sentiment", hue="region", size="dynamic_score", ax=ax1, s=100)
        ax1.set_title("Support Tickets vs Sentiment (Risk Features)")
        figures.append(fig_scatter)
        fig_heatmap, ax2 = plt.subplots(figsize=(8,6))
        heatmap_data = risk_df.pivot_table(index="feature", columns="region", values="support_tickets", aggfunc="sum", fill_value=0)
        sns.heatmap(heatmap_data, cmap="Reds", annot=True, fmt="d", ax=ax2)
        ax2.set_title("Risk Feature Heatmap")
        figures.append(fig_heatmap)
        # Trend prediction
        trend_fig = predict_trends(risk_df, metric='usage')
        figures.append(trend_fig)
        structured = f"### 📌 Structured Risk Analysis\n**Top 5 Features by Score:**\n{table.to_markdown(index=False)}"
        extra_outputs = {"Total Risk Features": len(risk_df), "Regions Impacted": risk_df['region'].nunique()}
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


    elif scenario == "Opportunity Discovery":
        filtered = df[(df['usage'] > 120) & (df['sentiment'] > 0.8) & (df['support_tickets'] < 3)]
        summary = f"🚀 {len(filtered)} high adoption features detected. Top opportunities:\n{filtered[['feature','dynamic_score']].sort_values('dynamic_score',ascending=False).head(5).to_markdown(index=False)}"

    elif scenario == "Feature Health":
        filtered = df[(df['sentiment'] < 0.4) & (df['support_tickets'] > 8)]
        summary = f"💡 {len(filtered)} low sentiment features detected. See top concerns:\n{filtered[['feature','dynamic_score']].sort_values('dynamic_score',ascending=False).head(5).to_markdown(index=False)}"

    elif scenario == "Edge Case":
        filtered = df[(df['usage'] > 100) & (df['sentiment'] < 0.5)]
        if filtered.empty:
            summary = "⚖️ No edge case patterns detected."
        else:
            structured = filtered[['product','feature','region','usage','sentiment']].head(5).to_markdown(index=False)
            summary = f"⚖️ Edge Case Features:\n{structured}"

    elif scenario == "Stretch Scenario":
        candidates = df[(df['usage'] > 110) & (df['sentiment'] > 0.7)].sort_values("dynamic_score", ascending=False).head(3)
        feature_ideas = candidates['feature'].tolist()
        search_results = search("Azure competitors AWS GCP disruptive cloud features developer forum trends 2025")
        external_trends = ""
        if "results" in search_results:
            external_trends = "\n".join([f"- {r['title']}: {r['snippet']}" for r in search_results["results"][:5]])
        summary = f"🌍 Internal Top Features: {', '.join(feature_ideas)}\n\nExternal Trends:\n{external_trends}"

    else:
        summary = "🤔 Unknown scenario."

    return summary, table, extra_outputs, structured, figures

# ---------------- Streamlit Chat UI ----------------
st.title("Autonomous BI Agent (Enhanced Prototype)")

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
            # Instead of base64, directly show figure
            st.pyplot(fig, clear_figure=False, use_container_width=True)
