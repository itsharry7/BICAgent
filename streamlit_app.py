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
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ---------------- Load Groq API ----------------
load_dotenv()
groq_chat = ChatGroq(
    groq_api_key="YOUR_GROQ_KEY_HERE",
    model_name="llama-3.1-8b-instant",
    temperature=0.9,
    max_tokens=1000
)

# ---------------- Utilities ----------------
def load_data():
    return pd.read_csv("synthetic_enterprise_data.csv")

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

def compute_dynamic_scores(df):
    df = df.copy()
    df.fillna({'support_tickets':0,'sentiment':0.5,'usage':0,'anomaly_flag':0}, inplace=True)
    scaler = StandardScaler()
    metrics = scaler.fit_transform(df[['usage','support_tickets','sentiment','anomaly_flag']])
    weights = np.array([0.4,0.3,0.2,0.1])
    df['dynamic_score'] = metrics.dot(weights)
    return df

def predict_trends(df, metric='usage'):
    df_sorted = df.sort_values(['product','feature'])
    fig, ax = plt.subplots()
    for _, group in df_sorted.groupby(['product','feature']):
        X = np.arange(len(group)).reshape(-1,1)
        y = group[metric].values
        if len(y) < 2: continue
        model = LinearRegression().fit(X,y)
        ax.plot(X, model.predict(X), label=f"{group['feature'].iloc[0]} ({group['product'].iloc[0]})")
    ax.set_title(f"Predicted Trend for {metric}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    return fig

def render_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"![plot](data:image/png;base64,{img_base64})"

def search(query, max_results=5):
    try:
        url = "https://duckduckgo.com/html/"
        res = requests.get(url, params={"q":query}, timeout=10)
        res.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(res.text, "html.parser")
        results = []
        for a in soup.select(".result__a")[:max_results]:
            snippet_tag = a.find_parent().select_one(".result__snippet")
            results.append({
                "title": a.get_text(),
                "link": a.get("href"),
                "snippet": snippet_tag.get_text() if snippet_tag else ""
            })
        return results
    except Exception:
        return []

# ---------------- Session State ----------------
if "history" not in st.session_state: st.session_state.history = []
if "current_scenario" not in st.session_state: st.session_state.current_scenario = None
if "pending_visual_choice" not in st.session_state: st.session_state.pending_visual_choice = False
if "risk_prompt_sent" not in st.session_state: st.session_state.risk_prompt_sent = False
if "user_df" not in st.session_state: st.session_state.user_df = None

# ---------------- Data Upload ----------------
uploaded_file = st.file_uploader("Upload CSV data", type="csv")
if uploaded_file:
    st.session_state.user_df = pd.read_csv(uploaded_file)
    st.success("Custom data uploaded!")

df = st.session_state.user_df if st.session_state.user_df is not None else load_data()

# ---------------- Scenario Detection ----------------
def auto_detect_scenario(df):
    df = compute_dynamic_scores(df)
    df = detect_anomalies(df)
    # Risk
    if not df[(df['anomaly_flag']==1)|((df['support_tickets']>10)&(df['sentiment']<0.5))].empty: return "Risk Synthesis"
    # Edge Case
    if not df[(df['usage']>100)&(df['sentiment']<0.5)].empty: return "Edge Case"
    # Opportunity
    if not df[(df['usage']>120)&(df['sentiment']>0.8)&(df['support_tickets']<3)].empty: return "Opportunity Discovery"
    # Feature Health fallback
    if df['usage'].mean()>0: return "Feature Health"
    return None

# ---------------- Scenario Classifier ----------------
def classify_scenario(user_text, last_scenario=None):
    continuation = ["continue","go on","complete","carry on","elaborate","finish"]
    if any(p in user_text.lower() for p in continuation) and last_scenario:
        return last_scenario
    prompt = f"""
Classify user input into ONE of:
- Risk Synthesis
- Opportunity Discovery
- Edge Case
- Stretch Scenario
- Feature Health
- Unknown

User request: "{user_text}"
"""
    try:
        resp = groq_chat.invoke(prompt)
        label = getattr(resp,"content",str(resp)).strip()
        return label if label in ["Risk Synthesis","Opportunity Discovery","Edge Case","Stretch Scenario","Feature Health","Unknown"] else "Unknown"
    except Exception:
        return "Unknown"

# ---------------- Scenario Summarization ----------------
def summarize_scenario(scenario, df, context=""):
    df = compute_dynamic_scores(df)
    df = detect_anomalies(df)
    table = pd.DataFrame()
    figures = []
    structured = ""
    summary = ""

    if scenario=="Risk Synthesis":
        risk_df = df[(df['anomaly_flag']==1)|((df['support_tickets']>10)&(df['sentiment']<0.5))]
        summary = f"⚠️ {len(risk_df)} risky features detected across {risk_df['region'].nunique()} regions."
        table = risk_df[['product','feature','region','support_tickets','sentiment','dynamic_score']].head(5)
        fig = plt.figure()
        sns.scatterplot(data=risk_df,x="support_tickets",y="sentiment",hue="region",size="dynamic_score",s=100)
        plt.title("Support Tickets vs Sentiment (Risk Features)")
        figures.append(fig)
        figures.append(predict_trends(risk_df))
        structured = "### 📌 AI Insights (Risk Synthesis)"
    
    elif scenario=="Opportunity Discovery":
        opp_df = df[(df['usage']>120)&(df['sentiment']>0.8)&(df['support_tickets']<3)]
        summary = f"🚀 {len(opp_df)} high adoption features detected."
        table = opp_df[['product','feature','region','dynamic_score']].head(5)
        structured = "### 📌 AI Opportunity Insights"

    elif scenario=="Edge Case":
        edge_df = df[(df['usage']>100)&(df['sentiment']<0.5)]
        if edge_df.empty: summary="✅ No unusual patterns detected."
        else:
            summary = f"⚖️ {len(edge_df)} unusual feature(s) detected."
            table = edge_df[['product','feature','region','usage','sentiment']].head(5)
            fig = plt.figure()
            sns.scatterplot(data=edge_df,x="usage",y="sentiment",hue="region",size="support_tickets",s=100,alpha=0.8)
            plt.title("Usage vs Sentiment (Edge Cases)")
            figures.append(fig)
            structured = "### 📌 AI Edge Case Insights"

    elif scenario=="Stretch Scenario":
        candidates = df[(df['usage']>110)&(df['sentiment']>0.7)].sort_values("dynamic_score",ascending=False).head(3)
        summary = f"🌍 Top internal features: {', '.join(candidates['feature'].tolist())}"
        structured = "### 📌 AI Trend Insights"

    else:
        summary = "🤔 Scenario not recognized."
    
    return summary, table, structured, figures

# ---------------- Streamlit UI ----------------
st.title("Autonomous BI Agent with Groq AI")
user_input = st.chat_input("Ask about risks, opportunities, feature health, edge cases, or trends...")

# ---------------- Handle User Input ----------------
if user_input:
    st.session_state.history.append(("user", user_input))
    scenario = classify_scenario(user_input, st.session_state.current_scenario)
    st.session_state.current_scenario = scenario
    summary, table, structured, figures = summarize_scenario(scenario, df)
    st.session_state.last_table = table
    st.session_state.last_figures = figures

    st.session_state.history.append(("agent", f"**Scenario:** {scenario}\n\n{summary}\n\n{structured}"))
    if not table.empty: st.session_state.history.append(("agent_table", table))
    if figures: st.session_state.history.append(("agent_figures", figures))

# ---------------- Display Chat ----------------
for speaker, message in st.session_state.history:
    if speaker=="user":
        st.markdown(f"<div class='stChatMessage user-bubble'>{message}</div>", unsafe_allow_html=True)
    elif speaker=="agent":
        st.markdown(f"<div class='stChatMessage agent-bubble'>{message}</div>", unsafe_allow_html=True)
    elif speaker=="agent_table":
        st.table(message)
    elif speaker=="agent_figures":
        for fig in message: st.pyplot(fig, use_container_width=True)
