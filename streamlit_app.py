import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Autonomous BI Agent (Conversational Prototype)", layout="wide")

# ----------------------------
# Data Loader
# ----------------------------
@st.cache_data
def load_data():
    """
    Load default data from CSV if present; otherwise, create a synthetic dataset
    with the required columns so the app always runs.
    """
    try:
        df = pd.read_csv("synthetic_enterprise_data.csv")
    except FileNotFoundError:
        # Create a small synthetic dataset as a fallback
        np.random.seed(42)
        n = 300
        products = np.random.choice(["Alpha", "Beta", "Gamma"], size=n)
        features = np.random.choice(["Search", "Share", "Sync", "Export"], size=n)
        regions = np.random.choice(["NA", "EMEA", "APAC", "LATAM"], size=n)
        teams = np.random.choice(["Core", "Growth", "Infra", "Support"], size=n)
        roles = np.random.choice(["IC", "Manager", "Director"], size=n)
        usage = np.random.randint(50, 200, size=n)
        support_tickets = np.random.poisson(lam=5, size=n)
        sentiment = np.clip(np.random.normal(loc=0.7, scale=0.2, size=n), 0, 1)
        anomaly_flag = np.random.choice([0, 1], size=n, p=[0.85, 0.15])

        df = pd.DataFrame({
            "product": products,
            "feature": features,
            "region": regions,
            "team": teams,
            "role": roles,
            "usage": usage,
            "support_tickets": support_tickets,
            "sentiment": sentiment,
            "anomaly_flag": anomaly_flag,
        })

    # Ensure basic types/ranges
    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].clip(0, 1)
    return df


# ----------------------------
# Core Analytics
# ----------------------------
def summarize_and_tabulate(scenario: str, df: pd.DataFrame):
    """
    Given a scenario and dataframe, return summary markdown, a table (pd.DataFrame),
    and optional extra outputs for 'Risk Synthesis'.
    """
    summary = ""
    table = pd.DataFrame()
    extra_outputs = {}

    # ---- Scenario: Risk Synthesis
    if scenario == "Risk Synthesis":
        # Simulate external metrics
        df2 = df.copy()
        np.random.seed(42)
        df2["External Adoption"] = df2["usage"] + np.random.normal(loc=-10, scale=15, size=len(df2))
        df2["External Reliability"] = 1 - (df2["support_tickets"] / (df2["usage"] + 1)) + np.random.normal(loc=0, scale=0.05, size=len(df2))
        df2["External Engagement"] = df2["sentiment"] + np.random.normal(loc=-0.1, scale=0.1, size=len(df2))

        # Clip to valid ranges
        df2["External Adoption"] = df2["External Adoption"].clip(lower=0)
        df2["External Reliability"] = df2["External Reliability"].clip(0, 1)
        df2["External Engagement"] = df2["External Engagement"].clip(0, 1)

        # Filter for risk
        risk_df = df2[(df2["anomaly_flag"] == 1) | ((df2["support_tickets"] > 10) & (df2["sentiment"] < 0.5))]

        summary = (
            "Several products and features across regions show anomalies or high support demand with low sentiment, "
            "indicating urgent risks that require attention. Below is a comparison of internal and simulated external metrics."
        )

        # Summary Table
        cols = [
            "product", "feature", "region", "team", "role",
            "usage", "External Adoption", "support_tickets",
            "External Reliability", "sentiment", "External Engagement"
        ]
        # Ensure all needed columns exist
        missing = [c for c in cols if c not in risk_df.columns]
        if missing:
            return (
                f"Missing required columns for Risk Synthesis: {missing}",
                pd.DataFrame(),
                {}
            )

        table = risk_df[cols].copy()
        table.rename(
            columns={
                "usage": "Internal Adoption",
                "support_tickets": "Internal Reliability (Tickets)",
                "sentiment": "Internal Engagement",
            },
            inplace=True,
        )

        # Divergence Analysis
        divergence = []
        for _, row in table.iterrows():
            issues = []
            if row["Internal Adoption"] > row["External Adoption"]:
                issues.append("Higher internal adoption")
            if (row["Internal Reliability (Tickets)"] > 10) and (row["External Reliability"] > 0.8):
                issues.append("Internal reliability issues not reflected externally")
            if row["Internal Engagement"] < row["External Engagement"]:
                issues.append("Lower internal engagement")
            divergence.append(", ".join(issues) if issues else "No significant divergence")
        table["Divergence"] = divergence

        # Reliability & Adoption Insights
        reliability_issues = []
        for _, row in risk_df.iterrows():
            if row["support_tickets"] > 10:
                reliability_issues.append({
                    "Product": row["product"],
                    "Feature": row["feature"],
                    "Region": row["region"],
                    "Insight": "High support ticket volume in internal usage",
                    "Confidence": "High",
                })

        # Prioritized Recommendations
        recommendations = []
        for issue in reliability_issues:
            recommendations.append({
                "Product": issue["Product"],
                "Feature": issue["Feature"],
                "Region": issue["Region"],
                "Recommendation": "Improve reliability through bug fixes and support automation",
                "Confidence": "High",
            })

        # Actionable Steps
        actions = []
        for issue in reliability_issues:
            actions.append({
                "Product": issue["Product"],
                "Feature": issue["Feature"],
                "Region": issue["Region"],
                "Action": "Review support logs, update documentation, and prioritize engineering fixes",
                "Confidence": "High",
            })

        extra_outputs = {
            "Reliability & Adoption Insights": reliability_issues,
            "Prioritized Recommendations": recommendations,
            "Actionable Steps": actions,
        }

    # ---- Scenario: Opportunity Discovery
    elif scenario == "Opportunity Discovery":
        filtered = df[(df["usage"] > 120) & (df["sentiment"] > 0.8) & (df["support_tickets"] < 3)]
        summary = (
            "Some features are highly used and loved by users, with minimal support issues—"
            "potential opportunities for deeper investment or expansion."
        )
        table = filtered.head(5)[["product", "feature", "region", "team", "role"]].copy()
        table["Insight"] = "High engagement and satisfaction, low friction"

    # ---- Scenario: Feature Health
    elif scenario == "Feature Health":
        filtered = df[(df["sentiment"] < 0.4) & (df["support_tickets"] > 8)]
