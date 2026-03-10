import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Page setup + simple styling
# -----------------------------
st.set_page_config(page_title="Client Risk Command Center", layout="wide")

st.markdown(
    """
    <style>
    .big-title{font-size:40px; font-weight:800; margin-bottom:0px;}
    .subtle{opacity:0.75; margin-top:0px;}
    .card{
        padding:14px 16px; border-radius:14px;
        border:1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">Client Risk Command Center</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">B2B Risk Scoring • Churn Prediction • Retention Actions • Responsible AI</div>', unsafe_allow_html=True)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("B2B_Client_Churn_5000.csv")

df = load_data()

# -----------------------------
# Column checks (prevents crash)
# -----------------------------
required_cols = [
    "Client_ID", "Industry", "Region",
    "Monthly_Usage_Score", "Payment_Delay_Days",
    "Contract_Length_Months", "Support_Tickets_Last30Days",
    "Monthly_Revenue_USD", "Renewal_Status"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"CSV columns missing: {missing}")
    st.write("Columns found:", list(df.columns))
    st.stop()

# Optional columns for better UI
for col, default in [("Company_Name", "NA"), ("Plan", "NA"), ("Lead_Source", "NA"), ("Account_Age_Months", 0)]:
    if col not in df.columns:
        df[col] = default

# Churn flag
df["Churned"] = df["Renewal_Status"].map({"Yes": 0, "No": 1}).fillna(0).astype(int)

# -----------------------------
# Part B: Risk scoring (NEW + different)
# Quantile-based scoring + weights (0–10) then category
# -----------------------------
# Create quantile thresholds from current dataset (dynamic + different from earlier code)
delay_q = df["Payment_Delay_Days"].quantile([0.50, 0.80]).values
usage_q = df["Monthly_Usage_Score"].quantile([0.20, 0.50]).values  # low usage -> risk
contract_q = df["Contract_Length_Months"].quantile([0.20, 0.50]).values  # short contract -> risk
tickets_q = df["Support_Tickets_Last30Days"].quantile([0.50, 0.80]).values

def risk_points(row):
    pts = 0

    # Payment delay (0–3)
    if row["Payment_Delay_Days"] >= delay_q[1]:
        pts += 3
    elif row["Payment_Delay_Days"] >= delay_q[0]:
        pts += 2
    elif row["Payment_Delay_Days"] > 0:
        pts += 1

    # Usage (0–3) (LOW usage => higher risk)
    if row["Monthly_Usage_Score"] <= usage_q[0]:
        pts += 3
    elif row["Monthly_Usage_Score"] <= usage_q[1]:
        pts += 2
    elif row["Monthly_Usage_Score"] <= df["Monthly_Usage_Score"].quantile(0.70):
        pts += 1

    # Contract length (0–2)
    if row["Contract_Length_Months"] <= contract_q[0]:
        pts += 2
    elif row["Contract_Length_Months"] <= contract_q[1]:
        pts += 1

    # Tickets (0–2)
    if row["Support_Tickets_Last30Days"] >= tickets_q[1]:
        pts += 2
    elif row["Support_Tickets_Last30Days"] >= tickets_q[0]:
        pts += 1

    return pts

df["Risk_Score"] = df.apply(risk_points, axis=1)  # 0–10

def risk_bucket(x):
    if x >= 7:
        return "High Risk"
    elif x >= 4:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk_Category"] = df["Risk_Score"].apply(risk_bucket)

# -----------------------------
# Sidebar: navigation + filters
# -----------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "📌 Segmentation", "🤖 Model Lab", "🛠 Action Center", "⚖️ Responsible AI", "📄 Data Export"],
    index=0
)

st.sidebar.divider()
st.sidebar.header("Filters")

all_regions = sorted(df["Region"].dropna().unique())
all_industries = sorted(df["Industry"].dropna().unique())
risk_levels = ["Low Risk", "Medium Risk", "High Risk"]

sel_region = st.sidebar.multiselect("Region", all_regions, default=all_regions)
sel_industry = st.sidebar.multiselect("Industry", all_industries, default=all_industries)
sel_risk = st.sidebar.multiselect("Risk Category", risk_levels, default=risk_levels)

rev_min, rev_max = float(df["Monthly_Revenue_USD"].min()), float(df["Monthly_Revenue_USD"].max())
sel_rev = st.sidebar.slider("Revenue Range (USD)", rev_min, rev_max, (rev_min, rev_max))

f = df[
    df["Region"].isin(sel_region) &
    df["Industry"].isin(sel_industry) &
    df["Risk_Category"].isin(sel_risk) &
    (df["Monthly_Revenue_USD"] >= sel_rev[0]) &
    (df["Monthly_Revenue_USD"] <= sel_rev[1])
].copy()

# -----------------------------
# KPI helper
# -----------------------------
def kpi_row(data):
    total = len(data)
    high = int((data["Risk_Category"] == "High Risk").sum())
    churn_pct = (data["Churned"].mean() * 100) if total else 0
    avg_rev = data["Monthly_Revenue_USD"].mean() if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Clients", f"{total}")
    c2.metric("High Risk Clients", f"{high}")
    c3.metric("Churn Rate %", f"{churn_pct:.2f}%")
    c4.metric("Avg Revenue / Client", f"${avg_rev:,.2f}")

# =============================
# PAGE 1: OVERVIEW (Part D visuals)
# =============================
if page == "🏠 Overview":
    st.subheader("Executive Snapshot")
    kpi_row(f)

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Risk Category Distribution (Bar)")
        counts = f["Risk_Category"].value_counts().reindex(risk_levels).fillna(0)

        fig = plt.figure()
        plt.bar(counts.index, counts.values)
        plt.xlabel("Risk Category")
        plt.ylabel("Clients")
        st.pyplot(fig)
        st.caption("Shows overall client risk mix (Low / Medium / High).")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Revenue vs Risk (Bubble-style)")
        # Bubble effect using scatter with size = revenue scaled
        sizes = (f["Monthly_Revenue_USD"] - f["Monthly_Revenue_USD"].min() + 1) / 50.0
        fig2 = plt.figure()
        plt.scatter(f["Risk_Score"], f["Monthly_Revenue_USD"], s=sizes)
        plt.xlabel("Risk Score (0–10)")
        plt.ylabel("Monthly Revenue (USD)")
        st.pyplot(fig2)
        st.caption("Find high-revenue clients with high risk to prioritize retention.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Contract Length vs Churn (Binned)")
    if len(f) > 0:
        tmp = f.copy()
        tmp["Contract_Bin"] = pd.cut(tmp["Contract_Length_Months"], bins=6)
        churn_by_contract = tmp.groupby("Contract_Bin")["Churned"].mean() * 100

        fig3 = plt.figure()
        plt.plot(range(len(churn_by_contract)), churn_by_contract.values, marker="o")
        plt.xticks(range(len(churn_by_contract)), [str(x) for x in churn_by_contract.index], rotation=45, ha="right")
        plt.ylabel("Churn %")
        plt.xlabel("Contract Length Bin")
        st.pyplot(fig3)
    else:
        st.info("No data after filters.")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# PAGE 2: SEGMENTATION (Industry analysis + Top 20 table)
# =============================
elif page == "📌 Segmentation":
    st.subheader("Segmentation & Prioritization")
    kpi_row(f)

    st.divider()
    colA, colB = st.columns([1.1, 0.9])

    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Industry-wise Risk Analysis (Stacked view)")
        pivot = pd.pivot_table(
            f, index="Industry", columns="Risk_Category",
            values="Client_ID", aggfunc="count", fill_value=0
        ).reindex(columns=risk_levels, fill_value=0)

        # Stacked bar using matplotlib (different from earlier)
        fig = plt.figure()
        x = np.arange(len(pivot.index))
        bottom = np.zeros(len(pivot.index))
        for cat in risk_levels:
            plt.bar(x, pivot[cat].values, bottom=bottom, label=cat)
            bottom += pivot[cat].values
        plt.xticks(x, pivot.index, rotation=45, ha="right")
        plt.ylabel("Clients")
        plt.xlabel("Industry")
        plt.legend()
        st.pyplot(fig)
        st.caption("Highlights which industries have more High Risk clients.")
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### Risk Score Distribution (Histogram)")
        fig2 = plt.figure()
        plt.hist(f["Risk_Score"], bins=11)
        plt.xlabel("Risk Score (0–10)")
        plt.ylabel("Count")
        st.pyplot(fig2)
        st.caption("Shows how risk is spread across the filtered population.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Top 20 High-Risk Clients (Action List)")
    top20 = f.sort_values(["Risk_Score", "Monthly_Revenue_USD"], ascending=[False, False]).head(20)

    cols_show = [
        "Client_ID", "Company_Name", "Industry", "Region", "Plan",
        "Monthly_Usage_Score", "Payment_Delay_Days", "Contract_Length_Months",
        "Support_Tickets_Last30Days", "Monthly_Revenue_USD", "Risk_Score",
        "Risk_Category", "Renewal_Status"
    ]
    cols_show = [c for c in cols_show if c in top20.columns]

    # Simple conditional styling (makes it look different/pro)
    def highlight_risk(s):
        return ["font-weight:700;" if v >= 7 else "" for v in s]

    st.dataframe(top20[cols_show].style.apply(highlight_risk, subset=["Risk_Score"]), use_container_width=True)
    st.caption("Use this as the immediate call list for retention outreach.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🔎 Client Drill-Down")
    pick = st.selectbox("Select a client", sorted(f["Client_ID"].astype(str).unique()) if len(f) else [])
    if pick:
        row = f[f["Client_ID"].astype(str) == str(pick)].head(1)
        if len(row):
            r = row.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Risk Score", f"{r['Risk_Score']:.0f}/10")
            m2.metric("Risk Category", r["Risk_Category"])
            m3.metric("Revenue", f"${r['Monthly_Revenue_USD']:,.0f}")
            m4.metric("Renewal Status", r["Renewal_Status"])
            st.dataframe(row[cols_show], use_container_width=True)

# =============================
# PAGE 3: MODEL LAB (Part C)
# =============================
elif page == "🤖 Model Lab":
    st.subheader("Churn Prediction (Decision Tree)")
    st.caption("Target: Renewal_Status (Yes/No). Shows accuracy, confusion matrix, and feature importance.")

    # Features (mix categorical + numeric)
    feature_cols = [
        "Industry", "Region", "Plan", "Lead_Source",
        "Account_Age_Months", "Contract_Length_Months",
        "Monthly_Usage_Score", "Support_Tickets_Last30Days",
        "Payment_Delay_Days", "Monthly_Revenue_USD", "Risk_Score"
    ]
    X = df[feature_cols].copy()
    y = df["Renewal_Status"].map({"Yes": 1, "No": 0})

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    depth = st.slider("Tree Depth (controls complexity)", 2, 14, 6)
    min_leaf = st.slider("Min Samples per Leaf", 1, 50, 10)

    model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    a1, a2, a3 = st.columns(3)
    a1.metric("Accuracy", f"{acc:.4f}")
    a2.metric("Test Size", f"{len(X_test)}")
    a3.metric("Features Used", f"{X.shape[1]}")

    st.write("**Confusion Matrix** (rows = actual, columns = predicted)")
    st.write(confusion_matrix(y_test, pred))

    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
    st.write("**Top 12 Feature Importances**")
    st.bar_chart(imp)
    st.caption("Interpretation: Higher importance means stronger influence on churn prediction.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### Predicted Churn Probability for a Client (Demo)")
    client_pick = st.selectbox("Pick a client to score", sorted(df["Client_ID"].astype(str).unique()))
    if client_pick:
        row = df[df["Client_ID"].astype(str) == str(client_pick)].copy()
        rowX = pd.get_dummies(row[feature_cols], drop_first=True)
        # align columns
        rowX = rowX.reindex(columns=X.columns, fill_value=0)

        proba = model.predict_proba(rowX)[0][0]  # probability of class 0? careful: class order
        # Ensure probability for churn (Renewal_Status=No => 0)
        classes = model.classes_
        churn_index = list(classes).index(0)  # 0 corresponds to "No" (churn)
        churn_proba = model.predict_proba(rowX)[0][churn_index]

        st.write(f"Client **{client_pick}** estimated churn probability:")
        st.progress(float(churn_proba))
        st.write(f"**{churn_proba*100:.2f}%** (higher means more likely to churn)")

# =============================
# PAGE 4: ACTION CENTER (Part E)
# =============================
elif page == "🛠 Action Center":
    st.subheader("Retention Strategy Generator")
    st.caption("Click the button to generate 3–5 actions based on the selected (filtered) population.")

    # Quick signals from filtered data (looks different)
    if len(f) > 0:
        avg_delay = f["Payment_Delay_Days"].mean()
        avg_usage = f["Monthly_Usage_Score"].mean()
        avg_tickets = f["Support_Tickets_Last30Days"].mean()

        x1, x2, x3 = st.columns(3)
        x1.metric("Avg Payment Delay (days)", f"{avg_delay:.1f}")
        x2.metric("Avg Usage Score", f"{avg_usage:.1f}")
        x3.metric("Avg Tickets (30d)", f"{avg_tickets:.1f}")
    else:
        st.info("No rows in current filters. Adjust filters to generate insights.")

    st.divider()
    if st.button("Generate Retention Strategy"):
        st.success("Recommended Retention Actions")
        st.write("1) **Payment recovery:** For clients with payment delay > 30 days, offer a flexible plan + early-pay discount.")
        st.write("2) **Adoption boost:** For low usage clients, conduct onboarding refresh + training + weekly usage nudges.")
        st.write("3) **Support stabilization:** For high-ticket clients, assign a dedicated account manager and priority SLA.")
        st.write("4) **Renewal upgrade:** For short contracts, offer long-term incentives (discount, add-ons, feature bundles).")
        st.write("5) **Protect revenue:** For high-revenue & high-risk accounts, schedule leadership call + custom success roadmap.")

    st.divider()
    st.markdown("### Quick Target List (High Risk + High Revenue)")
    if len(f) > 0:
        target = f[(f["Risk_Category"] == "High Risk")].sort_values("Monthly_Revenue_USD", ascending=False).head(15)
        st.dataframe(target[["Client_ID","Company_Name","Industry","Region","Monthly_Revenue_USD","Risk_Score","Renewal_Status"]], use_container_width=True)
    else:
        st.write("-")

# =============================
# PAGE 5: RESPONSIBLE AI (Part F)
# =============================
elif page == "⚖️ Responsible AI":
    st.subheader("Ethical Implications of Predicting Client Churn")
    st.markdown(
        """
**1) Bias in predictive models**  
If some industries/regions historically churn more due to external conditions, the model can learn those patterns and unfairly label them as risky.

**2) Impact of labeling clients as “High Risk”**  
A “high risk” label can change how teams treat clients. If used to reduce service or increase strictness, it may increase churn (self-fulfilling outcome).

**3) Data privacy concerns**  
Usage, payment behavior, and support history are sensitive. Access should be role-based, data minimized, and stored securely.

**4) Responsible decision-making**  
Predictions are probabilities, not facts. They should support account managers—not replace human judgment and relationship context.

**5) Transparency & monitoring**  
Explain key drivers (feature importance), re-check performance over time, and audit for fairness across industries/regions.
        """
    )

# =============================
# PAGE 6: DATA EXPORT
# =============================
elif page == "📄 Data Export":
    st.subheader("Data View & Export")
    kpi_row(f)

    st.divider()
    st.write("Preview (first 200 rows of filtered data):")
    st.dataframe(f.head(200), use_container_width=True)

    st.download_button(
        "⬇️ Download Filtered Data (CSV)",
        data=f.to_csv(index=False).encode("utf-8"),
        file_name="filtered_b2b_clients.csv",
        mime="text/csv"
    )

    st.download_button(
        "⬇️ Download Top 20 High Risk (CSV)",
        data=f.sort_values(["Risk_Score","Monthly_Revenue_USD"], ascending=[False,False]).head(20).to_csv(index=False).encode("utf-8"),
        file_name="top20_high_risk.csv",
        mime="text/csv"
    )
