import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------- Page Setup & Styling --------------------
st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .title-box {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .stButton > button {
        background-color: #ffffff;
        color: #1e3c72;
        border: 3px solid transparent;
        padding: 0.75em 1.5em;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        transition: 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #dbe9ff;
    }
    .stButton.active > button {
        border: 3px solid #00cc88 !important;
        background-color: #c6fff1 !important;
        color: #1e3c72 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-box"><h1>💳 Credit Card Fraud Detector</h1><h4>Explore Supervised vs Unsupervised ML Analysis</h4></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding-top: 10px;'>
    <a href='https://github.com/Brenhuber/CardFraudDetect/blob/main/models.ipynb' target='_blank' style='color: #ffffff; font-size: 16px; text-decoration: none;'>
        🔗 View Full Jupyter Notebook
    </a>
</div>
""", unsafe_allow_html=True)

# -------------------- Cache Logic --------------------
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(r"C:\Code Files\Data Files\creditcard.csv")
    df = df.drop_duplicates().dropna()
    df["Amount"] = StandardScaler().fit_transform(df[["Amount"]])
    df["Time"] = StandardScaler().fit_transform(df[["Time"]])
    return df

@st.cache_resource
def train_supervised(df):
    X = df.drop(columns=['Class'])
    y = df['Class']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(C=10, penalty='l1', solver='liblinear')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred, output_dict=True)
    fraud_count = (df['Class'] == 1).sum()
    importance = pd.Series(model.coef_[0], index=X.columns).abs().sort_values(ascending=False).reset_index()
    importance.columns = ['Feature', 'Importance']

    return acc, fraud_count, report, importance

@st.cache_resource
def train_unsupervised(df):
    model = IsolationForest(n_estimators=400, contamination=0.0025, random_state=42)
    df = df.copy()
    df['anomaly_score'] = model.fit_predict(df.drop(columns=['Class']))

    anomaly_count = (df['anomaly_score'] == -1).sum()
    y_true = df['Class']
    y_pred = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

    acc = accuracy_score(y_true, y_pred) * 100
    report = classification_report(y_true, y_pred, output_dict=True)

    return acc, anomaly_count, report

# -------------------- Load + Train Once --------------------
df = load_and_preprocess_data()
sup_acc, sup_fraud_count, sup_report, importance_df = train_supervised(df)
unsup_acc, unsup_anomaly_count, unsup_report = train_unsupervised(df)

# -------------------- Session Setup --------------------
if 'mode' not in st.session_state:
    st.session_state.mode = 'supervised'

# -------------------- Buttons --------------------
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    b1 = st.button("👀 Supervised", key="sup")
    if b1:
        st.session_state.mode = 'supervised'
with col2:
    b2 = st.button("🧠 Unsupervised", key="unsup")
    if b2:
        st.session_state.mode = 'unsupervised'
with col3:
    b3 = st.button("📊 Compare Both", key="compare")
    if b3:
        st.session_state.mode = 'compare'

# -------------------- Add active class to selected button --------------------
active_css = """
<script>
const buttons = window.parent.document.querySelectorAll('.stButton');
buttons.forEach((btn, i) => {
    btn.classList.remove('active');
});
buttons[%d].classList.add('active');
</script>
""" % {'supervised': 0, 'unsupervised': 1, 'compare': 2}[st.session_state.mode]
st.components.v1.html(active_css, height=0)

st.markdown("---")

# -------------------- Content Areas --------------------
if st.session_state.mode == 'supervised':
    st.subheader("📊 Supervised Analysis (Logistic Regression)")
    df_rep = pd.DataFrame(sup_report).T.reset_index().rename(columns={'index': 'Class'})
    fig = px.bar(df_rep[df_rep['Class'].isin(['0', '1'])], x='Class',
                 y=['precision', 'recall', 'f1-score'],
                 barmode='group', title='Supervised Classification Report')
    st.metric("Accuracy", f"{sup_acc:.2f}%")
    st.metric("Anomaly Amount", sup_fraud_count)
    st.plotly_chart(fig, use_container_width=True)
    fig_importance = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importance')
    st.plotly_chart(fig_importance, use_container_width=True)

elif st.session_state.mode == 'unsupervised':
    st.subheader("🔍 Unsupervised Analysis (Isolation Forest)")
    df_rep = pd.DataFrame(unsup_report).T.reset_index().rename(columns={'index': 'Class'})
    fig = px.bar(df_rep[df_rep['Class'].isin(['0', '1'])], x='Class',
                 y=['precision', 'recall', 'f1-score'],
                 barmode='group', title='Unsupervised Classification Report')
    st.metric("Accuracy", f"{unsup_acc:.2f}%")
    st.metric("Anomalies Detected", unsup_anomaly_count)
    st.plotly_chart(fig, use_container_width=True)

elif st.session_state.mode == 'compare':
    st.subheader("📈 Comparison: Supervised vs Unsupervised")
    comp_df = pd.DataFrame({
        'Method': ['Supervised', 'Unsupervised'],
        'Accuracy': [sup_acc, unsup_acc],
        'Anomalies Detected': [sup_fraud_count, unsup_anomaly_count],
        'F1-Score (Fraud)': [
            sup_report['1']['f1-score'],
            unsup_report['1']['f1-score']
        ]
    })
    fig = px.bar(comp_df.melt(id_vars='Method'), x='Method', y='value', color='variable',
                 barmode='group', title="Model Comparison Metrics")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Select an analysis method to get started.")
