import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ========== Load model and scaler ==========
with open('models/attrition_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/model_features.pkl', 'rb') as f:
    model_columns = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ========== Load data for EDA ==========
@st.cache_data
def load_data():
    df = pd.read_csv('data/employee.csv')
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df

df = load_data()

# ========== Page Navigation ==========
st.set_page_config(page_title="Attrition App", layout="wide")
page = st.sidebar.selectbox("📌 Navigate", ["🏠 Home", "📊 EDA", "🔮 Prediction", "👤 Creator"])

# ==============================
# 🏠 Home Page
# ==============================
if page == "🏠 Home":
    st.title("👥 Employee Attrition Prediction App")
    st.markdown("""
    ---
    This interactive dashboard helps HR teams **predict whether an employee is likely to leave** the company based on key features like job satisfaction, income, and more.

    ### 🎯 Project Goals
    - Visualize HR data and attrition trends
    - Use ML to predict attrition risk
    - Help make proactive retention decisions

    ### 🔍 Model
    - Logistic Regression
    - Top 10 Features only
    - Balanced with SMOTE for fair prediction
    ---
    """)

# ==============================
# 📊 EDA Page
# ==============================
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition Count")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Attrition', data=df, ax=ax1)
        st.pyplot(fig1)

        st.subheader("Monthly Income by Attrition")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, ax=ax2)
        st.pyplot(fig2)

    with col2:
        st.subheader("Attrition by Overtime")
        fig3, ax3 = plt.subplots()
        sns.countplot(x='OverTime', hue='Attrition', data=df, ax=ax3)
        st.pyplot(fig3)

        st.subheader("Age Distribution")
        fig4, ax4 = plt.subplots()
        sns.histplot(data=df, x='Age', hue='Attrition', kde=True, bins=20, ax=ax4)
        st.pyplot(fig4)

# ==============================
# 🔮 Prediction Page
# ==============================
elif page == "🔮 Prediction":
    st.title("🔮 Predict Employee Attrition")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 18, 60, 28)
            monthly_income = st.number_input("Monthly Income", 1000, 20000, value=3000, step=500)
            job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
            environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
            overtime = st.radio("Works Overtime?", ["Yes", "No"])

        with col2:
            distance = st.slider("Distance From Home", 0, 50, 30)
            years_at_company = st.slider("Years at Company", 0, 40, 1)
            num_companies = st.slider("Number of Companies Worked", 0, 10, 3)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            job_role = st.selectbox("Job Role", ["Sales Executive", "Other"])

        submitted = st.form_submit_button("📈 Predict")

    if submitted:
        # One-hot encode input
        input_dict = {
            'Age': age,
            'MonthlyIncome': monthly_income,
            'JobSatisfaction': job_satisfaction,
            'OverTime_Yes': 1 if overtime == "Yes" else 0,
            'DistanceFromHome': distance,
            'EnvironmentSatisfaction': environment_satisfaction,
            'YearsAtCompany': years_at_company,
            'NumCompaniesWorked': num_companies,
            'MaritalStatus_Single': 1 if marital_status == "Single" else 0,
            'JobRole_Sales Executive': 1 if job_role == "Sales Executive" else 0
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        st.subheader("🧪 Input to Model")
        st.dataframe(input_df)

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        # Display
        if prediction == 1:
            st.error(f"⚠️ High Attrition Risk (Score: {prob:.2f})")
        else:
            st.success(f"✅ Low Attrition Risk (Confidence: {1 - prob:.2f})")

# ==============================
# 👤 Creator Page
# ==============================
elif page == "👤 Creator":
    st.title("👤 About the Creator")
    st.markdown("""
    ---
    - **Name**: Thirukumaran 
    - **GitHub**: https://github.com/itzzthiru

    Built with ❤️ using Python, Streamlit, and scikit-learn.
    ---
    """)
