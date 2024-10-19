import streamlit as st
import pandas as pd
import plotly.express as px
from credit_card_attrition_model import predict_attrition, get_feature_importances, calculate_financial_health

st.set_page_config(layout="wide")

st.title("Credit Card Customer Attrition Prediction and Financial Health Assessment")

# Sidebar for user input
st.sidebar.header("Enter Customer Information")
customer_age = st.sidebar.slider("Customer Age", 18, 100, 45)
gender = st.sidebar.selectbox("Gender", ["M", "F"])
dependent_count = st.sidebar.slider("Number of Dependents", 0, 10, 3)
education_level = st.sidebar.selectbox("Education Level", ["High School", "Graduate", "Uneducated", "Unknown", "College", "Post-Graduate", "Doctorate"])
marital_status = st.sidebar.selectbox("Marital Status", ["Married", "Single", "Unknown", "Divorced"])
income_category = st.sidebar.selectbox("Income Category", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"])
card_category = st.sidebar.selectbox("Card Category", ["Blue", "Silver", "Gold", "Platinum"])
months_on_book = st.sidebar.slider("Months on Book", 0, 60, 39)
total_relationship_count = st.sidebar.slider("Total Relationship Count", 1, 10, 5)
months_inactive_12_mon = st.sidebar.slider("Months Inactive (12 mon)", 0, 12, 1)
contacts_count_12_mon = st.sidebar.slider("Contacts Count (12 mon)", 0, 10, 3)
credit_limit = st.sidebar.number_input("Credit Limit", 1000, 100000, 12000)
total_revolving_bal = st.sidebar.number_input("Total Revolving Balance", 0, 50000, 1000)
avg_open_to_buy = st.sidebar.number_input("Average Open to Buy", 0, 100000, 11000)
total_amt_chng_q4_q1 = st.sidebar.number_input("Total Amount Change Q4-Q1", 0.0, 5.0, 1.3)
total_trans_amt = st.sidebar.number_input("Total Transaction Amount", 0, 20000, 1500)
total_trans_ct = st.sidebar.number_input("Total Transaction Count", 0, 200, 40)
total_ct_chng_q4_q1 = st.sidebar.number_input("Total Count Change Q4-Q1", 0.0, 5.0, 1.6)
avg_utilization_ratio = st.sidebar.number_input("Average Utilization Ratio", 0.0, 1.0, 0.1)

# Create input dataframe
input_data = pd.DataFrame({
    'Customer_Age': [customer_age],
    'Gender': [gender],
    'Dependent_count': [dependent_count],
    'Education_Level': [education_level],
    'Marital_Status': [marital_status],
    'Income_Category': [income_category],
    'Card_Category': [card_category],
    'Months_on_book': [months_on_book],
    'Total_Relationship_Count': [total_relationship_count],
    'Months_Inactive_12_mon': [months_inactive_12_mon],
    'Contacts_Count_12_mon': [contacts_count_12_mon],
    'Credit_Limit': [credit_limit],
    'Total_Revolving_Bal': [total_revolving_bal],
    'Avg_Open_To_Buy': [avg_open_to_buy],
    'Total_Amt_Chng_Q4_Q1': [total_amt_chng_q4_q1],
    'Total_Trans_Amt': [total_trans_amt],
    'Total_Trans_Ct': [total_trans_ct],
    'Total_Ct_Chng_Q4_Q1': [total_ct_chng_q4_q1],
    'Avg_Utilization_Ratio': [avg_utilization_ratio]
})

# Make predictions
attrition_probability = predict_attrition(input_data)
financial_health_score = calculate_financial_health(input_data)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Attrition Risk Assessment")
    st.metric("Attrition Probability", f"{attrition_probability:.2%}")
    
    if attrition_probability < 0.3:
        st.success("Low risk of customer attrition. Keep up the good work!")
    elif attrition_probability < 0.7:
        st.warning("Moderate risk of customer attrition. Consider implementing retention strategies.")
    else:
        st.error("High risk of customer attrition. Immediate action is recommended to retain this customer.")

    st.subheader("Financial Health Score")
    st.metric("Financial Health", f"{financial_health_score:.1f}/100")
    
    if financial_health_score >= 80:
        st.success("Excellent financial health. The customer is managing their credit well.")
    elif financial_health_score >= 60:
        st.info("Good financial health. There's room for improvement in credit management.")
    elif financial_health_score >= 40:
        st.warning("Fair financial health. The customer may benefit from credit counseling.")
    else:
        st.error("Poor financial health. The customer needs immediate financial advice and support.")

with col2:
    st.subheader("Feature Importance")
    importances = get_feature_importances()
    imp_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])
    imp_df = imp_df.sort_values('Importance', ascending=False).head(10)
    fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig)

# Recommendations
st.header("Personalized Recommendations")

if attrition_probability >= 0.7:
    st.write("1. Reach out to the customer to address any concerns or dissatisfaction")
    st.write("2. Offer personalized incentives or rewards to encourage continued card usage")
    st.write("3. Provide information about additional benefits or services they may not be aware of")
elif attrition_probability >= 0.3:
    st.write("1. Send a customer satisfaction survey to identify potential issues")
    st.write("2. Highlight card benefits that align with the customer's spending habits")
    st.write("3. Consider offering a loyalty program or enhanced rewards for continued usage")
else:
    st.write("1. Thank the customer for their loyalty and continued business")
    st.write("2. Offer upgrades or additional services that may benefit the customer")
    st.write("3. Provide financial education resources to help maintain good credit habits")

if financial_health_score < 60:
    st.write("4. Recommend credit counseling or financial planning services")
    st.write("5. Provide resources on budgeting and managing credit card debt")
    st.write("6. Consider offering a balance transfer option or lower interest rate if appropriate")

# Blog and Information Section
st.header("Credit Card Tips and Information")

tab1, tab2, tab3 = st.tabs(["Credit Score Basics", "Managing Credit Card Debt", "Maximizing Card Benefits"])

with tab1:
    st.subheader("Understanding Your Credit Score")
    st.write("""
    1. **Payment History (35%)**: Always pay your bills on time.
    2. **Credit Utilization (30%)**: Keep your credit card balances low.
    3. **Length of Credit History (15%)**: Maintain long-standing accounts responsibly.
    4. **Credit Mix (10%)**: Have a diverse mix of credit types.
    5. **New Credit (10%)**: Avoid opening too many new accounts in a short period.
    """)

with tab2:
    st.subheader("Strategies to Manage Credit Card Debt")
    st.write("""
    1. **Pay More Than the Minimum**: This reduces interest and pays off debt faster.
    2. **Debt Snowball Method**: Pay off smallest debts first for psychological wins.
    3. **Debt Avalanche Method**: Focus on highest interest debts first to save money.
    4. **Balance Transfer**: Move high-interest debt to a 0% APR card, but beware of fees.
    5. **Create a Budget**: Track spending and cut unnecessary expenses to free up money for debt payments.
    """)

with tab3:
    st.subheader("How to Maximize Your Credit Card Benefits")
    st.write("""
    1. **Understand Your Rewards Program**: Know how to earn and redeem points or cashback.
    2. **Use Category Bonuses**: Maximize rewards by using the right card for each purchase category.
    3. **Take Advantage of Sign-Up Bonuses**: Meet spending requirements to earn substantial rewards.
    4. **Utilize Travel Benefits**: Many cards offer travel insurance, lounge access, or hotel status.
    5. **Read the Fine Print**: Be aware of annual fees, foreign transaction fees, and other terms.
    """)

st.sidebar.info("Disclaimer: This app provides general financial advice based on limited information. For personalized financial planning, please consult with a certified financial advisor.")