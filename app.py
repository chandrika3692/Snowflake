import streamlit as st
import pandas as pd
from snowflake.ml.registry import Registry

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🏥", layout="wide")
st.title("🏥 Medical Insurance Cost Predictor & Analytics")
st.write("Enter patient details below to predict their estimated medical insurance charges.")

# Connect to Snowflake
try:
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
except Exception as e:
    st.error("Failed to connect to Snowflake.")
    st.stop()

# --- PREDICTION SECTION ---
with st.form("prediction_form"):
    st.subheader("Patient Details")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
        
    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoker?", ["yes", "no"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
        
    submitted = st.form_submit_button("Predict Insurance Charges")

if submitted:
    st.info("Querying Snowflake XGBoost Model...")
    try:
        registry = Registry(session, database_name="ML_DEMO", schema_name="PUBLIC")
        model = registry.get_model("INSURANCE_XGB_MODEL").version("V1")
        
        # Map categorical variables exactly how the model was trained
        sex_mapped = 0 if sex == 'male' else 1
        smoker_mapped = 1 if smoker == 'yes' else 0
        region_mapping = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
        region_mapped = region_mapping[region]
        
        input_data = pd.DataFrame([{
            'AGE': age, 
            'SEX': sex_mapped, 
            'BMI': bmi, 
            'CHILDREN': children, 
            'SMOKER': smoker_mapped, 
            'REGION': region_mapped
        }])
        
        result_df = model.run(input_data, function_name="predict")
        prediction_col = result_df.columns[-1]
        prediction = result_df[prediction_col][0]
        
        st.success(f"💰 Predicted Medical Insurance Charges: **${prediction:,.2f}**")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# --- ANALYTICS & GRAPHS SECTION ---
st.markdown("---")
st.subheader("📊 Historical Insurance Insights")

try:
    data = session.table("ML_DEMO.PUBLIC.INSURANCE_DATA").to_pandas()
    data.columns = [col.upper() for col in data.columns]
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**Average Charges: Smoker vs Non-Smoker**")
        smoker_charges = data.groupby('SMOKER')['CHARGES'].mean()
        st.bar_chart(smoker_charges, color="#e74c3c")
        
    with chart_col2:
        st.markdown("**Average Charges by Region**")
        region_charges = data.groupby('REGION')['CHARGES'].mean()
        st.bar_chart(region_charges, color="#f39c12")
        
except Exception as e:
    st.warning("Could not load charts. Make sure the INSURANCE_DATA table exists!")
