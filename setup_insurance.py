import pandas as pd
from xgboost import XGBRegressor
from snowflake.ml.registry import Registry

def setup_snowflake():
    # 1. Connect to Snowflake
    print("Connecting to Snowflake...")
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
    print("Connected successfully inside Snowflake!")

    # 2. Prepare the Data
    print("Loading Medical Insurance data from Snowflake...")
    df = session.table("ML_DEMO.PUBLIC.INSURANCE_DATA").to_pandas()
    
    # Convert all columns to uppercase to match Snowflake standards
    df.columns = [col.upper() for col in df.columns]
    
    # Map categorical features to numbers so XGBoost can process them
    df['SEX'] = df['SEX'].map({'male': 0, 'female': 1})
    df['SMOKER'] = df['SMOKER'].map({'yes': 1, 'no': 0})
    df['REGION'] = df['REGION'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})
    
    features = ['AGE', 'SEX', 'BMI', 'CHILDREN', 'SMOKER', 'REGION']
    target = 'CHARGES'
    
    df = df.fillna(0)

    # 3. Train the XGBoost Regressor Model
    print("Training the XGBoost Machine Learning Model...")
    xgb_model = XGBRegressor()
    xgb_model.fit(df[features], df[target])
    
    # 4. Save to Snowflake Model Registry
    print("Saving model to Snowflake Registry...")
    registry = Registry(session, database_name="ML_DEMO", schema_name="PUBLIC")
    
    try:
        registry.delete_model("INSURANCE_XGB_MODEL")
    except:
        pass

    registry.log_model(
        model=xgb_model,
        model_name="INSURANCE_XGB_MODEL",
        version_name="V1",
        sample_input_data=df[features].head(),
        target_platforms=["WAREHOUSE"]
    )
    
    print("🎉 Success! The model is trained and saved in Snowflake as INSURANCE_XGB_MODEL (V1)!")

if __name__ == "__main__":
    setup_snowflake()
