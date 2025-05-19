import streamlit as st
import pandas as pd
import traceback
import sys
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

def main():
    try:
        st.set_page_config(page_title="Air Quality Predictor", layout="centered")
        st.title("🌫️ Advanced Air Quality Predictor with ML & Time-Series Support")

        # Load dataset from file
        st.header("📁 Dataset Overview")
        dataset_path = "AirQuality.csv"  # Make sure this file exists in your directory
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            st.success(f"Loaded dataset: {dataset_path}")
            st.dataframe(df)

            if not df.empty:
                st.header("⚙️ Select Features and Target")

                # Select input features
                input_features = st.multiselect(
                    "Input features",
                    options=df.columns.tolist(),
                    help="Select features to be used for training."
                )

                # Select target/output
                output_feature = st.selectbox(
                    "Output feature to predict",
                    options=[col for col in df.columns if col not in input_features],
                    help="Select the target column to predict."
                )

                if input_features and output_feature:
                    st.subheader("📈 Train Model and View Results")
                    if st.button("Train XGBoost Model"):
                        try:
                            X = df[input_features]
                            y = df[output_feature]
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                            model = XGBRegressor()
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            rmse = mean_squared_error(y_test, y_pred, squared=False)

                            st.success(f"Model trained successfully! RMSE on test set: {rmse:.2f}")
                        except Exception as model_err:
                            st.error("🚨 Model training failed.")
                            model_traceback = ''.join(traceback.format_exception(*sys.exc_info()))
                            print("Model Training Error:\n", model_traceback)
                            st.code(model_traceback)
        else:
            st.error(f"❌ Dataset file '{dataset_path}' not found. Please make sure it's in the app directory.")

    except Exception as e:
        st.error("⚠️ An unexpected error occurred.")
        error_traceback = ''.join(traceback.format_exception(*sys.exc_info()))
        print("App Error:\n", error_traceback)
        st.code(error_traceback)

if __name__ == "__main__":
    main()
