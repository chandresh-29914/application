import streamlit as st
import pandas as pd
import numpy as np
import traceback
import sys
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def main():
    st.set_page_config(page_title="Air Quality Predictor", layout="centered")
    st.title("🌫️ Advanced Air Quality Predictor")

    try:
        dataset_path = "AirQuality.csv"
        df = pd.read_csv(dataset_path)
        st.subheader("📄 Dataset Preview")
        st.dataframe(df)

        if df.empty:
            st.error("Dataset is empty.")
            return

        st.subheader("⚙️ Select Features and Target")
        input_features = st.multiselect("Select input features", df.columns.tolist())
        output_feature = st.selectbox("Select target feature to predict", [col for col in df.columns if col not in input_features])

        if input_features and output_feature:
            X = df[input_features]
            y = df[output_feature]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = XGBRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            st.success(f"Model trained! RMSE: {rmse:.2f}")

            st.subheader("🎯 Make a Prediction")
            user_input = {}
            for feature in input_features:
                user_input[feature] = st.number_input(f"Enter value for {feature}", value=float(df[feature].mean()))

            if st.button("Predict"):
                input_df = pd.DataFrame([user_input])
                prediction = model.predict(input_df)[0]
                st.success(f"Predicted {output_feature}: {prediction:.2f}")

                # Visualization
                st.subheader("📊 Prediction Visualization")
                fig, ax = plt.subplots()
                ax.barh([output_feature], [prediction], color='skyblue')
                ax.set_xlabel("Predicted Value")
                ax.set_title("Prediction Result")
                st.pyplot(fig)

    except Exception:
        st.error("An error occurred.")
        st.code(''.join(traceback.format_exception(*sys.exc_info())))

if __name__ == "__main__":
    main()
