import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Example model
from sklearn.metrics import mean_squared_error, r2_score

# Set page title
st.title("Air Quality Predictor")

# Load the dataset (replace 'air_quality_data.csv' with your actual dataset file)
# For this example, I’ll assume the dataset is in the same directory as the app
try:
    df = pd.read_csv("air_quality_data.csv")
except FileNotFoundError:
    st.error("Dataset file 'air_quality_data.csv' not found. Please upload the dataset.")
    df = pd.DataFrame()  # Empty dataframe to avoid errors

# Display the dataset
if not df.empty:
    st.write("### Dataset Preview")
    st.dataframe(df)

    # Preprocess DateTime column if it exists
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df["Year"] = df["DateTime"].dt.year
        df["Month"] = df["DateTime"].dt.month
        df["Day"] = df["DateTime"].dt.day
        df["Hour"] = df["DateTime"].dt.hour
        df = df.drop(columns=["DateTime"])  # Drop the original DateTime column

    # Feature and Target Selection
    st.write("### Select Features and Target")

    # Get all columns except DateTime (already processed)
    feature_options = df.columns.tolist()

    # Select input features
    input_features = st.multiselect(
        "Input features",
        options=feature_options,
        default=[col for col in feature_options if col != "CO(GT)"]  # Default: all except target
    )

    # Select target variable
    target = st.selectbox(
        "Target variable",
        options=feature_options,
        index=feature_options.index("CO(GT)") if "CO(GT)" in feature_options else 0
    )

    # Validate selections
    if not input_features:
        st.error("Please select at least one input feature.")
    elif target in input_features:
        st.error("Target variable cannot be in the input features.")
    else:
        # Prepare data for modeling
        X = df[input_features]
        y = df[target]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a simple model (RandomForest as an example)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Display metrics
        st.write("### Model Performance")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

else:
    st.write("No results to display. Please ensure the dataset is loaded correctly.")
