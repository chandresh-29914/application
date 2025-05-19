import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set page title
st.title("Air Quality Predictor")

# Load the dataset
try:
    # Load the CSV file with semicolon separator and comma as decimal
    df = pd.read_csv("airquality.csv", sep=";", decimal=",")
    st.success("Dataset loaded successfully!")
except FileNotFoundError:
    st.error("Dataset file 'airquality.csv' not found. Please ensure the file is in the same directory as this script.")
    df = pd.DataFrame()  # Empty DataFrame to prevent errors
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    df = pd.DataFrame()

# Proceed if dataset is loaded
if not df.empty:
    # Clean the dataset (remove extra empty columns if any)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
    df = df.iloc[:, :-2]  # Remove the last two empty columns (based on dataset structure)

    # Replace missing values (-200) with NaN and drop rows with missing values
    df.replace(-200, pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Display dataset preview
    st.write("### Dataset Preview")
    st.dataframe(df)

    # Preprocess DateTime column
    if "Date" in df.columns and "Time" in df.columns:
        # Combine Date and Time into a single DateTime column
        df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S")
        df["Year"] = df["DateTime"].dt.year
        df["Month"] = df["DateTime"].dt.month
        df["Day"] = df["DateTime"].dt.day
        df["Hour"] = df["DateTime"].dt.hour
        df = df.drop(columns=["Date", "Time", "DateTime"])  # Drop original columns

    # Feature and Target Selection
    st.write("### Select Features and Target")

    # Get all columns
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

        # Train a RandomForest model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Display metrics
        st.write("### Model Performance")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

else:
    st.write("No results to display. Please ensure the dataset file is loaded correctly.")
