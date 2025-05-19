import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

# Load and clean dataset
@st.cache_data
def load_data():
    df = pd.read_csv("AirQuality.csv")

    # Drop object columns (like Date, Time) or convert them
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            if df[col].nunique() > 100:  # remove high cardinality timestamps
                df = df.drop(columns=[col])
            else:
                df[col] = df[col].astype(str)
        except:
            df = df.drop(columns=[col])

    # Drop rows with NaNs
    df = df.dropna()

    # Encode any remaining string columns
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df

# App UI
st.title("🌫️ Advanced Air Quality Predictor with ML & Time-Series Support")

# Load data
df = load_data()
st.subheader("📄 Dataset Overview")
st.dataframe(df.head())

# Feature selection
st.subheader("⚙️ Select Features and Target")

all_columns = df.columns.tolist()
input_features = st.multiselect("Input features", options=all_columns, default=all_columns[:-1])
output_feature = st.selectbox("Output feature to predict", options=[col for col in all_columns if col not in input_features])

# Model selection
st.subheader("🤖 Select Machine Learning Model")
model_choice = st.selectbox("Choose model", ["Random Forest", "XGBoost", "LightGBM"])

# If valid selections made
if input_features and output_feature:
    X = df[input_features]
    y = df[output_feature]

    # Time-series split or random split
    st.subheader("📆 Train/Test Split Type")
    split_type = st.radio("Choose split method", ["Random Split", "Time-Series Split"])
    test_size = st.slider("Test set size (%)", 10, 50, 20)

    if split_type == "Time-Series Split":
        split_index = int(len(X) * (1 - test_size / 100))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    # Train model
    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_choice == "XGBoost":
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    elif model_choice == "LightGBM":
        model = lgb.LGBMRegressor(n_estimators=100)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Show performance
    st.subheader("📊 Model Performance")
    st.write(f"**R² Score**: `{r2_score(y_test, y_pred):.3f}`")
    st.write(f"**RMSE**: `{mean_squared_error(y_test, y_pred, squared=False):.3f}`")

    # Plot actual vs predicted
    st.subheader("📉 Actual vs Predicted Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

    # New input for prediction
    st.subheader("🧪 Make a New Prediction")
    input_data = {}
    for col in input_features:
        val = st.number_input(f"Value for {col}", value=float(X[col].mean()))
        input_data[col] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]
        st.success(f"🔮 Predicted {output_feature}: **{pred:.3f}**")
