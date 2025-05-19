import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset from file
@st.cache_data
def load_data():
    return pd.read_csv("AirQuality.csv")

# App title
st.title("🌫️ Air Quality Prediction & Visualization")

# Load data
df = load_data()

st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# Clean column names
df.columns = df.columns.str.strip()

# Drop non-numeric columns (like Date, Time) if they exist
non_numeric = df.select_dtypes(include=["object"]).columns.tolist()
df = df.drop(columns=non_numeric, errors="ignore")

# Drop rows with missing values
df = df.dropna()

# Show data summary
st.subheader("📊 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Input/Output selection
st.subheader("⚙️ Select Inputs and Output for Prediction")
all_columns = df.columns.tolist()

input_features = st.multiselect("Select Input Features (X)", options=all_columns, default=all_columns[:-1])
output_feature = st.selectbox("Select Output Target (y)", options=[col for col in all_columns if col not in input_features])

if input_features and output_feature:
    # Split data
    X = df[input_features]
    y = df[output_feature]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    st.subheader("📈 Model Performance")
    st.write(f"R² Score: `{r2_score(y_test, y_pred):.3f}`")
    st.write(f"RMSE: `{mean_squared_error(y_test, y_pred, squared=False):.3f}`")

    # Plot predictions vs actual
    st.subheader("📉 Actual vs Predicted")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax2)
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.set_title("Actual vs Predicted")
    st.pyplot(fig2)

    # Make new prediction
    st.subheader("🧪 Make a New Prediction")

    input_values = {}
    for col in input_features:
        val = st.number_input(f"Input value for {col}", value=float(X[col].mean()))
        input_values[col] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([input_values])
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted {output_feature}: {prediction:.3f}")
