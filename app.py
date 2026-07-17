import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Load Saved Project
# -----------------------------
project = joblib.load("house_price_prediction.pkl")

model = project["model"]
r2 = project["r2_score"]
mae = project["mae"]
rmse = project["rmse"]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📘 Project Information")

st.sidebar.markdown("### Algorithm")
st.sidebar.success("Multiple Linear Regression")

st.sidebar.markdown("### Model Performance")
st.sidebar.write(f"**R² Score:** {r2:.4f}")
st.sidebar.write(f"**MAE:** {mae:,.0f}")
st.sidebar.write(f"**RMSE:** {rmse:,.0f}")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Nadeem Abdullah**")

# -----------------------------
# Title
# -----------------------------
st.title("🏠 House Price Prediction System")

st.markdown("""
Predict the estimated selling price of a house using a **Multiple Linear Regression** model.
""")

st.markdown("---")

# -----------------------------
# Two Column Layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area (Square Feet)",
        min_value=500,
        max_value=20000,
        value=5000,
        step=100
    )

    bedrooms = st.selectbox(
        "Bedrooms",
        [1,2,3,4,5,6]
    )

    bathrooms = st.selectbox(
        "Bathrooms",
        [1,2,3,4]
    )

    stories = st.selectbox(
        "Stories",
        [1,2,3,4]
    )

    parking = st.selectbox(
        "Parking Spaces",
        [0,1,2,3]
    )

with col2:

    mainroad = st.selectbox(
        "Main Road",
        ["Yes","No"]
    )

    guestroom = st.selectbox(
        "Guest Room",
        ["Yes","No"]
    )

    basement = st.selectbox(
        "Basement",
        ["Yes","No"]
    )

    hotwater = st.selectbox(
        "Hot Water Heating",
        ["Yes","No"]
    )

    airconditioning = st.selectbox(
        "Air Conditioning",
        ["Yes","No"]
    )

    prefarea = st.selectbox(
        "Preferred Area",
        ["Yes","No"]
    )

    furnishing = st.radio(
        "Furnishing Status",
        (
            "Furnished",
            "Semi-furnished",
            "Unfurnished"
        )
    )

# -----------------------------
# Helper Function
# -----------------------------
def yn(value):
    return 1 if value == "Yes" else 0

mainroad = yn(mainroad)
guestroom = yn(guestroom)
basement = yn(basement)
hotwater = yn(hotwater)
airconditioning = yn(airconditioning)
prefarea = yn(prefarea)

semi = 0
unfurnished = 0

if furnishing == "Semi-furnished":
    semi = 1

elif furnishing == "Unfurnished":
    unfurnished = 1

# -----------------------------
# Prediction Button
# -----------------------------
st.markdown("---")

if st.button("🔍 Predict House Price", use_container_width=True):

    features = np.array([[
        area,
        bedrooms,
        bathrooms,
        stories,
        mainroad,
        guestroom,
        basement,
        hotwater,
        airconditioning,
        parking,
        prefarea,
        semi,
        unfurnished
    ]])

    prediction = model.predict(features)[0]

    st.success("Prediction Completed Successfully!")

    st.markdown("## 💰 Estimated House Price")

    st.metric(
        label="Predicted Price",
        value=f"Rs. {prediction:,.0f}"
    )

    st.info(
        "This prediction is generated using a Multiple Linear Regression model trained on the Housing Prices dataset."
    )

st.markdown("---")

st.caption(
    "Machine Learning Assignment | Multiple Linear Regression | Streamlit"
)
