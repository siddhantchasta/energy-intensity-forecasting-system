import streamlit as st
st.set_page_config(layout="wide")

from PIL import Image
import numpy as np
import joblib
from xgboost import XGBRegressor

# -------------------------------
# Page Title & Banner
# -------------------------------
st.title("Energy Intensity Forecast Application 🏢⚡")
st.image(Image.open("Images/banner.jpg"))

# -------------------------------
# Load Trained Model
# -------------------------------
model = joblib.load("model_v2.joblib")

def get_forecast(data):
    model = joblib.load("model_v2.joblib")
    return model.predict(data)

# -------------------------------
# Main App Logic (Technical Report)
# -------------------------------
def main():
    with st.form('prediction_form'):
        st.subheader("Enter building characteristics and site weather conditions:")

        col0, col1, col2, col3 = st.columns([0.1, 0.4, 0.4, 0.1])

        # -------- Building Features --------
        floor_area = col1.number_input(
            label="Building floor area",
            min_value=1,
            step=1
        )

        building_type = col1.radio(
            "Select building type:",
            ["Commercial", "Residential"]
        )

        ratings = col1.slider(
            "Energy Star Rating",
            min_value=1,
            step=1,
            max_value=5
        )

        year_built = col1.number_input(
            "Year built",
            min_value=1920,
            step=1,
            max_value=2030
        )

        elevation = col1.number_input(
            label="Elevation of the building site",
            min_value=1,
            step=1
        )

        day30 = col1.number_input(
            label="Number of days below 30°F",
            min_value=1,
            step=1
        )

        # -------- Climate Features --------
        hdd = col2.number_input(
            label="Heating Degree Days",
            min_value=1,
            step=1
        )

        cdd = col2.number_input(
            label="Cooling Degree Days",
            min_value=1,
            step=1
        )

        precip = col2.number_input(
            label="Precipitation (in inches)"
        )

        snow = col2.number_input(
            label="Snowfall (in inches)"
        )

        day80 = col2.number_input(
            label="Number of days above 80°F"
        )

        # -------- Seasonal Temperature Averages --------
        avg_min_win = col1.slider("Average Minimum Winter Temperature")
        avg_max_win = col1.slider("Average Maximum Winter Temperature")
        avg_win = col1.slider("Average Winter Temperature")

        avg_min_sum = col2.slider("Average Minimum Summer Temperature")
        avg_max_sum = col2.slider("Average Maximum Summer Temperature")
        avg_sum = col2.slider("Average Summer Temperature")

        submit = st.form_submit_button("Forecast Site EUI")

    # -------------------------------
    # Prediction Logic
    # -------------------------------
    if submit:
        if building_type == "Commercial":
            building_type = 0
        elif building_type == "Residential":
            building_type = 1

        data = np.array([
            building_type,
            np.float64(floor_area),
            np.float64(year_built),
            np.float64(ratings),
            np.float64(elevation),
            np.int64(cdd),
            np.int64(hdd),
            np.float64(precip),
            np.float64(snow),
            np.int64(day80),
            np.float64(avg_min_win),
            np.float64(avg_max_win),
            np.float64(avg_win),
            np.float64(avg_min_sum),
            np.float64(avg_max_sum),
            np.float64(avg_sum),
            np.float64(avg_sum)  # kept same as original model input
        ]).reshape(1, -1)

        prediction = get_forecast(data)

        st.write(
            "Forecasted Site Energy Use Intensity (EUI): **{:.2f}** units"
            .format(prediction[0])
        )

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    main()
