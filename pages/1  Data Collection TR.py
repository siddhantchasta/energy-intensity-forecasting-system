import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

# -------------------------------
# Page Header
# -------------------------------
st.header("**Data Collection**")

st.write("""
The dataset used in this project is sourced from the 
[WiDS Datathon 2022](https://www.kaggle.com/competitions/widsdatathon2022) 
and contains over **100,000 data samples**.
""")

# -------------------------------
# Background
# -------------------------------
st.subheader("Background")

st.write("""
According to a report published by the International Energy Agency (IEA), 
the lifecycle of buildings—from construction to demolition—accounted for 
approximately **37% of global energy-related and process-related CO₂ emissions in 2020**.

However, building energy consumption can be significantly reduced through a 
combination of easy-to-implement improvements and advanced energy-efficiency strategies. 
For example, retrofitted buildings can reduce heating and cooling energy requirements 
by **50–90%**.

Many energy-efficiency measures also lead to overall cost savings and additional benefits, 
such as improved indoor air quality for occupants. Accurate energy consumption prediction 
can help policymakers and stakeholders target retrofit initiatives more effectively 
to maximize emissions reductions.
""")

# -------------------------------
# Dataset Overview
# -------------------------------
st.subheader("Overview: Dataset and Challenge")

st.write("""
The WiDS Datathon dataset was developed in collaboration with 
Climate Change AI (CCAI) and Lawrence Berkeley National Laboratory (Berkeley Lab).

Participants analyze variations in building energy efficiency and develop 
machine learning models to predict building energy consumption. The dataset includes 
features describing building characteristics along with climate and weather variables 
specific to each building’s location.

Accurate energy consumption predictions can support policymakers in identifying 
high-impact retrofit opportunities and designing targeted emission-reduction strategies.
""")

# -------------------------------
# Dependent Variable
# -------------------------------
st.subheader("Dependent Variable")

st.write("""
**Site Energy Use Intensity (Site EUI)** is the dependent variable we aim to predict.

Site EUI is a key metric used to assess a building’s energy performance and identify 
opportunities for energy efficiency improvements. It also serves as an indicator of 
a building’s overall CO₂ emissions.

Site EUI depends on both building-specific characteristics and location-based climate 
factors. In this project, we develop a machine learning model to predict Site EUI 
and deploy it for real-time inference using a web-based application.
""")

# -------------------------------
# Independent Variables
# -------------------------------
st.subheader("Independent Variables")

st.write("""
- **id**: Unique building identifier  
- **Year_Factor**: An anonymized year when weather and energy data were recorded  
- **State_Factor**: Anonymized state location of the building  
- **building_class**: Building classification  
- **facility_type**: Type of building usage  
- **floor_area**: Total floor area of the building (square feet)  
- **year_built**: Year the building was constructed  
- **energy_star_rating**: Energy Star efficiency rating  
- **ELEVATION**: Elevation of the building location  
- **january_min_temp**: Minimum January temperature (°F)  
- **january_avg_temp**: Average January temperature (°F)  
- **january_max_temp**: Maximum January temperature (°F)  
- **cooling_degree_days**: Annual cooling degree days  
- **heating_degree_days**: Annual heating degree days  
- **precipitation_inches**: Annual precipitation (inches)  
- **snowfall_inches**: Annual snowfall (inches)  
- **snowdepth_inches**: Annual snow depth (inches)  
- **avg_temp**: Average annual temperature  
- **days_below_30F**: Number of days below 30°F  
- **days_below_20F**: Number of days below 20°F  
- **days_below_10F**: Number of days below 10°F  
- **days_below_0F**: Number of days below 0°F  
- **days_above_80F**: Number of days above 80°F  
- **days_above_90F**: Number of days above 90°F  
- **days_above_100F**: Number of days above 100°F  
- **days_above_110F**: Number of days above 110°F  
- **direction_max_wind_speed**: Wind direction for maximum wind speed (degrees)  
- **direction_peak_wind_speed**: Wind direction for peak gust speed (degrees)  
- **max_wind_speed**: Maximum wind speed  
- **days_with_fog**: Number of foggy days  
""")

# -------------------------------
# Dataset Preview
# -------------------------------
st.header("**Dataset Preview**")

df = pd.read_csv("train.csv")
st.write(df)

st.write("---")
