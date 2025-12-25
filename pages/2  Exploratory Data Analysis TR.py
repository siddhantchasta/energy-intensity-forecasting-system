import streamlit as stl
stl.set_page_config(layout="wide")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import codecs

# -------------------------------
# Helper Function
# -------------------------------
def count_plot(variable, num, data):
    fig = plt.figure(figsize=(12, 7))
    sns.countplot(
        y=variable,
        data=data,
        color="salmon",
        facecolor=(0, 0, 0, 0),
        linewidth=5,
        edgecolor=sns.color_palette("BrBG", num)
    )
    plt.title("Train Dataset")
    return plt


# -------------------------------
# Main EDA Logic
# -------------------------------
def main():

    df = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")

    # -------------------------------
    # Dataset Preview
    # -------------------------------
    stl.header("**Dataset Preview**")
    stl.write(df)
    stl.write("---")

    # -------------------------------
    # EDA Introduction
    # -------------------------------
    stl.header("**Exploratory Data Analysis**")
    stl.write("""
    The objective of this analysis is to explore the training dataset and 
    understand how each feature relates to the target variable **Site EUI**
    (Site Energy Use Intensity – the amount of heat and electricity consumed by a building).

    The dataset contains **64 attributes**, including the target variable.
    """)

    stl.write("""
    Below is an interactive profile report of the dataset to better understand 
    distributions, missing values, and correlations. Click on variable names 
    to expand the associated visualizations.
    """)

    report_file = codecs.open("Pages/EDA.html", "r")
    page = report_file.read()
    components.html(page, width=2000, height=1500, scrolling=True)

    # -------------------------------
    # Dataset Insights
    # -------------------------------
    stl.header("**Key Insights from the Dataset**")

    stl.write("""
    The dataset contains **75,757 records** with **no duplicate entries**.
    Out of 64 variables, **57 are numerical** and **7 are categorical**.
    """)

    # -------------------------------
    # Categorical Variables
    # -------------------------------
    stl.subheader("**Categorical Variables**")

    # Year Factor
    stl.subheader("**Year Factor**")
    stl.write("""
    This feature represents the year in which data samples were recorded.
    A majority of records (over 50%) belong to **Year 5 and Year 6**.

    Median Site EUI values remain relatively consistent across years.
    Values above **200 Site EUI** appear to be outliers.
    The test dataset contains data from only a single year.
    """)
    stl.image("./EDA_Plots/Year_Factor_count_plot.jpg", caption="Year Factor – Training Dataset")
    stl.image("./EDA_Plots/Year_Factor_test_count_plot.jpg", caption="Year Factor – Test Dataset")
    stl.image("EDA_Plots/Year_Factor_vs_eui.jpg", caption="Year Factor vs Site EUI")
    stl.image("EDA_Plots/year_distribution.jpg", caption="Year-wise Site EUI Distribution")
    stl.write("---")

    # State Factor
    stl.subheader("**State Factor**")
    stl.write("""
    The dataset contains data from **7 unique states**.
    Most training samples belong to **State 6**, while the test dataset 
    contains no samples from this state. The majority of test samples belong to **State 11**.

    State 6 exhibits the **highest median Site EUI**.
    """)
    stl.image("./EDA_Plots/State_Factor_count_plot.jpg", caption="State Factor – Training Dataset")
    stl.image("./EDA_Plots/State_Factor_test_count_plot.jpg", caption="State Factor – Test Dataset")
    stl.image("EDA_Plots/State_Factor_vs_eui.jpg", caption="State Factor vs Site EUI")
    stl.image("EDA_Plots/state_distribution.jpg", caption="State-wise Site EUI Distribution")

    # Building Class
    stl.subheader("**Building Class**")
    stl.write("""
    Buildings are broadly categorized into **Residential** and **Commercial**.
    Median Site EUI values are surprisingly similar across both classes.
    This feature should be analyzed alongside **Facility Type** for deeper insights.

    The dataset contains more residential building samples than commercial ones.
    """)
    stl.image("./EDA_Plots/building_class_count_plot.jpg", caption="Building Class – Training Dataset")
    stl.image("./EDA_Plots/building_class__test_count_plot.jpg", caption="Building Class – Test Dataset")
    stl.image("EDA_Plots/building_class_vs_eui.jpg", caption="Building Class vs Site EUI")
    stl.image("EDA_Plots/building_distribution.jpg", caption="Building Class-wise Site EUI Distribution")

    # Facility Type
    stl.subheader("**Facility Type**")
    stl.write("""
    Among commercial buildings, **Data Centers** exhibit the highest median Site EUI.
    For residential buildings, **Mixed-use facilities** show the highest median Site EUI.
    """)
    stl.image("./EDA_Plots/facility_count.jpg", caption="Facility Type Distribution")
    stl.image("EDA_Plots/top_res_eui.jpg", caption="Residential Facilities with Highest Site EUI")
    stl.image("EDA_Plots/top_com_eui.jpg", caption="Commercial Facilities with Highest Site EUI")

    # -------------------------------
    # Numerical Variables
    # -------------------------------
    stl.subheader("**Numerical Variables**")

    # Floor Area
    stl.subheader("**Floor Area**")
    stl.write("""
    The dataset includes buildings with floor areas ranging from **900 sq. ft.**
    to over **63,000 sq. ft.** As expected, buildings with larger floor areas
    tend to have higher median Site EUI values.
    """)
    stl.image("EDA_Plots/floor_area.jpg", caption="Floor Area vs Site EUI")
    stl.image("EDA_Plots/floor_area_build.jpg", caption="Floor Area vs Site EUI by Building Class")

    # Year Built
    stl.subheader("**Year Built**")
    stl.write("""
    Newer buildings generally exhibit lower median Site EUI values compared 
    to older buildings, which aligns with expectations regarding energy-efficient designs.
    """)
    stl.image("EDA_Plots/year_built.jpg", caption="Median Site EUI vs Year Built")

    # Energy Star Rating
    stl.subheader("**Energy Star Rating**")
    stl.write("""
    Buildings with lower Energy Star ratings tend to have higher Site EUI values,
    while higher-rated buildings demonstrate better energy efficiency.
    """)
    stl.image("EDA_Plots/energy_rating_2.jpg", caption="Site EUI vs Energy Star Rating")
    stl.image("EDA_Plots/Energy_rating.jpg", caption="Median Site EUI vs Energy Star Rating")
    stl.image("EDA_Plots/Energy_rating3.jpg", caption="Energy Star Rating vs Floor Area")
    stl.image("EDA_Plots/energy_rating_vs_building.jpg", caption="Energy Star Rating vs Building Class")

    # Elevation
    stl.subheader("**Elevation**")
    stl.image("EDA_Plots/elevation.jpg", caption="Elevation vs Site EUI")

    # Temperature
    stl.subheader("**Temperature Analysis**")
    stl.write("""
    State 4 exhibits the lowest minimum temperatures and also shows high median Site EUI.
    In contrast, State 11 maintains more moderate temperatures (40°F–80°F),
    resulting in the lowest median Site EUI.
    """)
    stl.image("EDA_Plots/statewise_min_temp.jpg", caption="Minimum Temperature Across States")
    stl.image("EDA_Plots/statewise_max_temp.jpg", caption="Maximum Temperature Across States")

    # HDD & CDD
    stl.subheader("**Heating Degree Days & Cooling Degree Days**")
    stl.image("EDA_Plots/hdd_cdd.jpg", caption="Heating & Cooling Degree Days")
    stl.image("EDA_Plots/cdd_eui.jpg", caption="Median Cooling Degree Days vs Site EUI")
    stl.image("EDA_Plots/hdd_eui.jpg", caption="Median Heating Degree Days vs Site EUI")

    # Precipitation & Snow
    stl.subheader("**Precipitation and Snowfall**")
    stl.image("EDA_Plots/snow.jpg", caption="Precipitation vs Median Site EUI")
    stl.image("EDA_Plots/preci.jpg", caption="Snowfall vs Median Site EUI")

    # Correlation
    stl.subheader("**Feature Correlation Analysis**")
    stl.image("EDA_Plots/corr_heatmap.jpg", caption="Correlation Heatmap")
    stl.image("EDA_Plots/corr_heatmap2.jpg", caption="Correlation (Site EUI < 200)")
    stl.image("EDA_Plots/corr_heatmap3.jpg", caption="Correlation (Site EUI > 200)")

    # -------------------------------
    # Conclusions
    # -------------------------------
    stl.header("**Conclusions**")
    stl.write("""
    - Six features contain missing values.
    - Several numerical features show strong correlations.
    - Seasonal temperature variables are highly correlated.
    - Additional distribution plots for numerical features can further enhance analysis.
    """)


if __name__ == "__main__":
    main()
