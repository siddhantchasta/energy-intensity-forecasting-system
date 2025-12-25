import streamlit as stl
stl.set_page_config(layout="wide")

# -------------------------------
# Page Header
# -------------------------------
stl.header("Data Preprocessing & Feature Engineering")

# -------------------------------
# EDA Recap
# -------------------------------
stl.subheader("EDA Recap")

stl.write("""
Key conclusions derived from Exploratory Data Analysis (EDA):

- Six features contain missing values.
- The **Elevation** feature shows a strong correlation with **Site EUI**.
- The **Days with Fog** feature contains missing values and requires imputation using **KNN Imputer**.
- The remaining features with missing values can be handled using **mode-based imputation**, as most values are equal to 1.
- The **Facility Type** feature is removed due to high cardinality, which can introduce noise into the model.
- The **Days Above 110°F** feature is also removed due to limited predictive contribution.
""")

# -------------------------------
# Handling Missing Values
# -------------------------------
stl.subheader("Handling Missing Values")

stl.write("Percentage of missing values in the dataset:")
stl.image("EDA_Plots/Missing_values.jpg")

stl.write("""
The first step is to remove features with **significant missing values**
(i.e., more than 50% missing data), followed by appropriate imputation techniques
for the remaining features.

Although **KNN Imputer** can be used to fill missing numerical values by observing
patterns in related features, applying it to features with over 50% missing values
would introduce excessive noise into the dataset. Therefore, removing such features
is the most reliable approach.
""")

# -------------------------------
# Filling Missing Values
# -------------------------------
stl.subheader("Filling Missing Values")

stl.write("""
Missing values in selected categorical and ordinal features are filled using
**predictive modeling**. A simple regression model is trained on the remaining
features to estimate missing values.

Using this approach, missing values in **year_built** and **energy_star_rating**
are imputed.
""")

# -------------------------------
# Removing Noisy Features
# -------------------------------
stl.write("Removing Noisy Features")

stl.write("""
The following features are removed during preprocessing:

- **Year Factor**
- **Facility Type**
- **Days Above 110°F**

**Facility Type** is a high-cardinality categorical feature that can introduce
noise into the model.

All training samples have the same **Year Factor** value, while the test dataset
does not include this value. Additionally, Year Factor is an anonymized feature
that cannot be mapped back to real-world information. As a result, user input for
this feature is not meaningful, and it is excluded from the model.
""")

# -------------------------------
# Feature Engineering
# -------------------------------
stl.subheader("Feature Engineering")

stl.subheader("Handling Multicollinearity")

stl.write("""
The first step in handling multicollinearity is identifying highly correlated
features. This is achieved using the **Variance Inflation Factor (VIF)**, where
features with a VIF value greater than **4** are considered problematic.

Seasonal temperature features exhibit strong correlations with each other.
Aggregating these features helps reduce multicollinearity and also lowers the
overall feature count, making the prediction application more user-friendly.
""")

stl.write("""
The following aggregated features are created using mean aggregation:

- **Avg_min_temp_winter**  
  (January–April, October–December minimum temperatures)

- **Avg_max_temp_winter**  
  (January–April, October–December maximum temperatures)

- **Avg_temp_winter**  
  (January–April, October–December average temperatures)

- **Avg_min_temp_summer**  
  (May–September minimum temperatures)

- **Avg_max_temp_summer**  
  (May–September maximum temperatures)

- **Avg_temp_summer**  
  (May–September average temperatures)

- **Avg_days_below_30F**  
  (Days below 30°F, 20°F, 10°F, and 0°F)
""")

stl.write("At the end of feature engineering, the following features are used to train the model:")
stl.write("(Add final feature list here)")
