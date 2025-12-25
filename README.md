# 🔋 End-to-End Energy Intensity Forecasting System
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/Math-NumPy-013243?style=flat-square&logo=numpy)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-f7931e?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Boosting-XGBoost-ff6600?style=flat-square)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-ff4b4b?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Type-Machine%20Learning-success?style=flat-square)](#)


An end-to-end **machine learning and data analytics project** that predicts **Site Energy Use Intensity (Site EUI)** of buildings using building characteristics and climate data.  
This project covers the **complete ML lifecycle** — from exploratory data analysis and feature engineering to model training, evaluation, and deployment via an interactive **Streamlit web application**.

---

## 📌 Problem Statement

Buildings are responsible for a significant share of global energy consumption and carbon emissions.  
Accurately predicting building energy usage enables:

- Identification of inefficient buildings  
- Data-driven retrofitting strategies  
- Better planning for energy efficiency and sustainability  

The goal of this project is to **predict Site Energy Use Intensity (EUI)** — a standardized metric representing how much energy a building consumes — using **building attributes and weather-related variables**.

---

## 📊 Dataset

- **Source:** WiDS Datathon 2022 (Kaggle)  
- **Size:** ~75,000 building records  
- **Target Variable:** `site_eui`  

### Feature Categories
- **Building Characteristics:** floor area, year built, energy star rating, elevation  
- **Climate Variables:** temperature statistics, heating & cooling degree days, precipitation, snowfall  
- **Regional & Temporal Indicators:** anonymized state and year factors  

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand the structure and behavior of the dataset:

- Distribution analysis of numerical and categorical features  
- Identification of missing values and outliers  
- Correlation analysis between climate variables and Site EUI  
- Detection of strong multicollinearity among seasonal temperature features  
- Visualization of energy usage patterns across regions and building types  

Insights from EDA guided **feature selection, preprocessing, and modeling decisions**.

---

## 🧠 Data Preprocessing & Feature Engineering

### Handling Missing Values
- Removed features with excessive missing values  
- Applied statistical and model-based imputation techniques  
- Used KNN-based imputation where appropriate  

### Feature Selection
- Dropped high-dimensional and noisy categorical features  
- Removed anonymized features not suitable for inference  
- Eliminated redundant and weakly informative variables  

### Multicollinearity Reduction
- Used **Variance Inflation Factor (VIF)** analysis  
- Aggregated highly correlated seasonal temperature features into:
  - Average winter temperature metrics  
  - Average summer temperature metrics  
- Reduced feature dimensionality while preserving predictive power  

---

## 🤖 Modeling & Evaluation

Multiple regression models were implemented and evaluated:

### Models Trained
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- Support Vector Regressor (SVR)  
- K-Nearest Neighbors Regressor  
- **AutoGluon Tabular (benchmark model)**  

### Evaluation Metric
- **Mean Squared Error (MSE)** on a held-out test set  

Tree-based ensemble models consistently outperformed linear baselines, highlighting the non-linear relationship between climate conditions and building energy usage.

---

## 🚀 Deployment (Streamlit Application)

The final trained model is deployed using **Streamlit**, enabling:

- Interactive user input for building and climate parameters  
- Real-time prediction of Site Energy Use Intensity  
- A clean and intuitive UI suitable for non-technical users  

This transforms the project from a static analysis into a **fully usable end-to-end ML system**.

---

## 🗂 Project Structure

```text
.
├── Home.py                     # Main Streamlit application
├── pages/                      # EDA, Feature Engineering, and Modeling pages
├── model_v2.joblib             # Trained machine learning model
├── train.csv                   # Training dataset
├── Images/                     # UI assets
├── .streamlit/                 # Streamlit configuration
├── requirements.txt            # Python dependencies

```
## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/siddhantchasta/energy-intensity-forecasting-system.git
cd yourProjectFolder
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run Home.py
```

The application will launch locally in your browser.

## 🎯 Project Highlights

✔ Complete end-to-end ML pipeline

✔ Real-world dataset with practical relevance

✔ Strong emphasis on EDA and feature engineering

✔ Deployed interactive prediction system
