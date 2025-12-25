import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

st.set_page_config(layout="wide")
st.header("📊 Model Benchmarking & Comparison")

st.info(
    "Multiple regression models were benchmarked during experimentation. "
    "The final deployed model is an optimized XGBoost regressor selected "
    "for performance, interpretability, and deployment stability."
)

def model_comparison():

    df = pd.read_csv("train.csv")

    X = df.drop("site_eui", axis=1)
    y = df["site_eui"]

    imputer = SimpleImputer(strategy="most_frequent")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X = X.apply(LabelEncoder().fit_transform)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "SVR": SVR(),
        "ElasticNet": ElasticNet(),
        "KNN": KNeighborsRegressor(),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        results.append({"Model": name, "MSE": mse})

    df_results = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Model", y="MSE", data=df_results, ax=ax)
    ax.set_title("Model Performance Comparison")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Results Summary")
    st.dataframe(df_results.sort_values("MSE"))

def main():
    model_comparison()

if __name__ == "__main__":
    main()
