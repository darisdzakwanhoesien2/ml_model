import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


st.set_page_config(page_title="Universal ML Trainer", layout="wide")

st.title("🚀 Universal Machine Learning Trainer")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)

    target = st.selectbox("Select Target Column", df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        # Detect problem type
        if y.dtype == "object" or len(y.unique()) < 15:
            task_type = "classification"
        else:
            task_type = "regression"

        st.success(f"Detected Task: {task_type.upper()}")

        # Split
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Preprocessing
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )

        st.subheader("🤖 Select Models")

        if task_type == "classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "SVM": SVC(),
                "KNN": KNeighborsClassifier()
            }
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor()
            }

        selected_models = st.multiselect(
            "Choose Models to Train",
            list(models.keys())
        )

        if st.button("Train Models") and selected_models:

            results = []

            for model_name in selected_models:
                model = models[model_name]

                clf = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("model", model)
                ])

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                if task_type == "classification":
                    acc = accuracy_score(y_test, y_pred)
                    results.append((model_name, acc))
                else:
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    results.append((model_name, rmse, r2))

            st.subheader("📈 Model Comparison")

            if task_type == "classification":
                results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
                st.dataframe(results_df)

                fig = px.bar(results_df, x="Model", y="Accuracy",
                             title="Model Accuracy Comparison")
                st.plotly_chart(fig, use_container_width=True)

            else:
                results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2"])
                st.dataframe(results_df)

                fig = px.bar(results_df, x="Model", y="RMSE",
                             title="Model RMSE Comparison")
                st.plotly_chart(fig, use_container_width=True)

            st.success("Training Complete 🎉")