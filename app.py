import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, roc_curve, auc
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Universal ML Research Dashboard", layout="wide")
st.title("🚀 Universal ML Research Dashboard")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    tab1, tab2, tab3 = st.tabs(["📊 Data", "🤖 Modeling", "🧠 Clustering & 2D"])

    # ==========================================================
    # TAB 1 — DATA
    # ==========================================================
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)
        st.write("Missing Values")
        st.write(df.isnull().sum())

    target = st.selectbox("Select Target Column", df.columns)

    if target:

        X = df.drop(columns=[target])
        y = df[target]

        if y.dtype == "object" or len(y.unique()) < 15:
            task_type = "classification"
        else:
            task_type = "regression"

        st.success(f"Detected Task: {task_type.upper()}")

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # ==========================================================
        # PREPROCESSING
        # ==========================================================
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

        # ==========================================================
        # TAB 2 — MODELING
        # ==========================================================
        with tab2:

            if task_type == "classification":
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(),
                    "SVM": SVC(probability=True),
                    "KNN": KNeighborsClassifier()
                }
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(),
                    "SVR": SVR(),
                    "KNN": KNeighborsRegressor()
                }

            selected_models = st.multiselect("Select Models", list(models.keys()))

            if st.button("Train Models") and selected_models:

                for model_name in selected_models:

                    st.markdown("---")
                    st.subheader(model_name)

                    model = models[model_name]

                    clf = Pipeline([
                        ("preprocessor", preprocessor),
                        ("model", model)
                    ])

                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    # ==============================
                    # CLASSIFICATION
                    # ==============================
                    if task_type == "classification":

                        acc = accuracy_score(y_test, y_pred)
                        st.write("Accuracy:", round(acc, 4))
                        st.text(classification_report(y_test, y_pred))

                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_title("Confusion Matrix")
                        st.pyplot(fig_cm)

                        if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
                            y_proba = clf.predict_proba(X_test)[:, 1]
                            fpr, tpr, _ = roc_curve(y_test, y_proba)
                            roc_auc = auc(fpr, tpr)

                            fig_roc, ax = plt.subplots()
                            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                            ax.plot([0, 1], [0, 1], "--")
                            ax.legend()
                            ax.set_title("ROC Curve")
                            st.pyplot(fig_roc)

                    # ==============================
                    # REGRESSION
                    # ==============================
                    else:
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        r2 = r2_score(y_test, y_pred)

                        st.write("RMSE:", round(rmse, 4))
                        st.write("R2:", round(r2, 4))

                        fig_reg, ax = plt.subplots()
                        ax.scatter(y_test, y_pred)
                        ax.set_title("Actual vs Predicted")
                        st.pyplot(fig_reg)

                    # ==============================
                    # FEATURE IMPORTANCE
                    # ==============================
                    if hasattr(model, "feature_importances_"):

                        feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
                        importances = clf.named_steps["model"].feature_importances_

                        fi_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Importance": importances
                        }).sort_values("Importance", ascending=False).head(20)

                        fig_fi = px.bar(
                            fi_df,
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            title="Top 20 Features"
                        )
                        st.plotly_chart(fig_fi, use_container_width=True)

                    # ==============================
                    # SAFE SHAP
                    # ==============================
                    st.subheader("SHAP Explainability")

                    try:
                        model_obj = clf.named_steps["model"]
                        X_test_transformed = clf.named_steps["preprocessor"].transform(X_test)
                        feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
                        X_sample = X_test_transformed[:200]

                        if isinstance(model_obj, (RandomForestClassifier, RandomForestRegressor)):
                            explainer = shap.TreeExplainer(model_obj)
                            shap_values = explainer.shap_values(X_sample)

                            fig, ax = plt.subplots()
                            shap.summary_plot(
                                shap_values,
                                X_sample,
                                feature_names=feature_names,
                                show=False
                            )
                            st.pyplot(fig)

                        elif isinstance(model_obj, (LogisticRegression, LinearRegression)):
                            explainer = shap.LinearExplainer(model_obj, X_sample)
                            shap_values = explainer.shap_values(X_sample)

                            fig, ax = plt.subplots()
                            shap.summary_plot(
                                shap_values,
                                X_sample,
                                feature_names=feature_names,
                                show=False
                            )
                            st.pyplot(fig)

                        else:
                            st.info("SHAP not supported for this model type.")

                    except Exception as e:
                        st.warning(f"SHAP skipped: {e}")

    # ==========================================================
    # TAB 3 — CLUSTERING & 2D
    # ==========================================================
    with tab3:

        st.subheader("2D PCA Projection + KMeans Clustering")

        X_processed = preprocessor.fit_transform(X)

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_processed)

        n_clusters = st.slider("Number of Clusters", 2, 10, 3)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_2d)

        plot_df = pd.DataFrame({
            "PC1": X_2d[:, 0],
            "PC2": X_2d[:, 1],
            "Cluster": cluster_labels.astype(str)
        })

        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            title="PCA 2D Clusters"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Decision Region
        st.subheader("Cluster Decision Regions")

        h = 0.05
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )

        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig_region, ax = plt.subplots()
        ax.contourf(xx, yy, Z, alpha=0.3)
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, edgecolor="k")
        ax.set_title("Cluster Regions (2D)")
        st.pyplot(fig_region)

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import shap

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import (
#     accuracy_score, classification_report, confusion_matrix,
#     mean_squared_error, r2_score, roc_curve, auc
# )

# # Classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier

# # Regression
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor


# # ==========================================
# # PAGE CONFIG
# # ==========================================
# st.set_page_config(page_title="Universal ML Trainer", layout="wide")
# st.title("🚀 Universal ML Trainer (Stable + Explainable)")

# # ==========================================
# # FILE UPLOAD
# # ==========================================
# uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# if uploaded_file:

#     df = pd.read_csv(uploaded_file)

#     st.subheader("📊 Dataset Preview")
#     st.dataframe(df.head())
#     st.write("Shape:", df.shape)

#     target = st.selectbox("Select Target Column", df.columns)

#     if target:

#         X = df.drop(columns=[target])
#         y = df[target]

#         # ==========================================
#         # TASK DETECTION
#         # ==========================================
#         if y.dtype == "object" or len(y.unique()) < 15:
#             task_type = "classification"
#         else:
#             task_type = "regression"

#         st.success(f"Detected Task: {task_type.upper()}")

#         test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, random_state=42
#         )

#         # ==========================================
#         # PREPROCESSING
#         # ==========================================
#         numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
#         categorical_features = X.select_dtypes(include=["object"]).columns

#         numeric_transformer = Pipeline([
#             ("imputer", SimpleImputer(strategy="median")),
#             ("scaler", StandardScaler())
#         ])

#         categorical_transformer = Pipeline([
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             ("encoder", OneHotEncoder(handle_unknown="ignore"))
#         ])

#         preprocessor = ColumnTransformer([
#             ("num", numeric_transformer, numeric_features),
#             ("cat", categorical_transformer, categorical_features)
#         ])

#         # ==========================================
#         # MODEL SELECTION
#         # ==========================================
#         if task_type == "classification":
#             models = {
#                 "Logistic Regression": LogisticRegression(max_iter=1000),
#                 "Random Forest": RandomForestClassifier(),
#                 "SVM": SVC(probability=True),
#                 "KNN": KNeighborsClassifier()
#             }
#         else:
#             models = {
#                 "Linear Regression": LinearRegression(),
#                 "Random Forest": RandomForestRegressor(),
#                 "SVR": SVR(),
#                 "KNN": KNeighborsRegressor()
#             }

#         selected_models = st.multiselect("Select Models", list(models.keys()))

#         # ==========================================
#         # TRAIN BUTTON
#         # ==========================================
#         if st.button("Train Models") and selected_models:

#             for model_name in selected_models:

#                 st.markdown("---")
#                 st.subheader(f"📌 {model_name}")

#                 model = models[model_name]

#                 clf = Pipeline([
#                     ("preprocessor", preprocessor),
#                     ("model", model)
#                 ])

#                 clf.fit(X_train, y_train)
#                 y_pred = clf.predict(X_test)

#                 # ==========================================
#                 # CLASSIFICATION
#                 # ==========================================
#                 if task_type == "classification":

#                     acc = accuracy_score(y_test, y_pred)
#                     st.write("Accuracy:", round(acc, 4))

#                     st.text("Classification Report")
#                     st.text(classification_report(y_test, y_pred))

#                     # Confusion Matrix
#                     cm = confusion_matrix(y_test, y_pred)

#                     fig_cm, ax = plt.subplots()
#                     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
#                     ax.set_xlabel("Predicted")
#                     ax.set_ylabel("Actual")
#                     ax.set_title("Confusion Matrix")
#                     st.pyplot(fig_cm)

#                     # ROC Curve (Binary only)
#                     if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
#                         y_proba = clf.predict_proba(X_test)[:, 1]
#                         fpr, tpr, _ = roc_curve(y_test, y_proba)
#                         roc_auc = auc(fpr, tpr)

#                         fig_roc, ax = plt.subplots()
#                         ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
#                         ax.plot([0, 1], [0, 1], "--")
#                         ax.set_xlabel("False Positive Rate")
#                         ax.set_ylabel("True Positive Rate")
#                         ax.set_title("ROC Curve")
#                         ax.legend()
#                         st.pyplot(fig_roc)

#                 # ==========================================
#                 # REGRESSION
#                 # ==========================================
#                 else:

#                     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#                     r2 = r2_score(y_test, y_pred)

#                     st.write("RMSE:", round(rmse, 4))
#                     st.write("R2:", round(r2, 4))

#                     fig_reg, ax = plt.subplots()
#                     ax.scatter(y_test, y_pred)
#                     ax.set_xlabel("Actual")
#                     ax.set_ylabel("Predicted")
#                     ax.set_title("Actual vs Predicted")
#                     st.pyplot(fig_reg)

#                 # ==========================================
#                 # FEATURE IMPORTANCE (Tree Models)
#                 # ==========================================
#                 if hasattr(model, "feature_importances_"):

#                     st.subheader("🌲 Feature Importance")

#                     feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
#                     importances = clf.named_steps["model"].feature_importances_

#                     fi_df = pd.DataFrame({
#                         "Feature": feature_names,
#                         "Importance": importances
#                     }).sort_values("Importance", ascending=False).head(20)

#                     fig_fi = px.bar(
#                         fi_df,
#                         x="Importance",
#                         y="Feature",
#                         orientation="h",
#                         title="Top 20 Features"
#                     )

#                     st.plotly_chart(fig_fi, use_container_width=True)

#                 # ==========================================
#                 # SAFE SHAP
#                 # ==========================================
#                 st.subheader("🧠 SHAP Explainability")

#                 try:

#                     model_obj = clf.named_steps["model"]
#                     X_test_transformed = clf.named_steps["preprocessor"].transform(X_test)
#                     feature_names = clf.named_steps["preprocessor"].get_feature_names_out()

#                     # Reduce size for speed
#                     X_sample = X_test_transformed[:200]

#                     # Random Forest (supports multiclass)
#                     if isinstance(model_obj, (RandomForestClassifier, RandomForestRegressor)):

#                         explainer = shap.TreeExplainer(model_obj)
#                         shap_values = explainer.shap_values(X_sample)

#                         fig, ax = plt.subplots()
#                         shap.summary_plot(
#                             shap_values,
#                             X_sample,
#                             feature_names=feature_names,
#                             show=False
#                         )
#                         st.pyplot(fig)

#                     # Logistic Regression / Linear
#                     elif isinstance(model_obj, (LogisticRegression, LinearRegression)):

#                         explainer = shap.LinearExplainer(model_obj, X_sample)
#                         shap_values = explainer.shap_values(X_sample)

#                         fig, ax = plt.subplots()
#                         shap.summary_plot(
#                             shap_values,
#                             X_sample,
#                             feature_names=feature_names,
#                             show=False
#                         )
#                         st.pyplot(fig)

#                     else:
#                         st.info("SHAP not supported for this model type.")

#                 except Exception as e:
#                     st.warning(f"SHAP skipped due to compatibility issue: {e}")

#             st.success("Training Completed Successfully 🎉")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import plotly.express as px
# import seaborn as sns
# import shap

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import (
#     accuracy_score, classification_report, confusion_matrix,
#     mean_squared_error, r2_score, roc_curve, auc
# )

# # Classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier

# # Regression
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor


# # ---------------------------------
# # PAGE CONFIG
# # ---------------------------------
# st.set_page_config(page_title="Universal ML Trainer", layout="wide")
# st.title("🚀 Universal ML Trainer with Explainability")

# # ---------------------------------
# # FILE UPLOAD
# # ---------------------------------
# uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)

#     st.subheader("📊 Dataset Preview")
#     st.dataframe(df.head())
#     st.write("Shape:", df.shape)

#     target = st.selectbox("Select Target Column", df.columns)

#     if target:
#         X = df.drop(columns=[target])
#         y = df[target]

#         # Task Detection
#         if y.dtype == "object" or len(y.unique()) < 15:
#             task_type = "classification"
#         else:
#             task_type = "regression"

#         st.success(f"Detected Task: {task_type.upper()}")

#         test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, random_state=42
#         )

#         # ---------------------------------
#         # PREPROCESSING
#         # ---------------------------------
#         numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
#         categorical_features = X.select_dtypes(include=["object"]).columns

#         numeric_transformer = Pipeline([
#             ("imputer", SimpleImputer(strategy="median")),
#             ("scaler", StandardScaler())
#         ])

#         categorical_transformer = Pipeline([
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             ("encoder", OneHotEncoder(handle_unknown="ignore"))
#         ])

#         preprocessor = ColumnTransformer([
#             ("num", numeric_transformer, numeric_features),
#             ("cat", categorical_transformer, categorical_features)
#         ])

#         # ---------------------------------
#         # MODELS
#         # ---------------------------------
#         if task_type == "classification":
#             models = {
#                 "Logistic Regression": LogisticRegression(max_iter=1000),
#                 "Random Forest": RandomForestClassifier(),
#                 "Gradient Boosting": GradientBoostingClassifier(),
#                 "SVM": SVC(probability=True),
#                 "KNN": KNeighborsClassifier()
#             }
#         else:
#             models = {
#                 "Linear Regression": LinearRegression(),
#                 "Random Forest": RandomForestRegressor(),
#                 "Gradient Boosting": GradientBoostingRegressor(),
#                 "SVR": SVR(),
#                 "KNN": KNeighborsRegressor()
#             }

#         selected_models = st.multiselect("Select Models", list(models.keys()))

#         # ---------------------------------
#         # TRAIN
#         # ---------------------------------
#         if st.button("Train Models") and selected_models:

#             for model_name in selected_models:

#                 model = models[model_name]

#                 clf = Pipeline([
#                     ("preprocessor", preprocessor),
#                     ("model", model)
#                 ])

#                 clf.fit(X_train, y_train)
#                 y_pred = clf.predict(X_test)

#                 st.markdown("---")
#                 st.subheader(f"📌 {model_name}")

#                 # ==============================
#                 # CLASSIFICATION
#                 # ==============================
#                 if task_type == "classification":

#                     acc = accuracy_score(y_test, y_pred)
#                     st.write("Accuracy:", round(acc, 4))

#                     st.text("Classification Report")
#                     st.text(classification_report(y_test, y_pred))

#                     # Confusion Matrix
#                     cm = confusion_matrix(y_test, y_pred)

#                     fig_cm, ax = plt.subplots()
#                     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
#                     ax.set_xlabel("Predicted")
#                     ax.set_ylabel("Actual")
#                     ax.set_title("Confusion Matrix (TP/TN/FP/FN)")
#                     st.pyplot(fig_cm)

#                     # ROC Curve (Binary only)
#                     if len(np.unique(y_test)) == 2:
#                         y_proba = clf.predict_proba(X_test)[:, 1]
#                         fpr, tpr, _ = roc_curve(y_test, y_proba)
#                         roc_auc = auc(fpr, tpr)

#                         fig_roc, ax = plt.subplots()
#                         ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
#                         ax.plot([0,1], [0,1], "--")
#                         ax.set_xlabel("False Positive Rate")
#                         ax.set_ylabel("True Positive Rate")
#                         ax.set_title("ROC Curve")
#                         ax.legend()
#                         st.pyplot(fig_roc)

#                 # ==============================
#                 # REGRESSION
#                 # ==============================
#                 else:

#                     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#                     r2 = r2_score(y_test, y_pred)

#                     st.write("RMSE:", round(rmse, 4))
#                     st.write("R2:", round(r2, 4))

#                     fig_reg, ax = plt.subplots()
#                     ax.scatter(y_test, y_pred)
#                     ax.set_xlabel("Actual")
#                     ax.set_ylabel("Predicted")
#                     ax.set_title("Actual vs Predicted")
#                     st.pyplot(fig_reg)

#                 # ==============================
#                 # FEATURE IMPORTANCE
#                 # ==============================
#                 if hasattr(model, "feature_importances_"):

#                     st.subheader("🌲 Feature Importance")

#                     feature_names = clf.named_steps["preprocessor"].get_feature_names_out()
#                     importances = clf.named_steps["model"].feature_importances_

#                     fi_df = pd.DataFrame({
#                         "Feature": feature_names,
#                         "Importance": importances
#                     }).sort_values("Importance", ascending=False).head(20)

#                     fig_fi = px.bar(
#                         fi_df,
#                         x="Importance",
#                         y="Feature",
#                         orientation="h",
#                         title="Top 20 Important Features"
#                     )
#                     st.plotly_chart(fig_fi, use_container_width=True)

#                 # ==============================
#                 # SHAP EXPLAINABILITY
#                 # ==============================
#                 if model_name in ["Random Forest", "Gradient Boosting"]:

#                     st.subheader("🧠 SHAP Explainability")

#                     X_test_transformed = clf.named_steps["preprocessor"].transform(X_test)

#                     explainer = shap.TreeExplainer(clf.named_steps["model"])
#                     shap_values = explainer.shap_values(X_test_transformed)

#                     feature_names = clf.named_steps["preprocessor"].get_feature_names_out()

#                     # Summary Plot
#                     st.write("SHAP Summary Plot")
#                     fig_shap, ax = plt.subplots()
#                     shap.summary_plot(
#                         shap_values,
#                         X_test_transformed,
#                         feature_names=feature_names,
#                         show=False
#                     )
#                     st.pyplot(fig_shap)

#                     # Waterfall (first sample)
#                     st.write("SHAP Waterfall (First Sample)")
#                     fig_force = shap.force_plot(
#                         explainer.expected_value,
#                         shap_values[0],
#                         feature_names=feature_names,
#                         matplotlib=True
#                     )
#                     st.pyplot(bbox_inches='tight')

#             st.success("Training Completed 🎉")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import plotly.express as px

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import (
#     accuracy_score, classification_report, confusion_matrix,
#     mean_squared_error, r2_score
# )

# # Classification Models
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier

# # Regression Models
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor


# st.set_page_config(page_title="Universal ML Trainer", layout="wide")

# st.title("🚀 Universal Machine Learning Trainer")

# # Upload CSV
# uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)

#     st.subheader("📊 Dataset Preview")
#     st.dataframe(df.head())

#     st.write("Shape:", df.shape)

#     target = st.selectbox("Select Target Column", df.columns)

#     if target:
#         X = df.drop(columns=[target])
#         y = df[target]

#         # Detect problem type
#         if y.dtype == "object" or len(y.unique()) < 15:
#             task_type = "classification"
#         else:
#             task_type = "regression"

#         st.success(f"Detected Task: {task_type.upper()}")

#         # Split
#         test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, random_state=42
#         )

#         # Preprocessing
#         numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
#         categorical_features = X.select_dtypes(include=["object"]).columns

#         numeric_transformer = Pipeline(steps=[
#             ("imputer", SimpleImputer(strategy="median")),
#             ("scaler", StandardScaler())
#         ])

#         categorical_transformer = Pipeline(steps=[
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             ("encoder", OneHotEncoder(handle_unknown="ignore"))
#         ])

#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ("num", numeric_transformer, numeric_features),
#                 ("cat", categorical_transformer, categorical_features)
#             ]
#         )

#         st.subheader("🤖 Select Models")

#         if task_type == "classification":
#             models = {
#                 "Logistic Regression": LogisticRegression(max_iter=1000),
#                 "Random Forest": RandomForestClassifier(),
#                 "Gradient Boosting": GradientBoostingClassifier(),
#                 "SVM": SVC(),
#                 "KNN": KNeighborsClassifier()
#             }
#         else:
#             models = {
#                 "Linear Regression": LinearRegression(),
#                 "Random Forest": RandomForestRegressor(),
#                 "Gradient Boosting": GradientBoostingRegressor(),
#                 "SVR": SVR(),
#                 "KNN": KNeighborsRegressor()
#             }

#         selected_models = st.multiselect(
#             "Choose Models to Train",
#             list(models.keys())
#         )

#         if st.button("Train Models") and selected_models:

#             results = []

#             for model_name in selected_models:
#                 model = models[model_name]

#                 clf = Pipeline(steps=[
#                     ("preprocessor", preprocessor),
#                     ("model", model)
#                 ])

#                 clf.fit(X_train, y_train)
#                 y_pred = clf.predict(X_test)

#                 if task_type == "classification":
#                     acc = accuracy_score(y_test, y_pred)
#                     results.append((model_name, acc))
#                 else:
#                     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#                     r2 = r2_score(y_test, y_pred)
#                     results.append((model_name, rmse, r2))

#             st.subheader("📈 Model Comparison")

#             if task_type == "classification":
#                 results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
#                 st.dataframe(results_df)

#                 fig = px.bar(results_df, x="Model", y="Accuracy",
#                              title="Model Accuracy Comparison")
#                 st.plotly_chart(fig, use_container_width=True)

#             else:
#                 results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2"])
#                 st.dataframe(results_df)

#                 fig = px.bar(results_df, x="Model", y="RMSE",
#                              title="Model RMSE Comparison")
#                 st.plotly_chart(fig, use_container_width=True)

#             st.success("Training Complete 🎉")