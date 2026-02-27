import streamlit as st
import os
import subprocess
import uuid
import sys
from datetime import datetime

BASE_DIR = os.path.abspath("experiments")
os.makedirs(BASE_DIR, exist_ok=True)


def create_experiment_folder():
    exp_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    exp_path = os.path.join(BASE_DIR, exp_id)
    os.makedirs(exp_path, exist_ok=True)
    return exp_id, exp_path


def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def run_script(script_filename, working_dir):
    result = subprocess.run(
        [sys.executable, script_filename],
        cwd=working_dir,
        capture_output=True,
        text=True
    )
    return result.stdout, result.stderr


st.title("🚀 Upload & Run")

train_file = st.file_uploader("Upload train.csv", type="csv")
test_file = st.file_uploader("Upload test.csv", type="csv")
sample_file = st.file_uploader("Upload sample_submission.csv", type="csv")
script_file = st.file_uploader("Upload Python model script (.py)", type="py")

if st.button("Run Experiment"):

    if not all([train_file, test_file, sample_file, script_file]):
        st.error("Please upload all required files.")
    else:
        exp_id, exp_path = create_experiment_folder()

        save_uploaded_file(train_file, os.path.join(exp_path, "train.csv"))
        save_uploaded_file(test_file, os.path.join(exp_path, "test.csv"))
        save_uploaded_file(sample_file, os.path.join(exp_path, "sample_submission.csv"))
        save_uploaded_file(script_file, os.path.join(exp_path, "script.py"))

        stdout, stderr = run_script("script.py", exp_path)

        st.subheader("Output")
        st.code(stdout)

        if stderr:
            st.error(stderr)

        submission_path = os.path.join(exp_path, "submission.csv")

        if os.path.exists(submission_path):
            with open(submission_path, "rb") as f:
                st.download_button(
                    "Download Submission",
                    f,
                    file_name="submission.csv",
                    mime="text/csv"
                )

# import streamlit as st
# import os
# import pandas as pd
# import subprocess
# import uuid
# import shutil
# import sys
# from datetime import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns
# import json

# # ==========================================
# # CONFIG
# # ==========================================

# st.set_page_config(page_title="ML Experiment Manager", layout="wide")

# BASE_DIR = os.path.abspath("experiments")
# os.makedirs(BASE_DIR, exist_ok=True)

# # ==========================================
# # UTILITIES
# # ==========================================

# def create_experiment_folder():
#     exp_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#     exp_path = os.path.join(BASE_DIR, exp_id)
#     os.makedirs(exp_path, exist_ok=True)
#     return exp_id, exp_path


# def save_uploaded_file(uploaded_file, save_path):
#     with open(save_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())


# def run_script(script_filename, working_dir):
#     """
#     Run script inside experiment folder safely.
#     """
#     result = subprocess.run(
#         [sys.executable, script_filename],
#         cwd=working_dir,
#         capture_output=True,
#         text=True
#     )
#     return result.stdout, result.stderr


# def list_experiments():
#     return sorted(os.listdir(BASE_DIR), reverse=True)


# # ==========================================
# # SIDEBAR
# # ==========================================

# st.sidebar.title("📂 Navigation")
# page = st.sidebar.radio("Go to", ["🚀 Upload & Run", "📊 Analytics", "📁 History"])

# # =====================================================
# # 🚀 PAGE 1 — UPLOAD & RUN
# # =====================================================

# if page == "🚀 Upload & Run":

#     st.title("🚀 ML Experiment Runner")

#     st.write("""
#     Upload:
#     - train.csv
#     - test.csv
#     - sample_submission.csv
#     - model script (.py)
#     """)

#     train_file = st.file_uploader("Upload train.csv", type="csv")
#     test_file = st.file_uploader("Upload test.csv", type="csv")
#     sample_file = st.file_uploader("Upload sample_submission.csv", type="csv")
#     script_file = st.file_uploader("Upload Python model script (.py)", type="py")

#     if st.button("Run Experiment"):

#         if not all([train_file, test_file, sample_file, script_file]):
#             st.error("Please upload all required files.")
#         else:
#             exp_id, exp_path = create_experiment_folder()

#             # Save files
#             save_uploaded_file(train_file, os.path.join(exp_path, "train.csv"))
#             save_uploaded_file(test_file, os.path.join(exp_path, "test.csv"))
#             save_uploaded_file(sample_file, os.path.join(exp_path, "sample_submission.csv"))
#             save_uploaded_file(script_file, os.path.join(exp_path, "script.py"))

#             st.info("Running your model script...")

#             stdout, stderr = run_script("script.py", exp_path)

#             st.subheader("📜 Script Output")
#             st.code(stdout)

#             if stderr:
#                 st.subheader("❌ Errors")
#                 st.error(stderr)

#             # Debug Info
#             st.subheader("🔎 Debug Info")
#             st.write("Experiment Folder:", exp_path)
#             st.write("Files in Folder:", os.listdir(exp_path))

#             # Check submission
#             submission_path = os.path.join(exp_path, "submission.csv")

#             if os.path.exists(submission_path):
#                 st.success("✅ submission.csv generated successfully!")

#                 submission_df = pd.read_csv(submission_path)
#                 st.dataframe(submission_df.head())

#                 with open(submission_path, "rb") as f:
#                     st.download_button(
#                         label="⬇ Download Submission",
#                         data=f,
#                         file_name=f"{exp_id}_submission.csv",
#                         mime="text/csv"
#                     )

#             else:
#                 st.warning("⚠ No submission.csv found. Make sure your script saves submission.csv")


# # =====================================================
# # 📊 PAGE 2 — ANALYTICS
# # =====================================================

# elif page == "📊 Analytics":

#     st.title("📊 Dataset Analytics Dashboard")

#     experiments = list_experiments()

#     if not experiments:
#         st.warning("No experiments found.")
#     else:
#         selected_exp = st.selectbox("Select Experiment", experiments)
#         exp_path = os.path.join(BASE_DIR, selected_exp)

#         train_path = os.path.join(exp_path, "train.csv")

#         if os.path.exists(train_path):

#             df = pd.read_csv(train_path)

#             st.subheader("📌 Dataset Shape")
#             st.write(df.shape)

#             st.subheader("📊 Statistical Summary")
#             st.dataframe(df.describe())

#             st.subheader("❗ Missing Values")
#             missing = df.isnull().sum()
#             st.bar_chart(missing)

#             numeric_cols = df.select_dtypes(include="number").columns

#             if len(numeric_cols) > 0:
#                 selected_col = st.selectbox("Select Numeric Column", numeric_cols)

#                 fig, ax = plt.subplots()
#                 sns.histplot(df[selected_col], kde=True, ax=ax)
#                 st.pyplot(fig)

#         else:
#             st.warning("No train.csv found in this experiment.")


# # =====================================================
# # 📁 PAGE 3 — HISTORY
# # =====================================================

# elif page == "📁 History":

#     st.title("📁 Experiment History")

#     experiments = list_experiments()

#     if not experiments:
#         st.warning("No experiments found.")
#     else:
#         for exp in experiments:

#             exp_path = os.path.join(BASE_DIR, exp)

#             with st.expander(exp):

#                 files = os.listdir(exp_path)
#                 st.write("Files:", files)

#                 submission_path = os.path.join(exp_path, "submission.csv")

#                 if os.path.exists(submission_path):
#                     st.success("Submission Available")

#                     with open(submission_path, "rb") as f:
#                         st.download_button(
#                             label="Download Submission",
#                             data=f,
#                             file_name=f"{exp}_submission.csv",
#                             mime="text/csv",
#                             key=exp
#                         )

#                 if st.button(f"Delete {exp}", key=f"del_{exp}"):
#                     shutil.rmtree(exp_path)
#                     st.success(f"{exp} deleted")
#                     st.rerun()