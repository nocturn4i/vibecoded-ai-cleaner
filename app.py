import streamlit as st
import pandas as pd
import numpy as np
import re
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import openai
import os

# ------------------------------
# ⚡ Set OpenAI API key
# ------------------------------
# You MUST set this in Streamlit Cloud Secrets:
# OPENAI_API_KEY = <your API key>
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ------------------------------
# Streamlit page config
# ------------------------------
st.set_page_config(page_title="VibeCoded AI Cleaner", layout="wide")
st.title("🌟 VibeCoded AI CSV Cleaner & Profiler with LLM")

# ------------------------------
# Upload CSV
# ------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# ------------------------------
# LLM helper function
# ------------------------------
def ask_llm_for_cleaning(csv_sample: str):
    prompt = f"""
You are a data-cleaning AI assistant. 
I provide you a CSV snippet below:

{csv_sample}

For each column, generate a Python pandas command to clean it.
- Keep missing values as <NA>, NaT, NaN
- Parse dates robustly
- Strip non-numeric from numeric columns
Return only valid Python code without explanations.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    code = response.choices[0].message.content
    return code

# ------------------------------
# Main logic
# ------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df)

    # ------------------------------
    # Step 1: Local AI Cleaning (existing logic)
    # ------------------------------
    df["id"] = df["id"].astype(str).str.lstrip("0").replace({"": None})
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")

    df["name"] = df["name"].fillna("Unknown").astype(str).str.normalize("NFKD")

    def parse_age(x):
        try:
            return int(str(x).strip())
        except:
            m = re.search(r"(\d+)", str(x))
            return int(m.group(1)) if m else pd.NA
    df["age"] = df["age"].apply(parse_age).astype("Int64")

    df["signup_date"] = pd.to_datetime(
        df["signup_date"], errors="coerce", infer_datetime_format=True
    )
    mask = df["signup_date"].isna()
    df.loc[mask, "signup_date"] = pd.to_datetime(
        df.loc[mask, "signup_date"].astype(str).str.replace(".", "-", regex=False),
        errors="coerce"
    )
    df["signup_date"] = df["signup_date"].dt.date

    df["revenue"] = df["revenue"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    st.subheader("Cleaned Data Preview (Local Rules)")
    st.dataframe(df)

    # ------------------------------
    # Step 2: Optional LLM Cleaning
    # ------------------------------
    if st.button("Run AI LLM Cleaning"):
        # Take a small sample to send to LLM (first 10 rows)
        sample_csv = df.head(10).to_csv(index=False)
        code_from_llm = ask_llm_for_cleaning(sample_csv)
        st.subheader("LLM Suggested Cleaning Code")
        st.code(code_from_llm, language="python")

        # ⚡ Execute the LLM code safely
        exec(code_from_llm, {"df": df, "pd": pd, "np": np, "re": re})

        st.subheader("Cleaned Data Preview (LLM Applied)")
        st.dataframe(df)

    # ------------------------------
    # Download cleaned CSV
    # ------------------------------
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned CSV",
        data=csv,
        file_name="cleaned_output.csv",
        mime="text/csv"
    )

    # ------------------------------
    # Data Profiling Report
    # ------------------------------
    st.subheader("Automated Data Profile")
    profile = ProfileReport(df, title="AI Data Cleaner Profile", minimal=True)
    st_profile_report(profile)
