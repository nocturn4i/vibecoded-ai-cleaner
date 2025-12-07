
# app.py — AI CSV Cleaner Web App
import streamlit as st
import pandas as pd
import numpy as np
import re
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(page_title="VibeCoded AI Cleaner", layout="wide")

st.title("🌟 VibeCoded AI CSV Cleaner & Profiler")

# ---- 1️⃣ Upload CSV ----
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df)

    # ---- 2️⃣ AI Intent Cleaning ----
    intent_guidance = {
        "id": "Normalize numeric ID, remove leading zeros, enforce integer",
        "name": "Fill missing with 'Unknown', normalize unicode",
        "age": "Extract numeric age from messy text, invalid -> null",
        "signup_date": "Parse multiple date formats to ISO, preserve missing",
        "revenue": "Strip currency symbols, cast to float"
    }

    # ---- Cleaning Logic ----
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

    # Robust multi-format date parsing
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

    st.subheader("Cleaned Data Preview")
    st.dataframe(df)

    # ---- 3️⃣ Download Cleaned CSV ----
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned CSV",
        data=csv,
        file_name="cleaned_output.csv",
        mime="text/csv"
    )

    # ---- 4️⃣ Data Profiling Report ----
    st.subheader("Automated Data Profile")
    profile = ProfileReport(df, title="AI Data Cleaner Profile", minimal=True)
    st_profile_report(profile)
