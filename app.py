import streamlit as st
import pandas as pd
import numpy as np
import re
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import openai

# ------------------------------
# ⚡ OpenAI client setup
# ------------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ------------------------------
# Streamlit page config
# ------------------------------
st.set_page_config(page_title="VibeCoded AI Cleaner", layout="wide")
st.title("🌟 VibeCoded AI CSV Cleaner & Profiler with LLM")

# ------------------------------
# CSV uploader
# ------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# ------------------------------
# LLM helper function
# ------------------------------
def ask_llm_for_cleaning(csv_sample: str):
    prompt = f"""
You are a data-cleaning AI assistant. Here is a CSV snippet:

{csv_sample}

Generate Python pandas code to clean it.
Return only code, no explanations.
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful data-cleaning assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content


# ------------------------------
# Main logic
# ------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df)

    # ------------------------------
    # Local cleaning rules (existing)
    # ------------------------------
    if "id" in df.columns:
        df["id"] = df["id"].astype(str).str.lstrip("0").replace({"": None})
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")

    if "name" in df.columns:
        df["name"] = df["name"].fillna("Unknown").astype(str).str.normalize("NFKD")

    if "age" in df.columns:
        def parse_age(x):
            try:
                return int(str(x).strip())
            except:
                m = re.search(r"(\d+)", str(x))
                return int(m.group(1)) if m else pd.NA
        df["age"] = df["age"].apply(parse_age).astype("Int64")

    if "signup_date" in df.columns:
        df["signup_date"] = pd.to_datetime(
            df["signup_date"], errors="coerce", infer_datetime_format=True
        )
        mask = df["signup_date"].isna()
        df.loc[mask, "signup_date"] = pd.to_datetime(
            df.loc[mask, "signup_date"].astype(str).str.replace(".", "-", regex=False),
            errors="coerce"
        )
        df["signup_date"] = df["signup_date"].dt.date

    if "revenue" in df.columns:
        df["revenue"] = df["revenue"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    st.subheader("Cleaned Data Preview (Local Rules)")
    st.dataframe(df)

    # ------------------------------
    # LLM Cleaning button
    # ------------------------------
    if st.button("Run AI LLM Cleaning"):
        sample_csv = df.head(10).to_csv(index=False)
        code_from_llm = ask_llm_for_cleaning(sample_csv)
        st.subheader("LLM Suggested Cleaning Code")
        st.code(code_from_llm, language="python")

        # ⚡ Execute LLM-generated code safely
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
    # Profile report
    # ------------------------------
    st.subheader("Automated Data Profile")
    profile = ProfileReport(df, title="AI Data Cleaner Profile", minimal=True)
    st_profile_report(profile)
