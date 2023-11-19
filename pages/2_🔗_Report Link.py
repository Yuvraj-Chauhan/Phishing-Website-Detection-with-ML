import streamlit as st
from streamlit_lottie import st_lottie
import requests
from config_creds import reportForm

# --- PAGE CONFIGURATIONS ---
st.set_page_config(page_title="Report Link", page_icon="🔗", layout="wide")
st.title("Report Phishing Website Link 🎣")
st.header("Join The Cause! 🫱🏻‍🫲🏻")
st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"]::before {
                content: "PhishGuardian🛡️";
                margin-left: 15px;
                # margin-top: 20px;
                font-size: 40px;
                font-weight: bold;
                position: relative;
                top: 50px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- LOCAL CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style//style.css")

# --- LOAD LOTTIE FILES ---
def load_lottieurl(url):
    rphish = requests.get(url)
    if rphish.status_code != 200:
        return None
    return rphish.json()

lottie_phish = load_lottieurl("https://lottie.host/65348f44-4d59-4a58-9905-ea814bf9fd7f/3ahxW4Awp8.json")

# --- REPORT LINK FORM ---
report_form = reportForm

left_column, right_column = st.columns(2)
with left_column:
    st.markdown(report_form, unsafe_allow_html=True)
with right_column:
    st_lottie(lottie_phish, height=300, key="phish")
    