import streamlit as st
from streamlit_lottie import st_lottie
import requests
from config_creds import contactForm

st.set_page_config(page_title="Contact", page_icon="✉️", layout="wide")
st.title("Contact Us 📬")
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
    rcontact = requests.get(url)
    if rcontact.status_code != 200:
        return None
    return rcontact.json()

lottie_contact = load_lottieurl("https://lottie.host/829cc1d2-6834-4d26-b419-0ccc5d35663d/eNCKbjiSFQ.json")


# --- CONTACT US FORM ---
contact_form = contactForm

left_column, right_column = st.columns(2)
with left_column:
    st.markdown(contact_form, unsafe_allow_html=True)
with right_column:
    st_lottie(lottie_contact, height=300, key="phish")
    