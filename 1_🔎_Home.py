import pyrebase
import streamlit as st
from config_creds import firebaseConfigCreds
from ml_app_screen import ml_app
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings 
disable_warnings(InsecureRequestWarning)

# CONFIGURATION KEY
firebaseConfig = firebaseConfigCreds

# FIREBASE AUTHENTICATION
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# DATABASES
db = firebase.database()
storage = firebase.storage()

def add_logo():
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

def main_app():
    # SETUP PAGE CONFIGURATIONS
    st.set_page_config(page_title="PhishGuardian", page_icon="🛡️", layout="wide")
    add_logo()
    local_css("style//style.css")

    if "user_email" not in st.session_state:
        st.session_state["user_email"] = ""

    # APP FOR TESTING PURPOSES
    def test_app(): 
        if st.session_state["user_login"]:
           st.write("Text App")

    #USER LOGIN FUNCTION
    def user_login():
        try:
            # st.header("Logged In")
            if not st.session_state["user_login"]:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state["user_login"] = True
                st.session_state["user_email"] = email
                st.success("Logged in Successfully", icon="✅")
                st.balloons()
                # run_app()
                # test_app()

                # st.session_state.signout = True 

        except:
            st.error("Invalid Login Credentials", icon="☠️")

    #REGISTER FUNCTION
    def register():
        try:
            user = auth.create_user_with_email_and_password(email, password)
            st.success("Your Account Is Created Successfully!", icon="✅")
            st.balloons()

            # #Login
            # user = auth.sign_in_with_email_and_password(email, password)
            db.child(user["localId"]).child("Username").set(username)
            # db.child(user["localId"]).child("ID").set(user["LocalId"])

            st.title(f"WELCOME {username} 👋🏻")
            st.info("You can now Login via Login Dropdown", icon="💡")

        except:
                st.warning("Enter Details Correctly", icon="😵‍💫")

    # GUEST LOGIN FUNCTION
    def guest_login():
        if not st.session_state["guest_login"]:
            st.session_state["guest_login"] = True
            st.session_state["user_login"] = False
            # st.session_state["signout"] = False
            st.session_state["user_email"] = email
            st.info("Continuing as a GUEST", icon="💡")

    # LOG OUT FUNCTION        
    def user_logout():
        st.session_state["user_login"] = False   
        st.session_state["user_email"] = ""
        st.session_state["guest_login"] = False
    
    # BACK TO LOGIN/REGISTER PAGE FUNCTION
    def guest_logout():
        st.session_state["guest_login"] = False

    # INITIALIZING SESSION STATES
    if "user_login" not in st.session_state:
        st.session_state["user_login"] = False

    if "guest_login" not in st.session_state:
        st.session_state["guest_login"] = False

    # LOGIN/REGISTER PAGE
    if not (st.session_state["user_login"] or st.session_state["guest_login"]):
        with st.container():
            st.title("Login or Register 🥷🏻")
            st.write("##")
            choice = st.selectbox("Select Login / Register", ["Login", "Register"])

            email = st.text_input("Enter Email Address", placeholder="Email ID")
            password = st.text_input("Enter Password", placeholder="Password", type="password")

            if choice == "Register":
                username = st.text_input("Enter Your Username", placeholder="Username")
                st.button("Create my Account", on_click=register)
                
            elif choice == "Login":
                st.button("Login", on_click=user_login)

            st.button("Continue As Guest", on_click=guest_login)

    # IF USER IS LOGGED IN OR GUEST IS LOGGED IN
    if st.session_state["user_login"] or st.session_state["guest_login"]:
            ml_app()
            # st.text('Name '+st.session_state.username)
            if st.session_state["user_login"]:
                st.write("---")
                st.text("User Email ID: "+ st.session_state["user_email"])
                st.button("Log Out", on_click=user_logout) 
            elif st.session_state["guest_login"]:
                st.write("---")
                st.subheader("Continue with Login or Register")
                st.button("Login/Register", on_click=guest_logout)
            else:
                st.error("Session State Error, Please Restart The Application", icon="❌")

if __name__ == "__main__":
    main_app()