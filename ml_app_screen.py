import streamlit as st
from streamlit_lottie import st_lottie
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests
from PIL import Image


def ml_app():
    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    
    # --- HEADER SECTION ---
    with st.container():
        left_column, right_column = st.columns((2,1))
        with left_column:
            st.title('WELCOME! KNIGHT⚔️')
            st.subheader('This a content based Machine Learning App which is developed to detect Phishing Websites.')

        with right_column:
            lottie_secure = load_lottieurl("https://lottie.host/ef003e80-0e69-436f-a632-aae38a02366d/PTzpA1SW3j.json")

            st_lottie(lottie_secure, height=300, key="secure") 

    with st.container():
        st.write('For details regarding DATASET and FEATURE SET, click on _"Project Details"_.')
        with st.expander("PROJECT DETAILS"):
            st.subheader('DATASET DETAILS')
            st.write('For the list of Phishing Websites _"phishtank.org"_ is used and for the list of Legitimate Websites _"tranco-list.eu"_ is used as data source.')
            st.write('Total 31600 Websites ==> **_16100_ Legitimate** Websites | **_15500_ Phishing** Websites')
            st.write('DATASET was created in November 2023.')

            # --- PIE CHART AND BAR GRAPH --- 
            with st.container():
                left_image_column, right_lottie_column = st.columns((1,1))
                with left_image_column:
                    dataset_img = Image.open("images/piechart.png")
                    st.image(dataset_img)

                with right_lottie_column:
                    bar_img = Image.open("images/bargraph.png")
                    st.image(bar_img)

            # --- FEATURES ---
            st.write('Features + URL + Label ==> Dataframe')
            st.markdown('Label is 1 for Phishing and Label 0 is for Legitimate')
            number = st.slider("Select row number to display", 0, 100)
            st.dataframe(ml.df.head(number))

            @st.cache_data
            def convert_df(df):
                # Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(ml.df)

            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name='phishing_legitimate_structured_data.csv',
                mime='text/csv',
            )

            st.subheader('Features')
            st.write('Only content-based features are used.'
                     ' Most of the features are extracted using find_all() method of BeautifulSoup Module after Parsing HTML.')

            # --- RESULTS ---
            st.subheader('Results')
            st.write('6 different ML Classifiers of scikit-learn are tested then implemented using k-fold cross validation.'
                    ' Firstly their confusion matrices is obtained then their accuracy, precision, recall and f1 Scores are calculated.'
                    ' Random Forest and Decision Tree Classifiers performed the best.'
                    ' _Comparison table is below:_')
            
            st.table(ml.df_results)
            st.write('NB --> Gaussian Naive Bayes')
            # st.write('SVM --> Support Vector Machine')
            st.write('DT --> Decision Tree')
            st.write('RF --> Random Forest')
            st.write('AB --> AdaBoost')
            st.write('NN --> Neural Network')
            st.write('KNN --> K-Nearest Neighbours')

    st.write("---")

    with st.expander('EXAMPLE PHISHING URLs:'):
        st.write('_https://new.express.adobe.com/webpage/Mqu6j30G7g4bz_')
        st.write('_https://poczta-polsku.top/wAW6po_')
        st.write('_http://support-45098ey.surge.sh/_')
        st.caption('PHISHING WEB PAGES HAVE SHORT LIFECYCLE! SO, THE EXAMPLES MAY NOT WORK!')

    st.write("---")

    choice = st.selectbox("Please select your desired Machine Learning Model",
                    [
                        'Gaussian Naive Bayes', 'Decision Tree', 'Random Forest',
                        'AdaBoost', 'Neural Network', 'K-Nearest Neighbours'
                    ]
                    )

    if choice == 'Gaussian Naive Bayes':
        model = ml.nb_model
        st.write('GNB model is selected!')

    # elif choice == 'Support Vector Machine':
    #     model = ml.svm_model
    #     st.write('SVM model is selected!')

    elif choice == 'Decision Tree':
        model = ml.dt_model
        st.write('DT model is selected!')

    elif choice == 'Random Forest':
        model = ml.rf_model
        st.write('RF model is selected!')

    elif choice == 'AdaBoost':
        model = ml.ab_model
        st.write('AB model is selected!')

    elif choice == 'Neural Network':
        model = ml.nn_model
        st.write('NN model is selected!')

    else:
        model = ml.knn_model
        st.write('KNN model is selected!')


    url = st.text_input('Enter complete URL', placeholder="https://website.xyz")
    # check the url is valid or not
    if st.button('Check!'):
        try:
            response = requests.get(url, verify=False, timeout=4)
            if response.status_code != 200:
                st.error("HTTP connection was not successful for the URL: ", url, icon="❌")
            else:
                soup = BeautifulSoup(response.content, "html.parser")
                vector = [fe.create_vector_optimised(soup)]  # it should be 2d array so [] is added
                # print(vector.shape)
                result = model.predict(vector)
                if result[0] == 0:
                    st.success("This Webpage seems Legitimate!", icon="😄")
                    st.balloons()
                else:
                    st.warning("Attention! This Webpage is a potential PHISHING WEBSITE!", icon="🥶")
                    st.snow()

        except requests.exceptions.RequestException as e: 
            st.error(e, icon="❌")