# DATA COLLECTION
import requests as re
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings 

# UNSTRUCTURED TO STRUCTURED DATA
from bs4 import BeautifulSoup
import pandas as pd
import feature_extraction as fe

disable_warnings(InsecureRequestWarning)

# CSV TO DATAFRAME
# URL_file_name = "datasets/tranco_list.csv"
URL_file_name = "datasets/verified_online.csv"


data_frame = pd.read_csv(URL_file_name)

# RETRIEVE "url" COLUMN ONLY AND CONVERT IT INTO A LIST
URL_list = data_frame['url'].to_list()

# RESTRICTING URL COUNT DUE TO LARGE NUMBER OF URLs IN THE LIST
begin = 3701
end = 3800
collection_list = URL_list[begin:end]
# print(collection_list)

# ONLY FOR LEGITIMATE URL FILE (Comment out below two lines while collecting data from Phishing URL File)
# tag = "http://"
# collection_list = [(tag + url) for url in collection_list]
# print(collection_list)

# FUNCTION TO SCRAPE THE CONTENT OF THE URL AND CONVERT IT INTO STRUCTURED FORM FOR EACH URL IN THE LIST
def create_structured_data(url_list):
    data_list = []

    for i in range(0, len(url_list)):
        try:
            response = re.get(url_list[i], verify=False, timeout=4)
            if response.status_code != 200:
                print(i, ". HTTP Connection failed for the URL: ", url_list[i])
                # response.close()
            else:
                soup = BeautifulSoup(response.content, "html.parser")
                vector = fe.create_vector(soup)
                vector.append(str(url_list[i]))
                data_list.append(vector)
                # response.close()

        except re.exceptions.RequestException as e:
            print(i, " ---> ", e)
            continue

    return data_list

dataset = create_structured_data(collection_list)

column_names = [
    'has_title',
    'has_input',
    'has_button',
    'has_image',
    'has_submit',
    'has_link',
    'has_password',
    'has_email_input',
    'has_hidden_element',
    'has_audio',
    'has_video',
    'number_of_inputs',
    'number_of_buttons',
    'number_of_images',
    'number_of_option',
    'number_of_list',
    'number_of_th',
    'number_of_tr',
    'number_of_href',
    'number_of_paragraph',
    'number_of_script',
    'length_of_title',
    'has_h1',
    'has_h2',
    'has_h3',
    'length_of_text',
    'number_of_clickable_button',
    'number_of_a',
    'number_of_img',
    'number_of_div',
    'number_of_figure',
    'has_footer',
    'has_form',
    'has_text_area',
    'has_iframe',
    'has_text_input',
    'number_of_meta',
    'has_nav',
    'has_object',
    'has_picture',
    'number_of_sources',
    'number_of_span',
    'number_of_table',
    'URL'
]

df = pd.DataFrame(data=dataset, columns=column_names)

# LABELLING OF THE DATA
# df['label'] = 0 # 0 --> Legitimate
df['label'] = 1 # 1 --> Phishing

# df.to_csv("structured_data_legitimate.csv", mode='a', index=False, header=False) #Header should be false after first run beacause we only want column name once and not after each run

df.to_csv("structured_data_phishing.csv", mode='a', index=False, header=False) #Header should be false after first run beacause we only want column name once and not after each run

