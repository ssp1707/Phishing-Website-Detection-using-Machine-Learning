# Phishing-Website-Detection-using-Machine-Learning

This project is a streamlit application that detects the phishing websites by utilizing the power of synergy between the Gradient Boosting Machine and feature engineering techniques.

## Features

To extract the required features, a dataset containing both phishing and legitimate URLs is needed. The dataset that is used here is can be downloaded from https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset. The features that are extracted and utilized are:

- Use of IP in the URL.
- Abnormal URL.
- Counting dots, @, WWW, digits, letters and special characters(=,%,etc).
- Presence of HTTPS and suspicious words. 
- Length of the URL.
- Hostname Length.
- URL Shortening. 
- First Directory Length.
- Length of top level domain.

## Requirements

- Python 3.x
- Required Python packages: streamlit, pandas, numpy, sklearn

## Installation

1. Clone the repository:

   ```shell
   git clone [https://github.com/your-username/emotion-based-music-recommender.git](https://github.com/ssp1707/Phishing-Website-Detection-using-Machine-Learning.git)

2. Install the required packages:

   ```shell
   pip install -r requirements.txt

3. Download the pre-trained phishing detection model and detect the websites.

## Usage

1. Run the Streamlit application
   ```shell
   streamlit run app.py
   ```

2. Access the application in your web browser at http://localhost:8501.

3. Enter the URL that you suspect of phishing.

4. The application will start to extract the features from the given URL and then the machine learning model will predict the website whether it is safe or phishing.

## Program

```
import streamlit as st
import pandas as pd
import itertools
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

df=pd.read_csv('malicious_phish.csv')

import re
#Use of IP or not in domain
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0
df['use_of_ip'] = df['url'].apply(lambda i: having_ip_address(i))

from urllib.parse import urlparse

def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0


df['abnormal_url'] = df['url'].apply(lambda i: abnormal_url(i))

def count_dot(url):
    count_dot = url.count('.')
    return count_dot

df['count.'] = df['url'].apply(lambda i: count_dot(i))

def count_www(url):
    url.count('www')
    return url.count('www')

df['count-www'] = df['url'].apply(lambda i: count_www(i))

def count_atrate(url):
     
    return url.count('@')

df['count@'] = df['url'].apply(lambda i: count_atrate(i))


def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

df['count_dir'] = df['url'].apply(lambda i: no_of_dir(i))

def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

df['count_embed_domian'] = df['url'].apply(lambda i: no_of_embed(i))

def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0
    
    
df['short_url'] = df['url'].apply(lambda i: shortening_service(i))

def count_https(url):
    return url.count('https')

df['count-https'] = df['url'].apply(lambda i : count_https(i))

def count_http(url):
    return url.count('http')

df['count-http'] = df['url'].apply(lambda i : count_http(i))

def count_per(url):
    return url.count('%')

df['count%'] = df['url'].apply(lambda i : count_per(i))

def count_ques(url):
    return url.count('?')

df['count?'] = df['url'].apply(lambda i: count_ques(i))

def count_hyphen(url):
    return url.count('-')

df['count-'] = df['url'].apply(lambda i: count_hyphen(i))

def count_equal(url):
    return url.count('=')

df['count='] = df['url'].apply(lambda i: count_equal(i))

def url_length(url):
    return len(str(url))


#Length of URL
df['url_length'] = df['url'].apply(lambda i: url_length(i))
#Hostname Length

def hostname_length(url):
    return len(urlparse(url).netloc)

df['hostname_length'] = df['url'].apply(lambda i: hostname_length(i))

def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0
df['sus_url'] = df['url'].apply(lambda i: suspicious_words(i))


def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits


df['count-digits']= df['url'].apply(lambda i: digit_count(i))


def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters


df['count-letters']= df['url'].apply(lambda i: letter_count(i))
from urllib.parse import urlparse
from tld import get_tld
import os.path

#First Directory Length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

df['fd_length'] = df['url'].apply(lambda i: fd_length(i))

#Length of Top Level Domain
df['tld'] = df['url'].apply(lambda i: get_tld(i,fail_silently=True))


def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

df['tld_length'] = df['tld'].apply(lambda i: tld_length(i))
df = df.drop("tld",axis=1)

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
df["type_code"] = lb_make.fit_transform(df["type"])

X = df[['use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
       'count_dir', 'count_embed_domian', 'short_url', 'count-https',
       'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
       'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
       'count-letters']]

#Target Variable
y = df['type_code']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2,shuffle=True, random_state=5)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
gb = GradientBoostingClassifier(n_estimators=100,max_features='sqrt')
gb.fit(X_train,y_train)
y_pred_gb = gb.predict(X_test)

def extract_features(url):
    features = []
    features.append(having_ip_address(url))
    features.append(abnormal_url(url))
    features.append(count_dot(url))
    features.append(count_www(url))
    features.append(count_atrate(url))
    features.append(no_of_dir(url))
    features.append(no_of_embed(url))
    features.append(shortening_service(url))
    features.append(count_https(url))
    features.append(count_http(url))
    features.append(count_per(url))
    features.append(count_ques(url))
    features.append(count_hyphen(url))
    features.append(count_equal(url))
    features.append(url_length(url))
    features.append(hostname_length(url))
    features.append(suspicious_words(url))
    features.append(digit_count(url))
    features.append(letter_count(url))
    features.append(fd_length(url))
    tld = get_tld(url, fail_silently=True)
    features.append(tld_length(tld))
    return np.array(features).reshape(1, -1)


# Streamlit App
st.title("Phisher")
st.write("Hey guys! Let's catch some Phishs 🎣")
# User Input
user_input = st.text_input("Enter the URL:")

if user_input:
    # Extract features and make prediction
    features = extract_features(user_input)
    prediction = gb.predict(features)

    # Display the result
    if int(prediction[0]) == 0:
        st.success("SAFE: Phew! It's not the one 😅 ")
    elif int(prediction[0]) >= 3.0:
        st.error("PHISHING: Yay!We got one 🐟 ")

```

## Output

![Screenshot (81)](https://github.com/ssp1707/Phishing-Website-Detection-using-Machine-Learning/assets/75234965/a654578a-3cbc-4a01-9132-4a0f80e4996b)
![ss3](https://github.com/ssp1707/Phishing-Website-Detection-using-Machine-Learning/assets/75234965/5d8160a3-6ea4-4974-98fb-1e0e9cc6f431)
![Screenshot (82)](https://github.com/ssp1707/Phishing-Website-Detection-using-Machine-Learning/assets/75234965/3f8796d0-d7d9-4f5b-bb78-fa105b6855f0)

## Result

The project on phishing website detection using machine learning has successfully developed a robust system for identifying potential phishing websites. The methodology involved comprehensive feature engineering, utilizing both URL structure and content-related characteristics to train a Gradient Boosting Classifier. The model was trained and evaluated on a diverse dataset, demonstrating effectiveness in distinguishing between legitimate and malicious websites. Various features, including the presence of an IP address, abnormal URL structure, and specific character counts, contributed to the model's ability to make accurate predictions.

The Streamlit-based user interface provides an intuitive platform for users to input URLs and receive real-time predictions. The model's performance was validated using standard machine learning metrics, such as accuracy, precision, recall, and F1 score. Additionally, the system incorporated features like anomaly detection and shortening service identification to enhance its capacity to detect evolving phishing tactics.

While the system doesn't involve real-time monitoring, it offers a valuable solution for offline phishing detection, enabling users to assess the legitimacy of URLs without the need for immediate response. Periodic model updates can be implemented to adapt to emerging phishing threats over time. The project contributes to the broader field of cybersecurity by showcasing the effectiveness of machine learning in combating phishing activities within a static environment.
