import streamlit as st
import tensorflow as tf
from config import _Bilstm_tweet_classifier_model
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
import re
import nltk
import datetime

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


def load_model(path):
    '''
    Load the tensorflow Bilstm tweet classifier model
    args:
     - path: Path to the tweet classifier model
    returns:
      - model: loaded tweet classifier
    '''
    model = tf.keras.models.load_model(path)
    return model


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def preprocess_text(text):
    # Ensure the text is in string format
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    elif isinstance(text, np.ndarray):
        text = text.astype(str)
    elif isinstance(text, datetime.datetime):
        # convert datetime dtype to string
        text = text.strftime('%Y-%m-%d %H:%M:%S')
    # convert other non text dtype to text
    elif not isinstance(text, str):
        text = str(text)
    # Convert text to lowercase
    lowercase_text = text.lower()
    # Remove punctuations using regex
    removed_punctuations = re.sub(r'[^\w\s]', '', lowercase_text)

    var_stop_words = set(stopwords.words('english'))
    words = removed_punctuations.split()
    words_filtered = [word for word in words if word.isalpha()
                      and word not in var_stop_words]
    if isinstance(words_filtered, list):
        lemmatizer = WordNetLemmatizer()
        words_lemmatized = [lemmatizer.lemmatize(
            word, get_wordnet_pos(word)) for word in words_filtered]
        return ' '.join(words_lemmatized)  # Join the list back into a string
    else:
        raise ValueError('Input should be a list of words')


def make_prediction(model, input_text):
    '''
    Makes prediction on the tweet using the model.predict method, and then
    converts the model into a class label and confidence score
    args:
     - model: the classifier model to make prediction with
     - input_text: the tweet to be classified
    returns:
     - pred_label: the predicted class label
     - confidence: the confidence score of the prediction
    '''
    # Ensure input_text is a list
    input_text = [input_text] if isinstance(input_text, str) else input_text
    input_text = preprocess_text(input_text[0])  # Preprocess the input text
    # Convert input_text to numpy array and reshape for prediction
    input_text = np.array([input_text])
    # make prediction on the text input
    pred = model.predict(input_text)
    # squeeze and get prediction probability
    pred_prob = np.squeeze(pred)
    confidence = pred_prob * 100
    pred_class = np.round(pred_prob).astype(int)
    # convert prediction probability into label
    class_labels = ['Disaster', 'Suicide']
    pred_label = class_labels[pred_class]
    return pred_label, confidence


def main():
    '''
    Main function for the Streamlit web app
    returns the Streamlit web app user interface
    '''
    st.title(
        'TWEET SENTIMENT ANALYSIS FOR DETECTION OF SUICIDALITY AND DISASTER WEB APP')
    st.write('Enter a Tweet to Classify:')

    text = st.text_input('Tweet')

    # Load the model
    model = load_model(_Bilstm_tweet_classifier_model)

    confidence_threshold = 0.65  # Threshold for low confidence

    if st.button('Predict'):
        if text:
            # Make prediction
            prediction, confidence = make_prediction(model, text)

            # Display the prediction
            st.write(
                f'This tweet is classified as a **{prediction}** tweet with a confidence score of **{confidence:.2f}%**.')

            # Inform the user if the confidence score is low
            if confidence < confidence_threshold:
                st.write(
                    "**Note:** The model is not very confident in this prediction. Please seek more information or confirmation.")

            # Additional information
            if prediction == 'Suicide':
                st.write(
                    "**Note:** This tweet contains language related to suicide. If you or someone you know is in crisis, please seek help immediately.")
                st.write("Helpful Resources:")
                st.write(
                    "- [National Suicide Prevention Lifeline](https://suicidepreventionlifeline.org/): 1-800-273-TALK (8255)")
                st.write(
                    "- [Crisis Text Line](https://www.crisistextline.org/): Text HOME to 741741")

            elif prediction == 'Disaster':
                st.write(
                    "**Note:** This tweet is related to a disaster. Stay informed and stay safe.")
                st.write("Helpful Resources:")
                st.write(
                    "- [Ready.gov](https://www.ready.gov/): Prepare, plan and stay informed for emergencies.")
                st.write(
                    "- [Red Cross](https://www.redcross.org/): Find help and ways to get involved in disaster relief.")

        else:
            st.write("Please enter a tweet to classify.")

    # Sidebar with more information
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app uses a BiLSTM model to classify tweets into categories of 'Suicide' and 'Disaster'. 
        It provides a confidence score for the prediction and resources for users who may need help.
        
        This is a final year project from the Department of Computer Engineering at Federal University Oye Ekiti.
        """
    )
    st.sidebar.write(
        "Developers: Adedapo Adebola Dorcas and Folarin Eniola Ayotomiwa")


if __name__ == '__main__':
    main()
