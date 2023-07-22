import streamlit as st
import pickle
import nltk
import re
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

st.title('Spam detector classifier')

tfidf=pickle.load(open('vectorizer.pkl','rb'))

algorithm=pickle.load(open('algorithm.pkl','rb'))

title = st.text_area('Enter the message', 'URGENT! You have won a 1 week FREE membership in our �100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18')


#the text preprocessing:-
def perform_preprocessing(text):
    text = text.lower()
    text = word_tokenize(text)
    
    #we have to get rid of alpha numerical columns
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)
if st.button('Predict'):
    processed_text=perform_preprocessing(title)

    transformed_text=tfidf.transform([processed_text])

    result=algorithm.predict(transformed_text)[0]

    if result==1:
        st.header('The message is likely spam')
    else:
        st.header('The message is likely ham')
