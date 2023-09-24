from tensorflow.keras.models import load_model
import streamlit as st
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle


st.title(" Next Word Predictor For Shona ")

words = st.text_input("Type Text")

st.write(words)

with open('tokenizer.pickle','rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)


encoded_data = tokenizer.texts_to_sequences([words])[0]
sequences = pad_sequences([encoded_data], maxlen=5, padding='pre')

# Load the model
model = load_model("best_model1.h5")
predicted_probs = model.predict(sequences)
predicted = np.argmax(predicted_probs, axis=-1)
    
# Convert the predicted word index to a word

output_word = ""

for word, index in tokenizer.word_index.items():
    if index == predicted:
        output_word = word
        break

st.write('your next word is', output_word)