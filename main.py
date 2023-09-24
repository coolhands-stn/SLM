import streamlit as st
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle 
import numpy as np


st.title(" Next Word Predictor For Shona ")

words = st.text_input("Type Text")

st.write(words)

with open('tokenizer.pickle','rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)


encoded_data = tokenizer.texts_to_sequences([words])[0]
sequences = pad_sequences([encoded_data], maxlen=5, padding='pre')

# Load the model
model = tf.keras.models.load_model("best_model1.h5")
predicted_probs = model.predict(sequences)
predicted = np.argmax(predicted_probs, axis=-1)
    
# Convert the predicted word index to a word

output_word = ""

for word, index in tokenizer.word_index.items():
    if index == predicted:
        output_word = word
        break

st.write('your next word is', output_word)