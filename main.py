import streamlit as st
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle 
import numpy as np

st.title("Stany Ganyani R204442S")
st.title("Next Word Predictor For Shona")

words = st.text_input("Nyora Apa")

# st.write(words)

with open('tokenizer.pickle','rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)


model =  tf.keras.models.load_model('second_model.h5')

def predict_next_words(model, tokenizer, text, num_words=1):
    for _ in range(num_words):

        encoded_data = tokenizer.texts_to_sequences([text])[0]
        sequence = pad_sequences([encoded_data], maxlen=5, padding='pre')


        predicted_probs = model.predict(sequence, verbose=0)
        predicted = np.argmax(predicted_probs, axis=-1)


        next_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                next_word = word
                break


        text += " " + next_word

    return ' '.join(text.split(' ')[-num_words:])


if st.button("Predict"):
    predicted_words = predict_next_words(model, tokenizer, words, num_words=1)
    st.write(f"mazwi anotevera: {predicted_words}")