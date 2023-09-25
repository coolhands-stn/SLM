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

# Convert Text to numerical data
encoded_data = tokenizer.texts_to_sequences([words])[0]
# Create sequences
sequences = pad_sequences([encoded_data], maxlen=5, padding='pre')

def predict_next_words(model, tokenizer, words, num_words=1):
    for _ in range(num_words):
        # Tokenize and pad the text
        sequence = tokenizer.texts_to_sequences([words])[0]
        sequence = pad_sequences([sequence], maxlen=5, padding='pre')
        
        # Predict the next word
        predicted_probs = model.predict(sequence, verbose=0)
        predicted = np.argmax(predicted_probs, axis=-1)
        
        # Convert the predicted word index to a word
        output_word = ""
        text = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        # Append the predicted word to the text
        text += " " + output_word

    return ' '.join(text.split(' ')[-num_words:])


model = tf.keras.models.load_model("first_model.h5")

predicted_words = predict_next_words(model, tokenizer, words, num_words=1)
st.write('izwi rinotevera : ', predicted_words)