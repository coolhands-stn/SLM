import streamlit as streamlit
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

st.title(" Shona words prediction ")

words = st.text_input("Input your text")

st.write(words)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)


encoded_data = tokenizer.texts_to_sequences([words])[0]
sequences = pad_sequences([encoded_data], maxlen=6, padding='pre')

# Load the model
model = tf.keras.models.load_model("nameofmodel")
predicted_probs = model.predict(sequences)
predicted = np.argmax(predicted_probs, axis=-1)
    
    # Convert the predicted word index to a word
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break

st.write('your next word is', output_word)

