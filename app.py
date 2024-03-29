import streamlit as st
import numpy as np
import pandas as pd
import string
from tensorflow.keras.models import load_model

# Load your trained model
loaded_model = load_model("Word_Correction.h5")

# Load your data
data = pd.read_csv("Word_label_dict.csv")  # Make sure to replace "your_data.csv" with your dataset file
dataa = pd.read_csv("OPTED-Dictionary.csv")

# Create arrays for uppercase and lowercase letters
lowercase_list = np.array(list(string.ascii_lowercase))
uppercase_list = np.array(list(string.ascii_uppercase))

def mat(input_string):
    lst = np.zeros(26, dtype=int)  # Initialize a NumPy array filled with zeros

    for char in input_string:
        if char.isupper():
            index = np.where(uppercase_list == char)[0]  # Find the index of the uppercase letter
            if len(index) > 0:
                lst[index[0]] += 1
        elif char.islower():
            index = np.where(lowercase_list == char)[0]  # Find the index of the lowercase letter
            if len(index) > 0:
                lst[index[0]] += 1

    return pred(lst)

def pred(array):
    y = loaded_model.predict(np.array([array]))  # Pass array as a numpy array
    predicted_classes = np.argmax(y, axis=1)
    predicted_prob = np.max(y)  # Get the highest probability
    return predicted_classes[0], predicted_prob

def main():
    st.title("**Smart Dictionary with Auto-Correction**")
    input_text = st.text_input("Enter the Word")
    if st.button("Predict"):
        result, probability = mat(input_text)
        predicted_word = data[data.Label == result].Word.to_list()[0]
        definition = dataa[dataa['Word'] == predicted_word]['Definition'].values
        if len(definition) > 0 and (len(dataa[dataa['Word'].str.upper() == input_text.upper()]) > 0): #probability*100 >= 90:  # Check if definition is not empty
            st.write(f"Entered Word is: {predicted_word}")
            #st.write(f"Probability: {probability:.2f}")
            st.write(f"Its dictionary meaning is: {definition[0]}")
        elif len(definition) > 0 and (len(dataa[dataa['Word'].str.upper() == input_text.upper()]) == 0):#probability*100 < 90:
            st.write(f"Did you mean: {predicted_word}?")
            #st.write(f"Probability: {probability:.2f}")
            st.write(f"Its dictionary meaning is: {definition[0]}")
        else:
            st.write(f"No definition found for '{predicted_word}' in the dictionary.")

if __name__ == "__main__":
    main()
