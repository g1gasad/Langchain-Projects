from dotenv import load_dotenv
load_dotenv() ## Load all the environment variables from .env

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load Gemini Pro Vision model and get responses

model = genai.GenerativeModel("gemini-pro-vision")

def get_gemini_response(input, image, prompt):
    response=model.generate_content([input, image[0], prompt])
    return response.text

def input_image_details(uploaded_file):
    if uploaded_file is not None:
    # Read the file into bytes
        bytes_data = uploaded_file.getvalue()
        
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No File Uploaded")
        

# Initialize our StreamLit app

st.set_page_config(page_title="Multi-Language Invoice Extractor")
st.header("Multi-Language Invoice Extractor")

input=st.text_input("Input Prompt: ", key='input')
uploaded_file = st.file_uploader("Choose an image of the invoice...", type=['jpg', 'jpeg', 'png'])
image=''

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

submit=st.button("Tell me about the invoice")

input_prompt = """
You are an expert in understanding invoice. We will upload an invoice as an image and
you will have to answer every questions based on the uploaded invoice image
"""

# When submit is clicked

if submit:
    image_data=input_image_details(uploaded_file)
    response=get_gemini_response(input_prompt, image_data, input)
    st.subheader("The Response is")
    st.write(response)
    
    
