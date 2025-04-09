import google.generativeai as genai

import json
from dotenv import load_dotenv # for loading api keys in
import os
import base64
import PIL

load_dotenv()




def encode_image(image):
  with open(image, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  

image = 'img/mehmeh.jpeg' 
base64_image = encode_image(image)

client =  genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))


# model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
# response = model.generate_content([
# "Perform the following steps--- Step 1: State the product shown. return only a string of what the product is:", PIL.Image.open('img/mehmeh.jpeg')])

# print(response.text)

def get_product(base64_image):
  model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
  response = model.generate_content([
"Perform the following steps--- Step 1: State the product shown. return only a string of what the product is:", PIL.Image.open('img/mehmeh.jpeg')])

  # print(response)
  product = response.text
  return product

