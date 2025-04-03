from openai import OpenAI
import json
from dotenv import load_dotenv # for loading api keys in
import os
import base64

load_dotenv()




def encode_image(image):
  with open(image, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  

image = 'img/img_1.jpeg' 
base64_image = encode_image(image)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 



def get_product(base64_image):
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text", "text": """Perform the following steps--- Step 1: State the product shown. 
            return only a string of what the product is"""
            },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64," + base64_image
            },
          },
        ],
      }
    ],
    max_tokens=300,               # max length of response
  )

  product = response.choices[0].message.content
  return product

