from openai import OpenAI
import json
from dotenv import load_dotenv # for loading api keys in
import os
import base64

load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY") # Load OpenAI API key from environment variable
open_ai_client = OpenAI(api_key=OPEN_AI_API_KEY) # Initialize OpenAI client

def encode_image(image):
  with open(image, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def get_product(base64_image):
  response = open_ai_client.chat.completions.create(
                                            
    model="gpt-4o-mini",
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text", "text": """State the product shown, do your best to identify the exact product brand and model.
            positive example:
            - Apple iPhone 13 Pro Max
            - Mercedes Benz G-Class
            - secret lab gaming chair
            - razor deathadder v3 pro
            negative example: 
            - phone
            - car
            - gaming chair
            - computer mouse

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


def product_scan(image):
  base64_img = encode_image(image)
  product = get_product(base64_img)
  
  return product


# === Main block for testing ===
if __name__ == "__main__":
    load_dotenv()
    OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY") # Load OpenAI API key from environment variable
    open_ai_client = OpenAI(api_key=OPEN_AI_API_KEY) # Initialize OpenAI client

    image = 'img/img_1.jpeg'
    print("Scanning image for product...")
    result = product_scan(image)
    print("Identified product:", result)