# from productscan import get_product, encode_image
from productscan_gemini import get_product, encode_image

from youtube import youtube_data_collection, load_data_to_pandas
from tavily import webscrape


image = 'img/mehmeh.jpeg' 
base64_image = encode_image(image)
product = get_product(base64_image)
print(product)


data = youtube_data_collection(product+ "review", max_result=5)
print("Data collection completed. Data saved to the directory 'youtube_data'.")

comments_df = load_data_to_pandas()
print(comments_df.columns)
print(comments_df["comment"])

webscrape_result = webscrape(product)
webscrape_answer = webscrape_result[0]
webscrape_url = webscrape_result[1]

