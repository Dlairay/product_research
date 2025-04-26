from productscan import product_scan
from competitor_finder import get_competitor_list
from pipeline import run_pipeline
from retriever import run_suggestions
from data_processing import load_and_combine_existing_dataframes,process_data
import pandas as pd
import os 
# === Step 1: Detect product from image ===
image = 'img/img_3.jpeg'
product = product_scan(image)
print("✅ Product detected:", product)

# === Step 2: Get competitors ===
competitor_list = get_competitor_list(product,n=0)

competitor_list = []

list_to_research = competitor_list + [product]


if 'DXRacer MASTER DM1200' in list_to_research:
    list_to_research.remove('DXRacer MASTER DM1200')
    

for item in list_to_research:
    print(f"🔄 Running pipeline for: {item}")
    run_pipeline(item)



combined_labelled_dfs = load_and_combine_existing_dataframes(list_to_research)
processed_data = process_data(combined_labelled_dfs)
print("✅ Processed data:", processed_data)
# === Step 5: Run suggestions (after all products are processed)
run_suggestions("Secretlab TITAN Evo 2022")
