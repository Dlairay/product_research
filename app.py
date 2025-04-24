from productscan import product_scan
from competitor_finder import get_competitor_list
from pipeline import run_pipeline
from retriever import run_suggestions
import pandas as pd
import concurrent.futures

# === Step 1: Detect product from image ===
image = 'img/img_1.jpeg'
product = product_scan(image)
# product = "Secretlab Titan Evo Lite"
print("✅ Product detected:", product)

# === Step 2: Get competitors ===
competitor_list = get_competitor_list(product)
print("✅ Competitors found:", competitor_list)

# === Step 3: Run pipeline for all products (parallel)
list_to_research = competitor_list + [product]
# list_to_research = ["competitor_list"] + [product]

all_labelled_dfs = []

print(f"🚀 Running pipeline for {len(list_to_research)} products in parallel...")

for item in list_to_research:
    print(f"🔄 Running pipeline for: {item}")
    labelled_df = run_pipeline(item)
    if labelled_df is not None:
        all_labelled_dfs.append(labelled_df)
        print(f"✅ Finished: {item}\n")

# === Step 4: Combine all labelled feedback
if all_labelled_dfs:
    overview_df = pd.concat(all_labelled_dfs, ignore_index=True)
    overview_df.to_pickle("pickle/all_labelled_feedback.pkl")
    print("✅ Combined feedback across products saved.")

# === Step 5: Run suggestions (after all products are processed)
run_suggestions(product)
