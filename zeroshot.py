from transformers import pipeline
import pandas as pd
from youtube_tool import load_YouTube_df

# Load the YouTube data
product = "Secretlab Titan Evo 2022 Gaming Chair"
youtube_df = load_YouTube_df(product, max_result=5, transcript_chunk_size=100)

zs_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def is_feedback(text):
    out = zs_classifier(
        text,
        candidate_labels=["Product feedback", "Irrelevant"],
        multi_label=False
    )
    return out["labels"][0] == "Product feedback"

df_feedback = youtube_df[youtube_df["content"].apply(is_feedback)]

