import os
import json
import pandas as pd
from dotenv import load_dotenv
from transformers import pipeline
from pydantic import RootModel, ValidationError

from typing import List
from openai import OpenAI

from youtube_tool import load_YouTube_df
from tavily_tool import webscrape

# Load environment and initialize OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Pipelines ===
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
zs_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# === Pydantic Schema ===
from typing import List

class LabelListModel(RootModel[List[str]]):
    pass

# === Functions ===
def analyze_sentiment_score(text):
    """Returns sentiment score [0, 1]"""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        result = sentiment_analyzer(text[:512])[0]
        label, score = result["label"].upper(), result["score"]
        return score if label == "POSITIVE" else 1 - score if label == "NEGATIVE" else 0.5
    except ValidationError as ve:
        print(f"[ERROR] Validation error: {ve}")
        return None


def is_feedback(text):
    """Returns True if text is likely product-related feedback"""
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        out = zs_classifier(
            text.strip(),
            candidate_labels=[
                "Feedback about product quality",
                "Joke or unrelated",
                "General praise or marketing",
                "Spam or noise"
            ],
            multi_label=False
        )
        return out["labels"][0] == "Feedback about product quality" and out["scores"][0] > 0.6
    except Exception as e:
        print(f"[ERROR] Zero-shot feedback detection failed: {e}")
        return False


def generate_labels_from_feedback(df, n_samples=50, model="gpt-3.5-turbo", seed=42):
    """Extracts improvement-related labels from negative feedback using OpenAI"""
    if df.empty:
        print("[WARN] Empty feedback DataFrame.")
        return []

    sample = df["content"].dropna()
    sample = sample.sample(n=min(n_samples, len(sample)), random_state=seed).tolist()

    prompt = f"""
You are a product analyst reviewing customer feedback for a physical consumer product.

Your task is to extract up to 10 **high-quality product improvement categories** based on the user comments below.

⚠️ The labels MUST be strictly related to:
- design flaws
- functionality issues
- durability or comfort problems
- assembly difficulty
- negative user experiences

⛔ Absolutely DO NOT include labels related to:
- price, cost, value for money (e.g. "overpriced")
- color choices, branding, stock issues (e.g. "limited edition", "color options")
- customer service, delivery, packaging
- warranties or guarantees (e.g. "no comfort guarantee")
- subjective praise or joke feedback

✅ Format:
Return a JSON array of short, clear labels like:
["backrest uncomfortable", "hard to assemble", "material wears out", "too bulky", "noisy wheels"]

User comments:
{chr(10).join(f"- {c}" for c in sample)}

Only return the JSON array. No explanations.
"""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        raw_output = completion.choices[0].message.content.strip()
        parsed = json.loads(raw_output)
        
        # ✅ Validate with Pydantic v2
        validated = LabelListModel.model_validate(parsed)

        # ✅ Access the validated list
        return validated.root

    except Exception as e:
        print("❌ Error:", e)
        return []

def assign_label(text, candidate_labels):
    """Assigns the most relevant label from candidate list"""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        out = zs_classifier(text, candidate_labels=candidate_labels, multi_label=False)
        return out["labels"][0]
    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")
        return None


def run_all(df, product: str, n_samples=50, model="gpt-3.5-turbo", seed=42):
    """Pipeline: filter → sentiment → label generation → label assignment"""
    df["is_feedback"] = df["content"].apply(is_feedback)
    df["sentiment_score"] = df.apply(
        lambda row: analyze_sentiment_score(row["content"]) if row["is_feedback"] else None,
        axis=1
    )

    negative_feedback_df = df[(df["is_feedback"]) & (df["sentiment_score"] < 0.5)].copy()
    labels_list = generate_labels_from_feedback(negative_feedback_df, n_samples, model, seed)

    df["label"] = df.apply(
        lambda row: assign_label(row["content"], labels_list) if row["is_feedback"] else None,
        axis=1
    )
    df.to_pickle(f"pickle/combined_{product}_data_labeled.pkl") ###super important cache
    return df





# === Main Execution ===
if __name__ == "__main__":
    product = "Secretlab Titan Evo 2022 Gaming Chair"

    youtube_df = load_YouTube_df(product)
    tavily_df = webscrape(product)
    df = pd.concat([youtube_df, tavily_df], ignore_index=True)

    df, labels_list = run_all(df=df)
