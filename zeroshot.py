from transformers import pipeline
import pandas as pd
from youtube_tool import load_YouTube_df
from tavily_tool import webscrape 
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 




zs_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def is_feedback(text):
    
    if not isinstance(text, str) or text.strip() == "":
        return False

    text = text.strip()

    out = zs_classifier(
        text,
        candidate_labels=[
            "Feedback about product quality",
            "Joke or unrelated",
            "General praise or marketing",
            "Spam or noise"
        ],
        multi_label=False
    )

    return out["labels"][0] == "Feedback about product quality" and out["scores"][0] > 0.6

def generate_labels_from_feedback(df, n_samples=50, model="gpt-3.5-turbo", seed=42):
    """
    Generates up to 10 improvement-related labels from a sample of feedback comments.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'content' column.
        n_samples (int): Number of feedback rows to sample.
        model (str): OpenAI model name (e.g., "gpt-4" or "gpt-3.5-turbo").
        seed (int): Random seed for reproducibility.
        
    Returns:
        list[str]: List of concise feedback labels from the LLM.
    """
    # Step 1: Sample content
    if len(df) < n_samples:
        print(f"[WARN] Only {len(df)} rows available, sampling all.")
        sample = df["content"].dropna().tolist()
    else:
        sample = df["content"].dropna().sample(n=n_samples, random_state=seed).tolist()

    # Step 2: Construct prompt
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


    # Step 3: Call OpenAI
    try:
        import ast
        completion = client.chat.completions.create(
                                                    model="gpt-3.5-turbo",
                                                    messages=[
                                                        {
                                                            "role": "user",
                                                            "content": prompt
                                                        }
                                                            ],)

        print(completion.choices[0].message.content)
        labels_raw = completion.choices[0].message.content
        real_list = ast.literal_eval(labels_raw)

        return real_list

    except Exception as e:
        print("[ERROR] Failed to generate or parse labels.something went wrong. in label generation")
        print("Exception:", e)
        return []

def assign_label(text, candidate_labels):
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        out = zs_classifier(text, candidate_labels=candidate_labels, multi_label=False)
        return out["labels"][0]
    except Exception as e:
        print(f"[ERROR] classification failed: {e}")
        return None


def run_all(df, n_samples=50, model="gpt-3.5-turbo", seed=42):
    df["is_feedback"] = df["content"].apply(is_feedback)

    feedback_df = df[df["is_feedback"]]
    labels_list = generate_labels_from_feedback(
        feedback_df, n_samples=n_samples, model=model, seed=seed
    )

    # assign each row its label
    df["label"] = df.apply(
        lambda row: assign_label(row["content"], labels_list) if row["is_feedback"] else None,
        axis=1)

    return df, labels_list


if __name__ == "__main__":
    # Load the YouTube DataFrame

    search_terms = "Secretlab Titan Evo 2022 Gaming Chair"
    youtube_df = load_YouTube_df(search_terms)
    tavily_df = webscrape(search_terms)
    df = pd.concat([youtube_df, tavily_df], ignore_index=True)
    # Run the zero-shot classification and label generation
    df, labels_list = run_all(df=df)
    
    # Save the labeled DataFrame to a new pickle file
    df.to_pickle("pickle/combined_data_labeled.pkl")
    print("columns,rows",df.columns, df.shape)
    
    # Print the generated labels
    print("Generated Labels:", labels_list)