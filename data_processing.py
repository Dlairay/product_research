import pickle
import pandas as pd
from urllib.parse import urlparse
import re
import os 
def extract_website_name(url):
    try:
        netloc = urlparse(url).netloc  # e.g. 'www.techradar.com'
        netloc = re.sub(r"^www\.", "", netloc)  # remove 'www.'
        netloc = re.sub(r"\.[a-z]{2,}(?:\.[a-z]{2,})?$", "", netloc)  # remove TLDs like '.com', '.co.uk'
        return netloc
    except:
        return None

def process_data(data):
    # Convert to DataFrame
    df = pd.DataFrame(data)

 
    df = df.drop(columns=["is_feedback"])    # Drop unnecessary column
    
    df = df.loc[df["label"].notna()] # Filter rows with non-null labels
    label_counts = df["label"].value_counts().to_dict()  # Count the number of each unique label
    #save to chromadb
    # print(label_counts)
    df["website"] = df["url"].apply(extract_website_name)  # Extract website names from URLs

    unique_websites = df["website"].unique()
    #save to chromadb
    # print(unique_websites)
    dict_out = {
        "df": df,
        "label_counts": label_counts,
        "unique_websites": unique_websites
    }
    return dict_out

def traceback(label=str,df = pd.DataFrame, n=5):
    """print the first n rows of the dataframe with the given label"""
    filtered_df = df.loc[df["label"] == label]
    print(f"{df.shape[0]} rows found, showing first {n} rows: \n {filtered_df.head(n)}")
    return filtered_df

def load_and_combine_existing_dataframes(product_list, pickle_dir="pickle"):
    """
    Checks for existing labeled data pickle files for a list of products,
    loads them as pandas DataFrames, and concatenates them into a single DataFrame.
    Returns the combined DataFrame (which will be empty if no files are found or loaded).
    """
    all_dfs = []

    if not os.path.exists(pickle_dir):
        print(f"[INFO] Pickle directory '{pickle_dir}' does not exist.")
        return pd.DataFrame()

    for product in product_list:
        file_name = f"combined_{product}_data_labeled.pkl"
        file_path = os.path.join(pickle_dir, file_name)

        if os.path.exists(file_path):
            try:
                df = pd.read_pickle(file_path)
                print(f"[INFO] Loaded existing data for '{product}' from '{file_path}'.")
                all_dfs.append(df)
            except Exception as e:
                print(f"[ERROR] Could not load '{file_path}': {e}")
        else:
            print(f"[INFO] No existing data found for '{product}' at '{file_path}'.")

    if not all_dfs:
        print("[INFO] No DataFrames loaded to combine.")
        return pd.DataFrame()
    else:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"[INFO] Concatenated {len(all_dfs)} DataFrames into one with {len(combined_df)} rows.")
        return combined_df
# Load pickle file
# with open("pickle/combined_data_labeled.pkl", "rb") as f:
#     data = pickle.load(f)