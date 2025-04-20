import pickle
import pandas as pd
from urllib.parse import urlparse
import re

def extract_website_name(url):
    try:
        netloc = urlparse(url).netloc  # e.g. 'www.techradar.com'
        netloc = re.sub(r"^www\.", "", netloc)  # remove 'www.'
        netloc = re.sub(r"\.[a-z]{2,}(?:\.[a-z]{2,})?$", "", netloc)  # remove TLDs like '.com', '.co.uk'
        return netloc
    except:
        return None

def data_processing(data):
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


# Load pickle file
with open("pickle/combined_data_labeled.pkl", "rb") as f:
    data = pickle.load(f)